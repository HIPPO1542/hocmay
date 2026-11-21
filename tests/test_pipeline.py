from io import BytesIO
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
import pytest

from src.config import CNN_SIZE, MOBILENET_SIZE, Paths
from src.dataset import find_split_root, load_manifest, prepare_dataset, verify_split
from src.hog_features import extract_hog
from src.predict import load_predictor
from src.preprocessing import image_dataset, read_image, rgb_pixels
from src.smoke import create_sample_images
from src.train_cnn import build_cnn
from src.train_svm import train_svm
from src.utils import ProjectError


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Paths:
    paths = Paths(tmp_path_factory.mktemp("images"))
    create_sample_images(paths)
    broken = paths.raw / "train" / "sample_square" / "broken.jpg"
    broken.write_bytes(b"invalid jpg")
    original = paths.raw / "test" / "sample_circle" / "00.png"
    # Cùng pixel với PNG gốc dù bytes khác định dạng.
    read_image(original).save(paths.raw / "train" / "sample_circle" / "duplicate.bmp")
    prepare_dataset(paths)
    return paths


def test_reads_exif_rgb_and_resizes():
    source = BytesIO()
    image = Image.new("RGB", (32, 16), (240, 80, 15))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, format="JPEG", exif=exif)
    source.seek(0)
    decoded = read_image(source, upload=True)
    assert decoded.mode == "RGB"
    assert decoded.size == (16, 32)
    resized = rgb_pixels(decoded, (12, 14))
    assert resized.shape == (12, 14, 3)
    assert resized.dtype == np.float32


@pytest.mark.parametrize("content", [b"invalid image", b""])
def test_invalid_upload(content):
    with pytest.raises(ProjectError, match="Không đọc được ảnh"):
        read_image(BytesIO(content), upload=True)


def test_unsupported_upload_format():
    source = BytesIO()
    Image.new("RGB", (10, 10)).save(source, format="GIF")
    source.seek(0)
    with pytest.raises(ProjectError, match="JPEG hoặc PNG"):
        read_image(source, upload=True)


def test_transparent_png_composites_on_white():
    source = BytesIO()
    Image.new("RGBA", (10, 10), (10, 40, 80, 0)).save(source, format="PNG")
    source.seek(0)
    assert read_image(source, upload=True).getpixel((0, 0)) == (255, 255, 255)


def test_hog_features(prepared):
    sample = prepared.raw / "train" / "sample_triangle" / "00.png"
    features = extract_hog(read_image(sample))
    assert features.shape == (1764,)
    assert np.all(np.isfinite(features))
    assert np.max(features) > 0


def test_label_mapping_and_duplicate_policy(prepared):
    manifest = load_manifest(prepared)
    assert manifest.class_to_index == {"sample_circle": 0, "sample_square": 1, "sample_triangle": 2}
    assert set(manifest.for_split(prepared, "test")[1]) == {0, 1, 2}
    assert not any(entry["path"].endswith("duplicate.bmp") for entry in manifest.images)
    assert len(manifest.for_split(prepared, "train")[0]) == 24
    summary = json.loads((prepared.metrics / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["statuses"]["invalid"] == 1
    assert summary["cross_split_duplicate_groups"] == 1
    assert summary["kept_by_split"] == {"train": 24, "validation": 9, "test": 9}


def test_nested_split_root(tmp_path):
    paths = Paths(tmp_path)
    for split in ("train", "validation", "test"):
        (paths.raw / "archive" / split).mkdir(parents=True)
    assert find_split_root(paths.raw) == paths.raw / "archive"


def test_validation_repaired_from_train_only_when_missing_a_class(tmp_path):
    paths = Paths(tmp_path)
    create_sample_images(paths)
    for test_image in (paths.raw / "test" / "sample_circle").glob("*.png"):
        copyfile(test_image, paths.raw / "validation" / "sample_circle" / test_image.name)
    summary = prepare_dataset(paths)
    manifest = load_manifest(paths)
    moved = [
        entry for entry in manifest.images
        if entry["source_split"] != entry["split"]
    ]
    assert summary["kept_by_original_split"] == {"train": 24, "validation": 6, "test": 9}
    assert summary["kept_by_split"] == {"train": 23, "validation": 7, "test": 9}
    assert len(moved) == 1
    assert moved[0]["source_split"] == "train" and moved[0]["split"] == "validation"
    assert moved[0]["class_index"] == manifest.class_to_index["sample_circle"]
    assert len({entry["pixel_hash"] for entry in manifest.images}) == len(manifest.images)
    assert prepare_dataset(paths)["dataset_id"] == manifest.dataset_id


def test_changed_image_rejected(tmp_path):
    paths = Paths(tmp_path)
    create_sample_images(paths)
    prepare_dataset(paths)
    manifest = load_manifest(paths)
    train_paths, _ = manifest.for_split(paths, "train")
    copyfile(train_paths[1], train_paths[0])
    with pytest.raises(ProjectError, match="Ảnh đã thay đổi"):
        verify_split(paths, manifest, "train")


@pytest.mark.parametrize("size", [CNN_SIZE, MOBILENET_SIZE])
def test_tensor_shape_and_pillow_inference_parity(prepared, size):
    paths, indices = load_manifest(prepared).for_split(prepared, "validation")
    tensor, labels = next(iter(image_dataset(paths, indices, size, 4, training=False)))
    assert tuple(tensor.shape) == (4, *size, 3)
    np.testing.assert_array_equal(tensor[0].numpy(), rgb_pixels(read_image(paths[0]), size))
    np.testing.assert_array_equal(labels.numpy(), indices[:4])


def test_cnn_normalization_and_inference_without_augmentation():
    cnn = build_cnn(3)
    pixels = np.full((1, *CNN_SIZE, 3), 255, dtype=np.float32)
    augmented = cnn.get_layer("augmentation")(pixels, training=False)
    np.testing.assert_array_equal(augmented.numpy(), pixels)
    np.testing.assert_allclose(cnn.get_layer("normalize_0_1")(pixels).numpy(), 1.0)
    scores = cnn(pixels, training=False).numpy()
    assert scores.shape == (1, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)


@pytest.mark.parametrize("probability", [False, True])
def test_svm_train_reload_and_predict(prepared, probability):
    train_svm(prepared, probability=probability)
    predictor = load_predictor(prepared, "svm")
    path = prepared.raw / "test" / "sample_triangle" / "00.png"
    output = predictor.predict(read_image(path, upload=True))
    assert output.class_name in predictor.metadata["class_names"]
    assert output.elapsed_ms >= 0
    if probability:
        assert len(output.top) == 3
        assert output.probability == dict(output.top)[output.class_name]
        assert sum(score for _, score in output.top) == pytest.approx(1.0)
    else:
        assert output.top == []
        assert output.probability is None


def test_missing_model(tmp_path):
    with pytest.raises(ProjectError, match="Chưa có mô hình"):
        load_predictor(Paths(tmp_path), "cnn")


def test_streamlit_missing_models_and_comparison(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRUIT_VEG_WORKSPACE", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception
    assert "Chưa có mô hình HOG + SVM" in app.warning[0].value
    app.radio[0].set_value("So sánh mô hình").run()
    assert not app.exception
    assert "Chưa có bảng so sánh" in app.info[0].value
