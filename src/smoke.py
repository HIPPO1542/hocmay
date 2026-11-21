"""Chạy workflow với ảnh tổng hợp nhỏ. Các artifact nằm ngoài thư mục thực nghiệm."""
from pathlib import Path
from tempfile import mkdtemp

from PIL import Image, ImageDraw

from .config import SEED, SPLITS, Paths
from .dataset import prepare_dataset
from .utils import run_cli, workspace_parser, write_json


def create_sample_images(paths: Paths) -> None:
    styles = {
        "sample_circle": ((215, 62, 66), "ellipse"),
        "sample_square": ((46, 131, 85), "rectangle"),
        "sample_triangle": ((49, 93, 182), "polygon"),
    }
    for split_index, split in enumerate(SPLITS):
        for name, (color, shape) in styles.items():
            folder = paths.raw / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(8 if split == "train" else 3):
                image = Image.new("RGB", (92, 76), (242 - 6 * split_index, 239, 230 + index))
                draw = ImageDraw.Draw(image)
                offset = 9 + index + split_index
                geometry = (
                    [(offset, 55), (offset + 23, 13), (offset + 46, 55)]
                    if shape == "polygon" else (offset, 12, offset + 47, 56)
                )
                getattr(draw, shape)(geometry, fill=color)
                image.save(folder / f"{index:02}.png")


def smoke_test(root: Path, *, include_mobilenet: bool = False, weights: Path | None = None) -> Paths:
    from .compare_models import compare_models
    from .evaluate import evaluate_models
    from .predict import load_predictor
    from .preprocessing import read_image
    from .train_cnn import train_cnn
    from .train_mobilenet import train_mobilenet
    from .train_svm import train_svm
    from .utils import ProjectError

    output = root / ".smoke"
    output.mkdir(parents=True, exist_ok=True)
    paths = Paths(Path(mkdtemp(prefix="run-", dir=output)))
    print(f"Smoke test · thư mục riêng: {paths.root}", flush=True)
    write_json(paths.root / ".smoke_run.json", {"synthetic_images": True, "seed": SEED})
    create_sample_images(paths)
    prepare_dataset(paths)
    train_svm(paths, probability=True)
    train_cnn(paths, epochs=1, batch_size=4)
    keys = ["svm", "cnn"]
    if include_mobilenet:
        train_mobilenet(paths, epochs=1, batch_size=4, fine_tune_epochs=1, weights=weights)
        keys.append("mobilenet")
    evaluate_models(paths, keys)
    compare_models(paths, keys)

    sample = paths.raw / "test" / "sample_circle" / "00.png"
    for key in keys:
        predictor = load_predictor(paths, key)
        prediction = predictor.predict(read_image(sample, upload=True))
        if prediction.class_name not in predictor.metadata["class_names"]:
            raise ProjectError(f"Smoke test {key} trả về nhãn ngoài danh sách.")
        if len(prediction.top) != 3 or not 0 <= prediction.probability <= 1:
            raise ProjectError(f"Smoke test {key} trả về xác suất không hợp lệ.")
    print(f"Smoke test đã chạy thành công cho {', '.join(keys)}. Dữ liệu tổng hợp chỉ dùng để kiểm tra.")
    return paths


def main() -> None:
    parser = workspace_parser("Smoke test trên dữ liệu ảnh tổng hợp.")
    parser.add_argument("--include-mobilenet", action="store_true", help="Kiểm thử thêm MobileNetV2 + fine-tuning.")
    parser.add_argument("--weights", type=Path, help="Tệp trọng số ImageNet MobileNetV2 đã tải.")
    args = parser.parse_args()
    smoke_test(args.workspace.resolve(), include_mobilenet=args.include_mobilenet or args.weights is not None,
               weights=args.weights)


if __name__ == "__main__":
    run_cli(main)
