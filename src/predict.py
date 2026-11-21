from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .config import CNN_SIZE, MOBILENET_SIZE, MODEL_NAMES, Paths
from .hog_features import HogConfig, extract_hog
from .model_artifacts import read_model_metadata
from .preprocessing import read_image, rgb_pixels
from .utils import ProjectError, run_cli, sha256_file, workspace_parser

DEEP_PREPROCESS = {
    "cnn": ("rgb_0_1", CNN_SIZE),
    "mobilenet": ("mobilenet_v2", MOBILENET_SIZE),
}


@dataclass(frozen=True)
class Prediction:
    class_index: int
    class_name: str
    probability: float | None
    top: list[tuple[str, float]]
    elapsed_ms: float


@dataclass(frozen=True)
class Predictor:
    model_key: str
    model: object
    metadata: dict

    def predict(self, image: Image.Image) -> Prediction:
        started = perf_counter()
        names = self.metadata["class_names"]
        preprocess = self.metadata["preprocess"]
        probabilities = None
        if self.model_key == "svm":
            features = extract_hog(image, HogConfig(**preprocess["hog"]))[None, :]
            class_index = int(self.model.predict(features)[0])
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
        else:
            pixels = rgb_pixels(image, tuple(preprocess["size"]))
            probabilities = self.model(pixels[None, ...], training=False).numpy()[0]
            class_index = int(np.argmax(probabilities))

        top = []
        probability = None
        if probabilities is not None:
            # Với SVC, nhãn predict có thể khác lớp đứng đầu predict_proba.
            ranking = np.argsort(-probabilities, kind="stable")[:3]
            top = [(names[index], float(probabilities[index])) for index in ranking]
            probability = float(probabilities[class_index])
        return Prediction(class_index, names[class_index], probability, top, (perf_counter() - started) * 1000)


def load_predictor(paths: Paths, model_key: str) -> Predictor:
    if model_key not in MODEL_NAMES:
        raise ProjectError(f"Mô hình không được hỗ trợ: {model_key}.")
    metadata = read_model_metadata(paths, model_key)
    artifact = paths.model_file(model_key)
    if sha256_file(artifact) != metadata.get("model_sha256"):
        raise ProjectError(f"Tệp {artifact.name} không khớp metadata. Hãy lấy lại cả hai tệp từ cùng lần huấn luyện.")

    preprocess = metadata["preprocess"]
    try:
        if model_key == "svm":
            import joblib

            if preprocess["mode"] != "hog":
                raise ValueError("preprocess không tương thích với HOG")
            # joblib dùng pickle. Chỉ nạp artifact do chính dự án tạo từ nguồn tin cậy.
            model = joblib.load(artifact)
            if list(model.classes_) != list(range(len(metadata["class_names"]))):
                raise ValueError("các chỉ số nhãn trong mô hình lệch metadata")
        else:
            from tensorflow import keras

            mode, size = DEEP_PREPROCESS[model_key]
            if preprocess["mode"] != mode or tuple(preprocess["size"]) != size:
                raise ValueError("preprocess của mô hình học sâu không hợp lệ")
            model = keras.models.load_model(artifact, compile=False)
            if model.input_shape[1:] != (*size, 3) or model.output_shape[-1] != len(metadata["class_names"]):
                raise ValueError("kích thước tensor trong mô hình lệch metadata")
    except (OSError, ValueError, TypeError, AttributeError, KeyError, ImportError, EOFError) as error:
        raise ProjectError(f"Không thể nạp {artifact.name}: {error}") from error
    return Predictor(model_key, model, metadata)


def main() -> None:
    import json

    parser = workspace_parser("Nhận diện ảnh bằng mô hình đã huấn luyện.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenet")
    args = parser.parse_args()
    image = read_image(args.image, upload=True)
    predictor = load_predictor(Paths(args.workspace.resolve()), args.model)
    print(json.dumps(asdict(predictor.predict(image)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli(main)
