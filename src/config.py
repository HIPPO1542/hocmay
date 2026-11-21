from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPLITS = ("train", "validation", "test")
DATASET_ID = "kritikseth/fruit-and-vegetable-image-recognition"
BATCH_SIZE = 32
MAX_EPOCHS = 30
CNN_SIZE = (128, 128)
MOBILENET_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MODEL_NAMES = {
    "svm": "HOG + SVM",
    "cnn": "CNN",
    "mobilenet": "MobileNetV2",
}


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "processed" / "manifest.json"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def metrics(self) -> Path:
        return self.root / "results" / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def predictions(self) -> Path:
        return self.root / "results" / "predictions"

    def model_file(self, model_key: str) -> Path:
        extension = "joblib" if model_key == "svm" else "keras"
        return self.models / f"{model_key}.{extension}"

    def metadata_file(self, model_key: str) -> Path:
        return self.models / f"{model_key}.json"
