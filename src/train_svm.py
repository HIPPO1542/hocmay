from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED, Paths
from .dataset import load_manifest, verify_split
from .hog_features import HogConfig, hog_matrix
from .model_artifacts import publish_model
from .utils import run_cli, workspace_parser, write_csv

PARAMETER_GRID = [
    {"kernel": ["linear"], "C": [0.1, 1.0, 10.0], "gamma": ["scale"]},
    {"kernel": ["rbf"], "C": [1.0, 10.0], "gamma": ["scale", 0.0001, 0.001]},
]


def svm_pipeline(parameters: dict, *, probability: bool) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**parameters, probability=probability, random_state=SEED, cache_size=256)),
    ])


def train_svm(paths: Paths, *, probability: bool = True) -> dict:
    manifest = load_manifest(paths)
    verify_split(paths, manifest, "train")
    verify_split(paths, manifest, "validation")
    train_paths, train_labels = manifest.for_split(paths, "train")
    val_paths, val_labels = manifest.for_split(paths, "validation")
    config = HogConfig()

    started = perf_counter()
    train_features = hog_matrix(train_paths, config)
    val_features = hog_matrix(val_paths, config)
    trials = []
    best_model = None
    best_rank = (-1.0, -1.0)
    best_parameters = {}
    labels = list(range(len(manifest.class_names)))
    # Scaler nằm trong Pipeline nên chỉ học trung bình và độ lệch chuẩn từ train.
    for parameters in ParameterGrid(PARAMETER_GRID):
        candidate = svm_pipeline(parameters, probability=False).fit(train_features, train_labels)
        predicted = candidate.predict(val_features)
        macro_f1 = float(f1_score(val_labels, predicted, labels=labels, average="macro", zero_division=0))
        accuracy = float(accuracy_score(val_labels, predicted))
        trials.append({**parameters, "val_macro_f1": macro_f1, "val_accuracy": accuracy})
        if (macro_f1, accuracy) > best_rank:
            best_rank = (macro_f1, accuracy)
            best_model = candidate
            best_parameters = parameters
    if probability:
        # SVC hiệu chỉnh xác suất bằng CV nội bộ trên train sau khi chốt siêu tham số.
        best_model = svm_pipeline(best_parameters, probability=True).fit(train_features, train_labels)
    duration = perf_counter() - started

    paths.models.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=paths.models) as staging:
        staged = Path(staging) / "svm.joblib"
        joblib.dump(best_model, staged, compress=3)
        metadata = publish_model(
            staged, paths, "svm", manifest, {"mode": "hog", "hog": asdict(config)}, duration,
            {
                "parameters": best_parameters,
                "selection": "validation_macro_f1",
                "val_macro_f1": best_rank[0],
                "val_accuracy": best_rank[1],
                "probability": probability,
            },
        )
    write_csv(paths.metrics / "svm_validation_search.csv",
              ["kernel", "C", "gamma", "val_macro_f1", "val_accuracy"], trials)
    return metadata


def main() -> None:
    import argparse

    parser = workspace_parser("Huấn luyện HOG + SVM, chọn siêu tham số bằng validation.")
    parser.add_argument("--probability", action=argparse.BooleanOptionalAction, default=True,
                        help="Hiệu chỉnh xác suất với CV nội bộ trên train.")
    args = parser.parse_args()
    paths = Paths(args.workspace.resolve())
    metadata = train_svm(paths, probability=args.probability)
    print(f"Đã lưu {paths.model_file('svm')}. {metadata['training']}")


if __name__ == "__main__":
    run_cli(main)
