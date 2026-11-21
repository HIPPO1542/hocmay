from contextlib import nullcontext
from importlib.metadata import version
from math import ceil
import os
import platform
from time import perf_counter
from uuid import uuid4

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from .config import MODEL_NAMES, Paths
from .dataset import Manifest, load_manifest, verify_split
from .eda import plt, save_figure
from .predict import load_predictor
from .preprocessing import read_image
from .utils import ProjectError, run_cli, workspace_parser, write_csv, write_json


def save_confusion(paths: Paths, model_key: str, names: list[str], matrix: np.ndarray) -> None:
    count = len(names)
    size = max(6, count * 0.31 + 2.5)
    figure, axis = plt.subplots(figsize=(size, size), layout="constrained")
    display = axis.imshow(matrix, cmap="Greens", interpolation="nearest")
    axis.set(xticks=range(count), yticks=range(count), yticklabels=names,
             xlabel="Nhãn dự đoán", ylabel="Nhãn thực",
             title=f"Ma trận nhầm lẫn · {MODEL_NAMES[model_key]}")
    axis.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    axis.tick_params(axis="y", labelsize=8)
    if count <= 12:
        for row, column in np.ndindex(matrix.shape):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "#173d31")
    figure.colorbar(display, ax=axis, shrink=0.75, label="Số ảnh")
    save_figure(figure, paths.figures / f"{model_key}_confusion_matrix.png")


def save_error_gallery(paths: Paths, model_key: str, mistakes: list[dict], pairs: list[dict]) -> None:
    destination = paths.figures / f"{model_key}_mistakes.png"
    if not mistakes:
        destination.unlink(missing_ok=True)
        return
    # Mỗi cặp nhầm thường gặp có một ảnh minh họa trước khi bổ sung các lỗi khác.
    representative = {}
    for item in mistakes:
        representative.setdefault((item["true_label"], item["predicted_label"]), item)
    examples = [representative[pair["true_label"], pair["predicted_label"]] for pair in pairs[:16]]
    selected_paths = {item["path"] for item in examples}
    examples.extend(item for item in mistakes if item["path"] not in selected_paths)
    examples = examples[:16]
    columns = min(4, len(examples))
    figure, axes = plt.subplots(ceil(len(examples) / columns), columns, figsize=(3 * columns, 3 * ceil(len(examples) / columns)),
                                squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for axis, item in zip(axes.flat, examples):
        image = read_image(paths.raw / item["path"])
        image.thumbnail((350, 350))
        axis.imshow(image)
        axis.set_title(f"Thực: {item['true_label']}\nDự đoán: {item['predicted_label']}", fontsize=9)
    figure.suptitle(f"Ảnh dự đoán sai · {MODEL_NAMES[model_key]}")
    save_figure(figure, destination)


def evaluate_model(paths: Paths, model_key: str, manifest: Manifest, evaluation_id: str, environment: dict) -> dict:
    if model_key == "svm":
        device = nullcontext()
    else:
        import tensorflow as tf

        device = tf.device("/CPU:0")

    names = manifest.class_names
    image_paths, actual = manifest.for_split(paths, "test")
    records = []
    with device:
        predictor = load_predictor(paths, model_key)
        metadata = predictor.metadata
        if metadata["dataset_id"] != manifest.dataset_id or metadata["class_names"] != names:
            raise ProjectError(f"{model_key} được huấn luyện trên manifest khác. Hãy huấn luyện lại hoặc dùng đúng bộ dữ liệu.")
        # Khởi động trên tối đa ba ảnh validation để đo latency test khi model đã nạp.
        warmup_paths, _ = manifest.for_split(paths, "validation")
        for path in warmup_paths[:3]:
            predictor.predict(read_image(path))
        for path, label_index in zip(image_paths, actual):
            started = perf_counter()
            prediction = predictor.predict(read_image(path))
            elapsed_ms = (perf_counter() - started) * 1000
            records.append({
                "path": path.relative_to(paths.raw).as_posix(),
                "true_label": names[label_index],
                "predicted_label": prediction.class_name,
                "predicted_index": prediction.class_index,
                "correct": prediction.class_index == label_index,
                "predicted_probability": prediction.probability,
                "inference_ms": elapsed_ms,
            })

    predicted = [record["predicted_index"] for record in records]
    indices = list(range(len(names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=indices, average="macro", zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=indices)
    report = classification_report(actual, predicted, labels=indices, target_names=names,
                                   output_dict=True, zero_division=0)
    # Không phụ thuộc các khóa tổng hợp thay đổi theo phiên bản sklearn.
    pairs = [
        {"true_label": names[row], "predicted_label": names[column], "count": int(matrix[row, column])}
        for row, column in np.ndindex(matrix.shape) if row != column and matrix[row, column] > 0
    ]
    pairs.sort(key=lambda pair: -pair["count"])
    mistakes = [record for record in records if not record["correct"]]
    latencies = [record["inference_ms"] for record in records]
    summary = {
        "model_key": model_key, "model": MODEL_NAMES[model_key],
        "dataset_id": manifest.dataset_id, "evaluation_id": evaluation_id,
        "model_sha256": metadata["model_sha256"], "test_count": len(actual), "error_count": len(mistakes),
        "accuracy": float(accuracy_score(actual, predicted)), "macro_precision": float(precision),
        "macro_recall": float(recall), "macro_f1": float(f1),
        "train_seconds": metadata["train_seconds"], "inference_ms": float(np.mean(latencies)),
        "p95_inference_ms": float(np.percentile(latencies, 95)),
        "model_size_bytes": paths.model_file(model_key).stat().st_size,
        "environment": environment,
    }
    write_csv(paths.predictions / f"{model_key}_test.csv", list(records[0]), records)
    write_csv(paths.predictions / f"{model_key}_mistakes.csv", list(records[0]), mistakes)
    write_csv(paths.metrics / f"{model_key}_confusion_pairs.csv", ["true_label", "predicted_label", "count"], pairs)
    write_json(paths.metrics / f"{model_key}_classification_report.json", report)
    write_json(paths.metrics / f"{model_key}_confusion_matrix.json", {"labels": names, "matrix": matrix.tolist()})
    save_confusion(paths, model_key, names, matrix)
    save_error_gallery(paths, model_key, mistakes, pairs)
    write_json(paths.metrics / f"{model_key}_test_metrics.json", summary)
    return summary


def evaluate_models(paths: Paths, model_keys: list[str] | None = None) -> list[dict]:
    keys = list(dict.fromkeys(model_keys or MODEL_NAMES))
    manifest = load_manifest(paths)
    absent = [key for key in keys if not paths.model_file(key).is_file() or not paths.metadata_file(key).is_file()]
    if absent:
        raise ProjectError(f"Cần huấn luyện các mô hình sau trước khi đánh giá: {', '.join(absent)}.")
    verify_split(paths, manifest, "test")
    evaluation_id = uuid4().hex
    environment = {
        "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(), "device": "CPU", "batch_size": 1,
        "logical_cpus": os.cpu_count(),
        "tensorflow_threads": {
            "intraop": os.environ.get("TF_NUM_INTRAOP_THREADS"),
            "interop": os.environ.get("TF_NUM_INTEROP_THREADS"),
        },
        "versions": {name: version(name) for name in ("tensorflow", "scikit-learn", "numpy", "Pillow")},
        "measurement": "read, decode, preprocessing, prediction and probability ranking; excludes model loading",
    }
    return [evaluate_model(paths, key, manifest, evaluation_id, environment) for key in keys]


def main() -> None:
    parser = workspace_parser("Đánh giá checkpoint đã chốt trên tập test sạch.")
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", help="Mặc định đánh giá cả ba trong cùng phiên.")
    args = parser.parse_args()
    for summary in evaluate_models(Paths(args.workspace.resolve()), args.models):
        print(f"{summary['model']}: Accuracy={summary['accuracy']:.4f}, Macro F1={summary['macro_f1']:.4f}, "
              f"CPU={summary['inference_ms']:.2f} ms/ảnh, n={summary['test_count']}")


if __name__ == "__main__":
    run_cli(main)
