from .config import MODEL_NAMES, Paths
from .dataset import load_manifest
from .model_artifacts import read_model_metadata
from .utils import ProjectError, read_json, run_cli, workspace_parser, write_csv, write_json

COMPARISON_FIELDS = [
    "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "train_seconds", "inference_ms", "model_size_mib",
]
TRADEOFFS = {
    "svm": "HOG cho baseline dễ giải thích và chạy trên CPU. Grayscale bỏ thông tin màu. "
           "SVC có thể tốn thời gian khi số ảnh tăng.",
    "cnn": "CNN học đặc trưng RGB trực tiếp và tạo checkpoint gọn. Khi train từ đầu, "
           "chất lượng phụ thuộc lượng ảnh và độ đa dạng của train.",
    "mobilenet": "MobileNetV2 tận dụng đặc trưng ImageNet và có thể tinh chỉnh backbone. "
                 "Cần trọng số ban đầu và dùng ảnh đầu vào 224×224.",
}


def current_metrics(paths: Paths, model_keys: list[str]) -> list[dict]:
    manifest = load_manifest(paths)
    rows = []
    for key in model_keys:
        metric_path = paths.metrics / f"{key}_test_metrics.json"
        if not metric_path.is_file():
            raise ProjectError(f"Chưa có metrics của {MODEL_NAMES[key]}. Chạy src.evaluate trước.")
        row = read_json(metric_path)
        metadata = read_model_metadata(paths, key)
        if row["dataset_id"] != manifest.dataset_id or row["model_sha256"] != metadata["model_sha256"]:
            raise ProjectError(f"Metrics của {MODEL_NAMES[key]} thuộc bộ dữ liệu hoặc checkpoint cũ. Hãy đánh giá lại.")
        rows.append(row)
    if len({row["evaluation_id"] for row in rows}) != 1:
        raise ProjectError("Các phép đo đến từ nhiều phiên. Chạy src.evaluate một lần cho toàn bộ mô hình cần so sánh.")
    return rows


def compare_models(paths: Paths, model_keys: list[str] | None = None) -> dict:
    from .eda import plt, save_figure

    keys = list(dict.fromkeys(model_keys or MODEL_NAMES))
    metrics = current_metrics(paths, keys)
    rows = []
    for item in metrics:
        row = {field: item[field] for field in COMPARISON_FIELDS if field != "model_size_mib"}
        row["model_size_mib"] = item["model_size_bytes"] / (1024 ** 2)
        rows.append(row)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1")):
        bars = axis.bar([row["model"] for row in rows], [row[metric] for row in rows], color="#23745e")
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set(ylim=(0, 1.12), title=title)
        axis.grid(axis="y", alpha=0.2)
    save_figure(figure, paths.figures / "model_comparison.png")

    best = max(metrics, key=lambda row: (row["macro_f1"], row["accuracy"], -row["inference_ms"]))
    fastest = min(metrics, key=lambda row: row["inference_ms"])
    smallest = min(metrics, key=lambda row: row["model_size_bytes"])
    payload = {
        "dataset_id": metrics[0]["dataset_id"], "evaluation_id": metrics[0]["evaluation_id"],
        "model_keys": keys, "test_count": metrics[0]["test_count"],
        "environment": metrics[0]["environment"], "rows": rows,
        "tradeoffs": {MODEL_NAMES[key]: TRADEOFFS[key] for key in keys},
        "recommendation": (
            f"Đề xuất {best['model']} nếu ưu tiên Macro F1 "
            f"({best['macro_f1']:.4f}, Accuracy {best['accuracy']:.4f}). "
            f"Trên cùng phiên đo CPU, {fastest['model']} có độ trễ trung bình thấp nhất "
            f"({fastest['inference_ms']:.2f} ms/ảnh) và {smallest['model']} có tệp nhỏ nhất "
            f"({smallest['model_size_bytes'] / (1024 ** 2):.2f} MiB)."
        ),
    }
    write_csv(paths.metrics / "model_comparison.csv", COMPARISON_FIELDS, rows)
    write_json(paths.metrics / "model_comparison.json", payload)
    return payload


def load_comparison(paths: Paths) -> dict:
    path = paths.metrics / "model_comparison.json"
    if not path.is_file():
        raise ProjectError("Chưa có bảng so sánh. Hãy chạy đánh giá và tạo bảng so sánh.")
    comparison = read_json(path)
    metrics = current_metrics(paths, comparison["model_keys"])
    if comparison["evaluation_id"] != metrics[0]["evaluation_id"]:
        raise ProjectError("Đã có phiên đánh giá mới. Hãy tạo lại bảng so sánh.")
    return comparison


def main() -> None:
    parser = workspace_parser("So sánh metrics của các mô hình trong cùng phiên đánh giá.")
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", help="Mặc định so sánh cả ba mô hình.")
    args = parser.parse_args()
    paths = Paths(args.workspace.resolve())
    comparison = compare_models(paths, args.models)
    print(comparison["recommendation"])
    print(f"Đã lưu {paths.metrics / 'model_comparison.csv'}")


if __name__ == "__main__":
    run_cli(main)
