from math import ceil

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import SPLITS, Paths
from .preprocessing import read_image

COLORS = ("#23745e", "#df9b3c", "#56748f")


def save_figure(figure, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_eda_figures(paths: Paths, summary: dict, class_counts: list[dict], kept: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(10, max(4, 0.36 * len(class_counts))))
    names = [row["class_name"] for row in class_counts]
    left = np.zeros(len(names))
    for split, color in zip(SPLITS, COLORS):
        counts = [row[split] for row in class_counts]
        axis.barh(names, counts, left=left, label=split, color=color)
        left += counts
    axis.set(xlabel="Số ảnh hợp lệ", title="Phân bố ảnh theo lớp và tập")
    axis.invert_yaxis()
    axis.legend()
    save_figure(figure, paths.figures / "eda_class_counts.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(SPLITS))
    axis.bar(positions - 0.18, [summary["candidates_by_split"][split] for split in SPLITS],
             width=0.36, label="Trước kiểm tra", color="#b8c3c9")
    bars = axis.bar(positions + 0.18, [summary["kept_by_split"][split] for split in SPLITS],
                    width=0.36, label="Dùng cho thực nghiệm", color=COLORS[0])
    axis.bar_label(bars, padding=2)
    axis.set(xticks=positions, xticklabels=SPLITS, ylabel="Số ảnh", title="Phân bố theo tập")
    axis.margins(y=0.23)
    axis.legend()
    save_figure(figure, paths.figures / "eda_splits.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, key, label in zip(axes, ("width", "height"), ("Chiều rộng", "Chiều cao")):
        axis.hist([record[key] for record in kept], bins=30, color=COLORS[0])
        axis.set(xlabel=f"{label} (pixel)", ylabel="Số ảnh")
    figure.suptitle("Kích thước ảnh gốc sau hiệu chỉnh EXIF")
    save_figure(figure, paths.figures / "eda_dimensions.png")

    samples = {}
    for record in kept:
        if record["effective_split"] == "train":
            samples.setdefault(record["class_name"], record["path"])
    columns = min(6, len(samples))
    rows = ceil(len(samples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2.25 * columns, 2.25 * rows),
                                squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for axis, (name, path) in zip(axes.flat, samples.items()):
        image = read_image(paths.raw / path)
        image.thumbnail((280, 280))
        axis.imshow(image)
        axis.set_title(name, fontsize=9)
    save_figure(figure, paths.figures / "eda_samples.png")
