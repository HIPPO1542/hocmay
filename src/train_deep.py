from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .config import BATCH_SIZE, MAX_EPOCHS, SEED, Paths
from .dataset import load_manifest, verify_split
from .model_artifacts import publish_model
from .preprocessing import image_dataset
from .utils import ProjectError, write_csv, write_json


def augmentation():
    from tensorflow import keras

    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(0.08, seed=SEED + 1),
        keras.layers.RandomZoom(0.1, seed=SEED + 2),
        keras.layers.RandomContrast(0.12, seed=SEED + 3),
    ], name="augmentation")


def compile_model(model, learning_rate: float) -> None:
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        jit_compile=False,
    )


def fit_phase(model, train, validation, epochs: int, checkpoint: Path, phase: str,
              learning_rate: float) -> list[dict]:
    from tensorflow import keras

    compile_model(model, learning_rate)
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(train, validation_data=validation, epochs=epochs, callbacks=callbacks).history
    required = ["accuracy", "val_accuracy", "loss", "val_loss"]
    if not checkpoint.is_file() or any(not np.isfinite(history.get(key, [np.nan])).all() for key in required):
        raise ProjectError(f"Huấn luyện {phase} không tạo được checkpoint và lịch sử hữu hạn.")
    return [
        {"phase": phase, **{key: float(history[key][index]) for key in required}}
        for index in range(len(history["val_loss"]))
    ]


def report_history(paths: Paths, model_key: str, rows: list[dict], best: dict) -> None:
    from .eda import plt, save_figure

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        axis.plot(epochs, [row[metric] for row in rows], label="Train", color="#23745e")
        axis.plot(epochs, [row[f"val_{metric}"] for row in rows], label="Validation", color="#cc852d")
        axis.axvline(best["epoch"], linestyle="--", color="#56748f", label="Checkpoint được chọn")
        axis.set(xlabel="Epoch", title=title)
        axis.legend()
    save_figure(figure, paths.figures / f"{model_key}_training.png")

    write_csv(paths.metrics / f"{model_key}_history.csv", list(rows[0]), rows)
    final = rows[-1]
    accuracy_gap = best["accuracy"] - best["val_accuracy"]
    val_loss_increase = final["val_loss"] > best["val_loss"] * 1.05
    signals = {
        "selected_epoch": best,
        "train_minus_validation_accuracy": accuracy_gap,
        "last_validation_loss": final["val_loss"],
        "possible_overfitting": accuracy_gap >= 0.10 or (
            final["loss"] < best["loss"] and val_loss_increase
        ),
        "both_accuracies_below_50_percent": max(best["accuracy"], best["val_accuracy"]) < 0.50,
        "interpretation": (
            "Đọc chênh lệch train–validation cùng đồ thị. Loss train tiếp tục giảm trong khi "
            "loss validation tăng sau checkpoint có thể là overfitting. Accuracy thấp ở cả hai "
            "tập có thể là underfitting hoặc dữ liệu khó. Các ngưỡng chỉ là chỉ báo mô tả."
        ),
    }
    write_json(paths.metrics / f"{model_key}_learning_analysis.json", signals)


def train_deep(
    paths: Paths, model_key: str, size: tuple[int, int], preprocess_mode: str, build_model: Callable,
    *, epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE, fine_tune_epochs: int = 0,
    unfreeze: Callable | None = None, fine_tune_layers: int = 30, training_details: dict | None = None,
) -> dict:
    if not 1 <= epochs <= MAX_EPOCHS or not 0 <= fine_tune_epochs <= MAX_EPOCHS:
        raise ProjectError(f"Số epoch mỗi giai đoạn phải nằm trong giới hạn {MAX_EPOCHS}.")
    if batch_size < 1 or fine_tune_layers < 1:
        raise ProjectError("Batch size và số lớp fine-tune phải là số dương.")
    if fine_tune_epochs and unfreeze is None:
        raise ProjectError("Mô hình này không hỗ trợ fine-tuning.")

    manifest = load_manifest(paths)
    verify_split(paths, manifest, "train")
    verify_split(paths, manifest, "validation")

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Notebook có thể đã khởi tạo GPU từ cell trước.
            pass
    tf.keras.utils.set_random_seed(SEED)
    train_paths, train_labels = manifest.for_split(paths, "train")
    val_paths, val_labels = manifest.for_split(paths, "validation")
    train = image_dataset(train_paths, train_labels, size, batch_size, training=True)
    validation = image_dataset(val_paths, val_labels, size, batch_size, training=False)
    model = build_model(len(manifest.class_names))

    paths.models.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=paths.models) as staging:
        checkpoints = {"initial": Path(staging) / "initial.keras", "fine_tune": Path(staging) / "fine_tune.keras"}
        started = perf_counter()
        rows = fit_phase(model, train, validation, epochs, checkpoints["initial"], "initial", 1e-3)
        if fine_tune_epochs:
            model = tf.keras.models.load_model(checkpoints["initial"], compile=False)
            unfreeze(model, fine_tune_layers)
            rows.extend(fit_phase(model, train, validation, fine_tune_epochs,
                                  checkpoints["fine_tune"], "fine_tune", 1e-5))
        duration = perf_counter() - started
        for index, row in enumerate(rows, start=1):
            row["epoch"] = index
        # Chọn checkpoint tốt nhất toàn bộ quá trình, kể cả khi fine-tuning kém hơn head.
        best = min(rows, key=lambda row: row["val_loss"])
        metadata = publish_model(
            checkpoints[best["phase"]], paths, model_key, manifest,
            {"mode": preprocess_mode, "size": list(size), "color": "RGB", "normalization_in_model": True},
            duration,
            {
                "selection": "validation_loss", "best_epoch": best, "batch_size": batch_size,
                "epochs_requested": epochs, "fine_tune_epochs_requested": fine_tune_epochs,
                "fine_tune_layers": fine_tune_layers if fine_tune_epochs else 0,
                **(training_details or {}),
            },
        )
    report_history(paths, model_key, rows, best)
    return metadata
