from pathlib import Path

from .config import BATCH_SIZE, MOBILENET_SIZE, Paths
from .train_deep import augmentation, train_deep
from .utils import ProjectError, run_cli, workspace_parser

INITIAL_EPOCHS = 15
BACKBONE_NAME = "mobilenet_backbone"


def build_mobilenet(num_classes: int, weights: Path | None = None):
    from tensorflow import keras

    try:
        backbone = keras.applications.MobileNetV2(
            input_shape=(*MOBILENET_SIZE, 3), include_top=False,
            weights=str(weights) if weights is not None else "imagenet",
            name=BACKBONE_NAME,
        )
    except Exception as error:
        # Keras bọc lỗi tải qua mạng trong Exception chung.
        raise ProjectError(
            "Không nạp được trọng số ImageNet cho MobileNetV2. Kiểm tra mạng hoặc truyền "
            "--weights đến tệp ImageNet no-top alpha=1.0, 224 chính thức (xem README). "
            f"Chi tiết: {error}"
        ) from error

    backbone.trainable = False
    inputs = keras.Input((*MOBILENET_SIZE, 3))
    augmented = augmentation()(inputs)
    # Tương đương MobileNetV2 preprocess_input với pixel RGB float32 trong [0, 255].
    normalized = keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess_input")(augmented)
    # Giữ BatchNormalization ở chế độ suy luận, kể cả khi bật fine-tuning.
    features = backbone(normalized, training=False)
    pooled = keras.layers.GlobalAveragePooling2D()(features)
    dropped = keras.layers.Dropout(0.3)(pooled)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(dropped)
    return keras.Model(inputs, outputs, name="transfer_mobilenetv2")


def unfreeze_backbone(model, layer_count: int) -> None:
    from tensorflow import keras

    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    frozen_count = len(backbone.layers) - layer_count
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= frozen_count and not isinstance(layer, keras.layers.BatchNormalization)


def train_mobilenet(
    paths: Paths, *, epochs: int = INITIAL_EPOCHS, batch_size: int = BATCH_SIZE,
    fine_tune_epochs: int = 0, fine_tune_layers: int = 30, weights: Path | None = None,
) -> dict:
    return train_deep(
        paths, "mobilenet", MOBILENET_SIZE, "mobilenet_v2", lambda count: build_mobilenet(count, weights),
        epochs=epochs, batch_size=batch_size, fine_tune_epochs=fine_tune_epochs,
        fine_tune_layers=fine_tune_layers, unfreeze=unfreeze_backbone,
        training_details={"backbone": "MobileNetV2", "weights": "local_imagenet" if weights else "imagenet"},
    )


def main() -> None:
    parser = workspace_parser("Huấn luyện MobileNetV2 từ trọng số ImageNet.")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--weights", type=Path, help="Tệp ImageNet no-top alpha 1.0, 224 đã tải.")
    args = parser.parse_args()
    paths = Paths(args.workspace.resolve())
    train_mobilenet(paths, epochs=args.epochs, batch_size=args.batch_size, weights=args.weights,
                    fine_tune_epochs=args.fine_tune_epochs, fine_tune_layers=args.fine_tune_layers)
    print(f"Đã lưu {paths.model_file('mobilenet')}")


if __name__ == "__main__":
    run_cli(main)
