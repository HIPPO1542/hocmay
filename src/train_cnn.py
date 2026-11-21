from .config import BATCH_SIZE, CNN_SIZE, MAX_EPOCHS, Paths
from .train_deep import augmentation, train_deep
from .utils import run_cli, workspace_parser


def build_cnn(num_classes: int):
    from tensorflow import keras

    return keras.Sequential([
        keras.Input((*CNN_SIZE, 3)),
        augmentation(),
        keras.layers.Rescaling(1 / 255.0, name="normalize_0_1"),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="scratch_cnn")


def train_cnn(paths: Paths, *, epochs: int = MAX_EPOCHS, batch_size: int = BATCH_SIZE) -> dict:
    return train_deep(paths, "cnn", CNN_SIZE, "rgb_0_1", build_cnn, epochs=epochs, batch_size=batch_size)


def main() -> None:
    parser = workspace_parser("Huấn luyện CNN từ đầu.")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    paths = Paths(args.workspace.resolve())
    train_cnn(paths, epochs=args.epochs, batch_size=args.batch_size)
    print(f"Đã lưu {paths.model_file('cnn')}")


if __name__ == "__main__":
    run_cli(main)
