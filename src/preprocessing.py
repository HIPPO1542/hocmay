from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .utils import ProjectError

UPLOAD_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_PIXELS = 32_000_000


def read_image(source: Path | BinaryIO, *, upload: bool = False) -> Image.Image:
    try:
        with Image.open(source) as image:
            if upload and image.format not in UPLOAD_FORMATS:
                raise ProjectError("Ảnh tải lên phải có định dạng JPEG hoặc PNG.")
            if upload and image.width * image.height > MAX_UPLOAD_PIXELS:
                raise ProjectError("Ảnh tải lên vượt quá 32 megapixel.")
            # Thống nhất chiều ảnh theo EXIF và thứ tự kênh giữa huấn luyện và demo.
            oriented = ImageOps.exif_transpose(image)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                return background
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ProjectError(f"Không đọc được ảnh: {error}") from error


def rgb_pixels(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize theo (chiều cao, chiều rộng), trả pixel RGB float32 trong [0, 255]."""
    height, width = size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def image_dataset(paths: list[Path], labels: list[int], size: tuple[int, int], batch_size: int, *, training: bool):
    import tensorflow as tf

    from .config import SEED

    def read_pixels(encoded_path):
        return rgb_pixels(read_image(Path(encoded_path.decode("utf-8"))), size)

    def preprocess(encoded_path, label):
        pixels = tf.numpy_function(read_pixels, [encoded_path], tf.float32)
        pixels.set_shape((*size, 3))
        return pixels, label

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in paths], np.asarray(labels, dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    dataset = dataset.with_options(options)
    # Chỉ pipeline train xáo trộn. Các lớp augmentation trong model hoạt động khi fit.
    return dataset.map(preprocess, num_parallel_calls=4).batch(batch_size).prefetch(1)
