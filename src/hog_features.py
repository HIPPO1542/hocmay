from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.feature import hog

from .preprocessing import read_image


@dataclass(frozen=True)
class HogConfig:
    image_size: tuple[int, int] = (64, 64)
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"


def grayscale_pixels(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    grayscale = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(grayscale, dtype=np.float32) / 255.0


def extract_hog(image: Image.Image, config: HogConfig = HogConfig()) -> np.ndarray:
    pixels = grayscale_pixels(image, config.image_size)
    return hog(
        pixels,
        orientations=config.orientations,
        pixels_per_cell=config.pixels_per_cell,
        cells_per_block=config.cells_per_block,
        block_norm=config.block_norm,
        feature_vector=True,
    ).astype(np.float32, copy=False)


def hog_matrix(paths, config: HogConfig) -> np.ndarray:
    return np.stack([extract_hog(read_image(path), config) for path in paths])
