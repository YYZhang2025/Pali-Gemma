from typing import Dict, List, Optional, Union, Tuple, Iterable

import numpy as np
from numpy.typing import DTypeLike

from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def rescale(
    image: np.ndarray, scale: float = 1 / 255.0, dtype: DTypeLike = np.float32
) -> np.ndarray:

    image = (image * scale).astype(dtype)

    return image


def resize(
    image: Image.Image,
    size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:

    height, width = size
    resized_image = image.resize((width, height), resample=resample)

    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std

    return image


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    rescale_factor: float = 1 / 255.0,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    resized_images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    resized_images = [np.array(image) for image in resized_images]
    # Rescale the pixel values to be in the range [0, 1]
    rescaled_images = [rescale(image, scale=rescale_factor) for image in resized_images]
    # Normalize the images to have mean 0 and standard deviation 1
    normalized_images = [
        normalize(image, mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD)
        for image in rescaled_images
    ]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    normalized_images = [image.transpose(2, 0, 1) for image in normalized_images]
    return normalized_images
