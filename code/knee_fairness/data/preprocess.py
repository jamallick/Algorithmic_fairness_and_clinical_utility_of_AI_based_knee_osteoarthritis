from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional


@dataclass(frozen=True)
class ImageStatistics:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    p01: float
    p99: float


def image_statistics(image: NDArray[np.float32]) -> ImageStatistics:
    if image.ndim != 2:
        raise ValueError("Radiograph must be two-dimensional")
    if not np.all(np.isfinite(image)):
        raise ValueError("Radiograph contains non-finite values")
    return ImageStatistics(
        float(image.min()),
        float(image.max()),
        float(image.mean()),
        float(image.std()),
        float(np.quantile(image, 0.01)),
        float(np.quantile(image, 0.99)),
    )


def percentile_window(
    image: NDArray[np.float32],
    lower: float = 0.01,
    upper: float = 0.99,
) -> NDArray[np.float32]:
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Percentile window is invalid")
    minimum, maximum = np.quantile(image, [lower, upper])
    if maximum <= minimum:
        return np.zeros_like(image)
    clipped = np.clip(image, minimum, maximum)
    return ((clipped - minimum) / (maximum - minimum)).astype(np.float32)


def standardize(image: NDArray[np.float32], epsilon: float = 1e-6) -> NDArray[np.float32]:
    deviation = float(image.std())
    if deviation < epsilon:
        return np.zeros_like(image)
    return ((image - image.mean()) / deviation).astype(np.float32)


def minmax_scale(image: NDArray[np.float32], epsilon: float = 1e-6) -> NDArray[np.float32]:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum - minimum < epsilon:
        return np.zeros_like(image)
    return ((image - minimum) / (maximum - minimum)).astype(np.float32)


def resize_tensor(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if image.ndim not in {2, 3, 4}:
        raise ValueError("Image tensor rank is invalid")
    original_rank = image.ndim
    if original_rank == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif original_rank == 3:
        image = image.unsqueeze(0)
    resized = functional.interpolate(image, size=size, mode="bilinear", align_corners=False)
    if original_rank == 2:
        return resized[0, 0]
    if original_rank == 3:
        return resized[0]
    return resized


def center_crop(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    source_height, source_width = image.shape[-2:]
    if height > source_height or width > source_width:
        raise ValueError("Crop exceeds source dimensions")
    top = (source_height - height) // 2
    left = (source_width - width) // 2
    return image[..., top : top + height, left : left + width]


def pad_to_square(image: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    height, width = image.shape[-2:]
    if height == width:
        return image
    target = max(height, width)
    vertical = target - height
    horizontal = target - width
    padding = (
        horizontal // 2,
        horizontal - horizontal // 2,
        vertical // 2,
        vertical - vertical // 2,
    )
    return functional.pad(image, padding, value=value)


def split_bilateral(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = image.shape[-1]
    if width < 2:
        raise ValueError("Bilateral image is too narrow")
    midpoint = width // 2
    return image[..., :midpoint], image[..., midpoint:]


def normalize_orientation(
    image: torch.Tensor,
    knee: str,
) -> torch.Tensor:
    normalized = knee.lower()
    if normalized == "left":
        return image
    if normalized == "right":
        return torch.flip(image, dims=(-1,))
    raise ValueError("Knee must be left or right")


def quality_flags(image: NDArray[np.float32]) -> tuple[str, ...]:
    statistics = image_statistics(image)
    flags: list[str] = []
    if statistics.standard_deviation < 1e-3:
        flags.append("low_contrast")
    if statistics.maximum == statistics.minimum:
        flags.append("constant_image")
    saturated_low = float(np.mean(image <= statistics.minimum))
    saturated_high = float(np.mean(image >= statistics.maximum))
    if saturated_low > 0.05:
        flags.append("lower_saturation")
    if saturated_high > 0.05:
        flags.append("upper_saturation")
    if image.shape[0] < 128 or image.shape[1] < 128:
        flags.append("low_resolution")
    return tuple(flags)
