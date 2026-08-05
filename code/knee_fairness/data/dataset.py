from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from knee_fairness.data.manifest import read_manifest
from knee_fairness.schema import CohortRecord

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def load_manifest(path: str | Path) -> list[CohortRecord]:
    return read_manifest(Path(path))


class KneeRadiographDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[CohortRecord],
        image_size: int = 224,
        transform: TensorTransform | None = None,
        augment: bool = False,
    ) -> None:
        if not records:
            raise ValueError("Dataset needs at least one record")
        if image_size < 16:
            raise ValueError("Image size is too small")
        self.records = tuple(records)
        self.image_size = image_size
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            grayscale = ImageOps.grayscale(image)
            resized = ImageOps.fit(
                grayscale,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    def _augment(self, image: torch.Tensor, index: int) -> torch.Tensor:
        if not self.augment:
            return image
        generator = torch.Generator().manual_seed(index + torch.initial_seed())
        if torch.rand((), generator=generator).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
        contrast = 0.9 + 0.2 * torch.rand((), generator=generator).item()
        brightness = -0.05 + 0.1 * torch.rand((), generator=generator).item()
        return (image * contrast + brightness).clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._augment(self._load_image(record.image_path), index)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "grade": torch.tensor(record.grade, dtype=torch.long),
            "participant_id": record.participant_id,
            "race": record.race,
            "sex": record.sex,
            "age": torch.tensor(record.age, dtype=torch.float32),
            "bmi": torch.tensor(record.bmi, dtype=torch.float32),
            "visit": record.visit,
            "cohort": record.cohort,
        }


class SyntheticKneeDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        size: int = 64,
        image_size: int = 32,
        seed: int = 7,
    ) -> None:
        if size < 5:
            raise ValueError("Synthetic dataset needs at least five examples")
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size) % 5
        noise = torch.randn(size, 1, image_size, image_size, generator=generator) * 0.05
        base = labels.float().view(-1, 1, 1, 1) / 4.0
        self.images = (base + noise).clamp(0.0, 1.0)
        self.labels = labels.long()
        self.groups = tuple(
            "White" if index % 2 == 0 else "African American" for index in range(size)
        )

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "grade": self.labels[index],
            "participant_id": f"synthetic-{index:05d}",
            "race": self.groups[index],
            "sex": "female" if index % 3 else "male",
            "age": torch.tensor(50.0 + index % 30),
            "bmi": torch.tensor(24.0 + index % 12),
            "visit": "V00",
            "cohort": "synthetic",
        }
