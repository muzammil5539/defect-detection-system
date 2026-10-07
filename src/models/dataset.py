"""Dataset and transforms.

Transforms stop at ToTensor (pixels in [0, 1]); normalisation happens inside the model.
Evaluation resizes with PIL bilinear, the same resize the API applies at inference time, so a
validation score means what a production score will mean.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T  # noqa: N812 (torchvision's own convention)
from torchvision.transforms import InterpolationMode

from src.utils.config import PROJECT_ROOT


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees. Lossless on square images; a steel surface has no 'up'."""

    def __call__(self, img: Image.Image) -> Image.Image:
        turns = random.randrange(4)
        return img.rotate(90 * turns) if turns else img


def eval_transform(image_size: int) -> T.Compose:
    return T.Compose([T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR), T.ToTensor()])


def train_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            # Mild scale/shift jitter: crops stay large because the defects fill much of each 200px image.
            T.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR
            ),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRotate90(),
            # Lighting varies along a production line; hue and saturation are meaningless on gray steel.
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ]
    )


class DefectDataset(Dataset):
    """Rows of a split CSV (image_path, label, label_idx) as (tensor, label_idx) pairs."""

    def __init__(
        self,
        csv_path: Path,
        transform,
        *,
        root: Path = PROJECT_ROOT,
        limit_per_class: int | None = None,
    ) -> None:
        frame = pd.read_csv(csv_path)
        if limit_per_class:  # for smoke tests: a few images of every class
            frame = frame.groupby("label_idx", sort=True).head(limit_per_class)
        self.paths: list[Path] = [root / rel for rel in frame["image_path"]]
        self.labels: list[int] = frame["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index]
