"""Model factory: an ImageNet-pretrained backbone, a fresh classification head, and the input
normalisation built into the network.

Because mean/std live inside the model, it takes plain RGB pixels in [0, 1]. Training, evaluation,
the ONNX export and the API therefore all feed it exactly the same thing, which removes the most
common train/serve skew bug: a normalisation mismatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torchvision import models

# Small CNNs on purpose: the dataset is 1.2k grayscale 200x200 textures, and the service runs on CPU.
ARCHS = ("resnet18", "mobilenet_v3_large", "efficientnet_b0")


class DefectClassifier(nn.Module):
    """normalise -> backbone -> logits."""

    def __init__(self, backbone: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model(
    arch: str,
    num_classes: int,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> DefectClassifier:
    """`pretrained=True` downloads the ImageNet weights on first use (needs internet once)."""
    if arch == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.fc.in_features, num_classes))
    elif arch == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture {arch!r}; choose one of {ARCHS}")
    return DefectClassifier(backbone, mean, std)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[DefectClassifier, dict]:
    """Rebuild the model from a checkpoint written by train.py. Returns (model in eval mode, checkpoint dict)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.models.train")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["class_names"]), ckpt["mean"], ckpt["std"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
