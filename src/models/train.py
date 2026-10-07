"""Fine-tune an ImageNet-pretrained CNN on the NEU-DET defect classes.

    uv run python -m src.models.train                       resnet18, up to 15 epochs
    uv run python -m src.models.train --arch efficientnet_b0 --epochs 20
    uv run python -m src.models.train --no-pretrained --epochs 1 --limit-per-class 20    quick smoke test

Why these choices (the README has the longer version):
  * transfer learning: ~1.2k training images are too few to learn good filters from scratch;
  * validation macro-F1 picks the best epoch, because accuracy can hide one weak class;
  * class-weighted loss: a no-op on today's balanced data, a safeguard when new data is not;
  * flips and 90-degree turns: a steel surface has no up or down, and they lose no pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.build import ARCHS, build_model
from src.models.dataset import DefectDataset, eval_transform, train_transform
from src.utils.config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    NORM_STATS_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    TRAIN_CSV,
    VAL_CSV,
    setup_logging,
)
from src.utils.provenance import git_short_sha, utc_now

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    arch: str = "resnet18"
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05  # keeps the softmax confidence from saturating at 1.0
    image_size: int = IMAGE_SIZE
    patience: int = 5  # stop after this many epochs without a better validation macro-F1
    pretrained: bool = True
    workers: int = 2
    seed: int = RANDOM_SEED
    limit_per_class: int | None = None
    train_csv: Path = TRAIN_CSV
    val_csv: Path = VAL_CSV
    norm_stats: Path = NORM_STATS_PATH
    output: Path = CHECKPOINT_PATH


def load_norm_stats(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Prepare the data first: python -m src.data.make_dataset")
    stats = json.loads(path.read_text(encoding="utf-8"))
    return stats["mean"], stats["std"]


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """n / (k * count): rare classes weigh more. Exactly 1.0 for every class when they are balanced."""
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(np.float64), 1)  # absent class: avoid /0
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, desc: str = ""):
    """One pass over `loader`. Trains when an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, seen = 0.0, 0
    truths: list[int] = []
    predictions: list[int] = []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False, disable=None):  # disable=None: off when not a TTY
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            truths += labels.tolist()
            predictions += logits.argmax(1).tolist()
    return total_loss / seen, np.array(truths), np.array(predictions)


def save_checkpoint(model, cfg: TrainConfig, mean, std, *, epoch: int, val_f1: float, val_loss: float) -> None:
    checkpoint = {
        "arch": cfg.arch,
        "class_names": list(CLASS_NAMES),
        "image_size": cfg.image_size,
        "mean": list(mean),
        "std": list(std),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "epoch": epoch,
        "val_macro_f1": val_f1,
        "val_loss": val_loss,
        "pretrained": cfg.pretrained,
        "trained_at": utc_now().isoformat(),
        "git_commit": git_short_sha(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.output.with_suffix(cfg.output.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, cfg.output)  # atomic: a crash never leaves a half-written checkpoint


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(10, 3.6))
    loss_ax.plot(history["epoch"], history["train_loss"], label="train")
    loss_ax.plot(history["epoch"], history["val_loss"], label="validation")
    loss_ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    loss_ax.legend()
    f1_ax.plot(history["epoch"], history["train_macro_f1"], label="train")
    f1_ax.plot(history["epoch"], history["val_macro_f1"], label="validation")
    f1_ax.set(xlabel="epoch", ylabel="macro-F1", title="Macro-F1", ylim=(0, 1.02))
    f1_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fit(cfg: TrainConfig, results_dir: Path = RESULTS_DIR) -> dict:
    """Train, keep the best epoch by validation macro-F1, return a summary."""
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown architecture {cfg.arch!r}; choose one of {ARCHS}")
    set_seed(cfg.seed)
    device = pick_device()
    mean, std = load_norm_stats(cfg.norm_stats)

    train_ds = DefectDataset(cfg.train_csv, train_transform(cfg.image_size), limit_per_class=cfg.limit_per_class)
    val_ds = DefectDataset(cfg.val_csv, eval_transform(cfg.image_size), limit_per_class=cfg.limit_per_class)
    loader_args = {
        "num_workers": cfg.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        drop_last=len(train_ds) % cfg.batch_size == 1,  # BatchNorm cannot train on a batch of one
        **loader_args,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_args)

    model = build_model(cfg.arch, len(CLASS_NAMES), mean, std, pretrained=cfg.pretrained).to(device)
    weights = class_weights(train_ds.labels, len(CLASS_NAMES))
    logger.info("Device %s | %s | train %d, val %d images", device, cfg.arch, len(train_ds), len(val_ds))
    logger.info("Class weights: %s", dict(zip(CLASS_NAMES, weights.round(decimals=3).tolist(), strict=True)))

    train_criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=cfg.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # plain loss, so validation numbers are comparable between runs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    rows: list[dict] = []
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0}
    epochs_without_gain = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(
            model, train_loader, train_criterion, device, optimizer, f"train {epoch}"
        )
        val_loss, val_true, val_pred = run_epoch(model, val_loader, val_criterion, device, None, f"val {epoch}")
        scheduler.step()

        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": float(f1_score(train_true, train_pred, average="macro", zero_division=0)),
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": float((val_true == val_pred).mean()),
            }
        )
        logger.info(
            "epoch %2d/%d | train loss %.4f | val loss %.4f | val macro-F1 %.4f | val acc %.4f",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_f1,
            rows[-1]["val_accuracy"],
        )

        better = val_f1 > best["f1"] + 1e-9 or (abs(val_f1 - best["f1"]) <= 1e-9 and val_loss < best["loss"])
        if better:
            best = {"f1": val_f1, "loss": val_loss, "epoch": epoch}
            epochs_without_gain = 0
            save_checkpoint(model, cfg, mean, std, epoch=epoch, val_f1=val_f1, val_loss=val_loss)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                logger.info("No validation gain for %d epochs: stopping early", cfg.patience)
                break

    results_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(rows)
    history.to_csv(results_dir / "training_history.csv", index=False)
    plot_history(history, results_dir / "learning_curves.png")
    logger.info("Best epoch %d: val macro-F1 %.4f. Checkpoint: %s", best["epoch"], best["f1"], cfg.output)
    return {
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["f1"],
        "checkpoint": str(cfg.output),
        "epochs_run": len(rows),
    }


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, Path]:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Fine-tune a CNN on the NEU-DET defect classes.")
    parser.add_argument("--arch", choices=ARCHS, default=defaults.arch)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false", help="random init (offline smoke tests)"
    )
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--limit-per-class", type=int, default=None, help="use only N images per class (smoke tests)")
    parser.add_argument("--output", type=Path, default=defaults.output, help="checkpoint path (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    values = {key: value for key, value in vars(args).items() if key != "results_dir"}
    return TrainConfig(**values), args.results_dir


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    cfg, results_dir = parse_args(argv)
    try:
        summary = fit(cfg, results_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Next: python -m src.models.evaluate --checkpoint %s", summary["checkpoint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
