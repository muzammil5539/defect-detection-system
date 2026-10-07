"""Build split CSVs, a sample grid and normalization stats from NEU-DET.

Uses the dataset's OFFICIAL train/validation split as-is (no re-splitting).
Run: uv run python -m src.data.make_dataset
"""
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed on servers/Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.inspect_data import imbalance_ratio, is_valid_image, list_images
from src.utils.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLITS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    setup_logging,
)

logger = logging.getLogger(__name__)


def build_split_df(images_dir: Path) -> pd.DataFrame:
    """One row per valid image: image_path, label (str), label_idx (int).

    image_path is relative to PROJECT_ROOT so the CSVs work on any machine / in Docker.
    Corrupted files are logged by is_valid_image and left out.
    """
    rows = []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = images_dir / cls
        if not cls_dir.is_dir():
            logger.error("Class folder missing: %s", cls_dir)
            continue
        for f in list_images(cls_dir):
            if is_valid_image(f):
                rows.append((f.relative_to(PROJECT_ROOT).as_posix(), cls, idx))
    return pd.DataFrame(rows, columns=["image_path", "label", "label_idx"])


def log_distribution(name: str, df: pd.DataFrame) -> None:
    """Log per-class counts + imbalance ratio; this drives the later loss/sampling choice."""
    counts = df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()
    logger.info("%s: %d images, imbalance ratio %.2f", name, len(df), imbalance_ratio(counts))
    for cls, n in counts.items():
        logger.info("  %-18s%d", cls, n)


def save_sample_grid(train_df: pd.DataFrame, out_path: Path, per_class: int = 4) -> None:
    """6x4 grid (rows = classes) so we can eyeball the defects and spot labeling problems."""
    rng = random.Random(RANDOM_SEED)  # fixed seed -> same grid every run
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(2.5 * per_class + 1, 2.5 * len(CLASS_NAMES)))
    for r, cls in enumerate(CLASS_NAMES):
        paths = train_df.loc[train_df["label"] == cls, "image_path"].tolist()
        picks = rng.sample(paths, min(per_class, len(paths)))
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(PROJECT_ROOT / picks[c]) as img:
                    ax.imshow(img.convert("L"), cmap="gray", vmin=0, vmax=255)
        # axis("off") hides ylabel too, so draw the row title as text instead
        axes[r, 0].text(-0.08, 0.5, cls, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample grid saved to %s", out_path)


def compute_norm_stats(train_df: pd.DataFrame) -> dict:
    """Per-channel mean/std over the TRAIN set only (val/test stats would leak information).

    Images are converted to RGB and resized to IMAGE_SIZE so the stats describe what the
    model will actually see; pixels are scaled to [0, 1] to match torchvision's ToTensor.
    Uses running sums (float64) so memory stays constant: std = sqrt(E[x^2] - E[x]^2).
    """
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_pixels = 0
    n_images = 0
    for rel in tqdm(train_df["image_path"], desc="norm stats"):
        try:
            with Image.open(PROJECT_ROOT / rel) as img:
                arr = np.asarray(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float64) / 255.0
        except Exception as exc:  # already validated, but never crash a long loop on one file
            logger.warning("Skipping %s in stats: %s", rel, exc)
            continue
        total += arr.sum(axis=(0, 1))
        total_sq += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]
        n_images += 1

    mean = total / n_pixels
    std = np.sqrt(total_sq / n_pixels - mean**2)
    return {
        "mean": mean.round(6).tolist(),
        "std": std.round(6).tolist(),
        "channels": "RGB",
        "pixel_range": "[0, 1]",
        "image_size": IMAGE_SIZE,
        "num_images": n_images,
    }


def main() -> int:
    setup_logging()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_split_df(TRAIN_IMAGES_DIR)
    val_df = build_split_df(VAL_IMAGES_DIR)
    if train_df.empty or val_df.empty:
        logger.error("Empty split (train=%d, val=%d); check %s", len(train_df), len(val_df), TRAIN_IMAGES_DIR.parent.parent)
        return 1

    for name, df in (("train", train_df), ("val", val_df)):
        df.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        log_distribution(name, df)
    logger.info("CSVs written to %s", SPLITS_DIR)

    save_sample_grid(train_df, RESULTS_DIR / "sample_grid.png")

    stats = compute_norm_stats(train_df)
    (RESULTS_DIR / "normalization_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("Normalization stats: mean=%s std=%s", stats["mean"], stats["std"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
