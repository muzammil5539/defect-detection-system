"""Build split CSVs, a sample grid and normalization stats from NEU-DET.

Three splits, all disjoint:
  train  85% of the official train folder (stratified by class, seeded)
  val    the other 15%: used for model selection and early stopping
  test   the official validation folder, untouched: used once, for the final report

The official validation folder is kept whole as the test set so results stay comparable
with published NEU-DET numbers and no test image can influence a training decision.
Run: uv run python -m src.data.make_dataset
"""
import hashlib
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.inspect_data import imbalance_ratio, is_valid_image, list_images
from src.utils.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    NORM_STATS_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLITS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_FRACTION,
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


def split_train_val(df: pd.DataFrame, val_fraction: float = VAL_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split: every class keeps the same share in train and val.

    Rows are re-sorted by path so the CSVs are byte-identical from run to run.
    """
    train_df, val_df = train_test_split(df, test_size=val_fraction, stratify=df["label_idx"], random_state=RANDOM_SEED)
    return (
        train_df.sort_values("image_path").reset_index(drop=True),
        val_df.sort_values("image_path").reset_index(drop=True),
    )


def _content_hash(rel_path: str) -> str:
    """md5 of the file bytes. Used for de-duplication only, not security."""
    return hashlib.md5((PROJECT_ROOT / rel_path).read_bytes()).hexdigest()


def drop_duplicate_images(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first of every group of byte-identical files.

    Done before splitting: otherwise two copies of one image can land in train AND val,
    and the val score then partly measures memorisation.
    """
    first_path: dict[str, str] = {}
    keep: list[bool] = []
    for rel in df["image_path"]:
        digest = _content_hash(rel)
        if digest in first_path:
            logger.warning("Dropping duplicate image %s (identical to %s)", rel, first_path[digest])
            keep.append(False)
        else:
            first_path[digest] = rel
            keep.append(True)
    return df[keep].reset_index(drop=True)


def find_cross_split_duplicates(splits: dict[str, pd.DataFrame]) -> list[str]:
    """One message per file whose bytes also appear in an earlier split (expected: none)."""
    first_seen: dict[str, tuple[str, str]] = {}  # content hash -> (split, path)
    found: list[str] = []
    for name, df in splits.items():
        for rel in df["image_path"]:
            digest = _content_hash(rel)
            if digest not in first_seen:
                first_seen[digest] = (name, rel)
            elif first_seen[digest][0] != name:
                found.append(f"{rel} [{name}] == {first_seen[digest][1]} [{first_seen[digest][0]}]")
    return found


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

    official_train_df = build_split_df(TRAIN_IMAGES_DIR)
    test_df = build_split_df(VAL_IMAGES_DIR)  # the official "validation" folder is our untouched test set
    if official_train_df.empty or test_df.empty:
        logger.error(
            "Empty split (train=%d, test=%d); check %s", len(official_train_df), len(test_df), TRAIN_IMAGES_DIR.parent.parent
        )
        return 1

    train_df, val_df = split_train_val(drop_duplicate_images(official_train_df))
    splits = {"train": train_df, "val": val_df, "test": test_df}

    duplicates = find_cross_split_duplicates(splits)
    for msg in duplicates:
        logger.warning("Cross-split duplicate: %s", msg)
    logger.info("Cross-split duplicate images: %d", len(duplicates))

    for name, df in splits.items():
        df.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        log_distribution(name, df)
    logger.info("CSVs written to %s", SPLITS_DIR)

    save_sample_grid(train_df, RESULTS_DIR / "sample_grid.png")

    stats = compute_norm_stats(train_df)
    NORM_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("Normalization stats: mean=%s std=%s", stats["mean"], stats["std"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
