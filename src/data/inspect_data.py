"""Inspect the raw NEU-DET dataset: structure, counts, corruption, formats, imbalance.

Run: uv run python -m src.data.inspect_data
"""
import logging
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from src.utils.config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    RAW_DIR,
    RESULTS_DIR,
    setup_logging,
)

logger = logging.getLogger(__name__)

SPLITS = {"train": RAW_DIR / "train" / "images", "validation": RAW_DIR / "validation" / "images"}


def list_images(folder: Path) -> list[Path]:
    """Sorted image files directly under `folder` (sorted so runs are reproducible)."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def is_valid_image(path: Path) -> bool:
    """True if PIL can parse the file. verify() checks integrity without decoding pixels."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as exc:  # PIL raises many types (OSError, SyntaxError, ...); we skip all
        logger.warning("Corrupted/unreadable image skipped: %s (%s)", path, exc)
        return False


def imbalance_ratio(counts: dict[str, int]) -> float:
    """Largest class / smallest class. 1.0 = perfectly balanced."""
    positive = [c for c in counts.values() if c > 0]
    return max(positive) / min(positive) if positive else float("nan")


def tree_lines(root: Path) -> list[str]:
    """Recursive folder tree with the number of files directly inside each folder.

    Files themselves are not listed (1800 lines of filenames help nobody).
    """
    lines = [f"{root.name}/"]
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        depth = len(d.relative_to(root).parts)
        n_files = sum(1 for f in d.iterdir() if f.is_file())
        lines.append(f"{'    ' * depth}{d.name}/  ({n_files} files)")
    return lines


def inspect_split(images_dir: Path) -> tuple[dict[str, int], Counter, Counter, Counter, list[Path]]:
    """Return per-class counts plus Counters of (size, mode, format) and the bad-file list."""
    counts: dict[str, int] = {}
    sizes: Counter = Counter()
    modes: Counter = Counter()
    formats: Counter = Counter()
    corrupted: list[Path] = []

    for cls in CLASS_NAMES:
        cls_dir = images_dir / cls
        files = list_images(cls_dir) if cls_dir.is_dir() else []
        counts[cls] = len(files)
        for f in files:
            if not is_valid_image(f):
                corrupted.append(f)
                continue
            # verify() leaves the image unusable, so reopen to read its properties
            with Image.open(f) as img:
                sizes[img.size] += 1
                modes[img.mode] += 1
                formats[img.format] += 1
    return counts, sizes, modes, formats, corrupted


def build_summary() -> list[str]:
    """Run every check and return the report as lines (logged and saved by main)."""
    out = ["NEU-DET DATASET SUMMARY", "=" * 60, "", "FOLDER TREE", *tree_lines(RAW_DIR), ""]

    for split, images_dir in SPLITS.items():
        counts, sizes, modes, formats, corrupted = inspect_split(images_dir)
        found = {p.name for p in images_dir.iterdir() if p.is_dir()} if images_dir.is_dir() else set()
        missing, extra = sorted(set(CLASS_NAMES) - found), sorted(found - set(CLASS_NAMES))

        out += [f"[{split.upper()}]  total images: {sum(counts.values())}"]
        out += [f"  classes: {'all 6 present' if not missing else 'MISSING ' + str(missing)}"]
        if extra:
            out += [f"  unexpected folders: {extra}"]
        out += [f"  {cls:<18}{n}" for cls, n in counts.items()]
        out += [
            f"  imbalance ratio (max/min): {imbalance_ratio(counts):.2f}",
            f"  corrupted images: {len(corrupted)}" + "".join(f"\n    {p}" for p in corrupted),
            f"  unique dimensions: {dict(sizes)}",
            f"  modes: {dict(modes)}",
            f"  formats: {dict(formats)}",
            "",
        ]
    return out


def main() -> int:
    setup_logging()
    if not RAW_DIR.is_dir():
        logger.error("Dataset not found at %s", RAW_DIR)
        return 1

    summary = build_summary()
    for line in summary:
        logger.info(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "data_summary.txt"
    out_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    logger.info("Summary written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
