"""Central configuration: paths, class names and global constants.

Every other module imports from here so a path or constant is defined exactly once.
"""
import logging
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Resolved from this file (not the CWD) so scripts work no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
TRAIN_IMAGES_DIR = RAW_DIR / "train" / "images"
VAL_IMAGES_DIR = RAW_DIR / "validation" / "images"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

# --- Classes ---------------------------------------------------------------
# Alphabetical so label_idx is deterministic and identical across every script/run.
CLASS_NAMES: list[str] = sorted(
    ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
)

# NOTE: NEU-DET has NO "Normal" images, so we train a 6-class defect-type classifier.
# With BINARY_MODE = True, the inference layer collapses the output to a binary decision:
# "defective" (any of the 6 classes). The model itself never learns a Normal class,
# so a real Normal-vs-defective decision needs real normal images or a confidence/OOD threshold.
BINARY_MODE = True

# --- Misc constants --------------------------------------------------------
RANDOM_SEED = 42
IMAGE_SIZE = 224  # standard input size for ImageNet-pretrained backbones
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with one consistent format for every script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # replace any handlers set by libraries so the format is always ours
    )
