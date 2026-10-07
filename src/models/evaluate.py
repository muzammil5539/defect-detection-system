"""Evaluate a checkpoint on a held-out split and analyse its mistakes.

    uv run python -m src.models.evaluate                  test split (the one to report)
    uv run python -m src.models.evaluate --split val      validation split

Writes to results/ (prefixed with the split name):
  <split>_metrics.json            accuracy, macro and per-class precision/recall/F1, confusion matrix,
                                  false positives/negatives per class, calibration, confidence trade-off
  <split>_classification_report.txt   the same table sklearn prints
  <split>_confusion_matrix.png    counts per (true, predicted) pair
  <split>_errors.csv              every misclassified image with its confidence
  <split>_error_examples.png      the most confident mistakes, which are the most worrying ones

This model has no "normal" class (NEU-DET contains only defects), so a false positive here means
"predicted defect type X when the truth was Y", and a false negative means "missed a Y".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.models.build import load_checkpoint
from src.models.dataset import DefectDataset, eval_transform
from src.models.train import pick_device
from src.utils.config import CHECKPOINT_PATH, PROJECT_ROOT, RESULTS_DIR, TEST_CSV, VAL_CSV, setup_logging

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9)
SPLIT_CSVS = {"val": VAL_CSV, "test": TEST_CSV}


def predict_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """(true labels, softmax probabilities) for every image in the loader."""
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Average gap between stated confidence and actual accuracy, weighted by how many predictions fall in each bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = y_pred == y_true

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "false_positives": int(matrix[:, i].sum() - matrix[i, i]),  # predicted this class, truth was another
            "false_negatives": int(matrix[i].sum() - matrix[i, i]),  # truth was this class, predicted another
        }
        for i, name in enumerate(class_names)
    }
    confusions = sorted(
        (
            {"true": class_names[i], "predicted": class_names[j], "count": int(matrix[i, j])}
            for i in labels
            for j in labels
            if i != j and matrix[i, j] > 0
        ),
        key=lambda item: item["count"],
        reverse=True,
    )
    trade_off = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        kept = confidence >= threshold
        trade_off[str(threshold)] = {
            "share_of_images_answered": float(kept.mean()),
            "accuracy_on_those": float(correct[kept].mean()) if kept.any() else None,
        }
    return {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "labels": class_names,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": confusions[:10],
        "mean_confidence_when_right": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_when_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "confidence_trade_off": trade_off,
    }


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)), xlabel="predicted", ylabel="true", title=title
    )
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    peak = matrix.max() or 1
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > peak / 2 else "black"
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_table(paths: list[Path], y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """One row per misclassified image, most confident mistakes first."""
    y_pred = probs.argmax(axis=1)
    rows = []
    for index in np.flatnonzero(y_pred != y_true):
        runner_up = np.argsort(probs[index])[-2]
        rows.append(
            {
                "image_path": paths[index].relative_to(PROJECT_ROOT).as_posix()
                if paths[index].is_relative_to(PROJECT_ROOT)
                else str(paths[index]),
                "true": class_names[y_true[index]],
                "predicted": class_names[y_pred[index]],
                "confidence": float(probs[index].max()),
                "probability_of_true_class": float(probs[index, y_true[index]]),
                "second_choice": class_names[runner_up],
            }
        )
    columns = ["image_path", "true", "predicted", "confidence", "probability_of_true_class", "second_choice"]
    return pd.DataFrame(rows, columns=columns).sort_values("confidence", ascending=False).reset_index(drop=True)


def plot_error_examples(errors: pd.DataFrame, path: Path, limit: int = 12, columns: int = 4) -> None:
    shown = errors.head(limit)
    rows = max(1, -(-len(shown) // columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), shown.iterrows(), strict=False):
        source = Path(row["image_path"])
        with Image.open(source if source.is_absolute() else PROJECT_ROOT / source) as img:
            ax.imshow(img.convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true: {row['true']}\npred: {row['predicted']} ({row['confidence']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(
    checkpoint: Path,
    csv_path: Path,
    out_dir: Path,
    *,
    split: str,
    batch_size: int = 64,
    workers: int = 2,
) -> dict:
    device = pick_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    class_names: list[str] = ckpt["class_names"]
    dataset = DefectDataset(csv_path, eval_transform(ckpt["image_size"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    logger.info("Evaluating %s on the %s split (%d images)", checkpoint.name, split, len(dataset))

    y_true, probs = predict_probabilities(model, loader, device)
    metrics = compute_metrics(y_true, probs, class_names)
    metrics["split"] = split
    metrics["model"] = {
        "arch": ckpt["arch"],
        "epoch": ckpt["epoch"],
        "git_commit": ckpt.get("git_commit"),
        "pretrained": ckpt.get("pretrained"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = classification_report(
        y_true,
        probs.argmax(axis=1),
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (out_dir / f"{split}_classification_report.txt").write_text(report + "\n", encoding="utf-8")
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names,
        out_dir / f"{split}_confusion_matrix.png",
        f"{split} split ({len(y_true)} images)",
    )

    errors = error_table(dataset.paths, y_true, probs, class_names)
    errors.to_csv(out_dir / f"{split}_errors.csv", index=False)
    if len(errors):
        plot_error_examples(errors, out_dir / f"{split}_error_examples.png")

    logger.info("\n%s", report)
    logger.info(
        "accuracy %.4f | macro-F1 %.4f | %d mistakes | ECE %.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
        len(errors),
        metrics["expected_calibration_error"],
    )
    for item in metrics["top_confusions"][:5]:
        logger.info("  confused: true %-16s predicted %-16s x%d", item["true"], item["predicted"], item["count"])
    logger.info("Artifacts written to %s", out_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and analyse its errors.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", choices=sorted(SPLIT_CSVS), default="test")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    setup_logging()
    try:
        evaluate(
            args.checkpoint,
            SPLIT_CSVS[args.split],
            args.results_dir,
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
