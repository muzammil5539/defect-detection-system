"""Export the trained checkpoint to ONNX: the file that gets published to the bucket.

    uv run python -m src.models.export_onnx                 models/best_model.pt -> models/best_model.onnx

What ends up in the file:
  * the network, with the input normalisation baked in, and a dynamic batch axis;
  * metadata (class names, input size, version, git commit) written with the keys defined in
    src/serving/contract.py, so the API needs nothing else to serve it.

Gate: before the file is moved into place, it is run through ONNX Runtime on real images and must
reproduce PyTorch's logits, using the same preprocessing function the API uses. If anything
differs, nothing is written, so a broken export can never reach the publish step.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from src.models.build import load_checkpoint
from src.models.dataset import eval_transform
from src.serving.contract import ModelInfo
from src.serving.inference import preprocess_image
from src.utils.config import CHECKPOINT_PATH, ONNX_PATH, PROJECT_ROOT, RESULTS_DIR, VAL_CSV, setup_logging
from src.utils.provenance import git_short_sha, utc_now

logger = logging.getLogger(__name__)

PARITY_SAMPLES = 8
NO_PIXEL_LIMIT = 10**9  # the parity images are ours; the API's upload limit does not apply here


class ExportError(RuntimeError):
    """The export did not reproduce the PyTorch model; nothing was written."""


def default_samples(size: int, count: int = PARITY_SAMPLES) -> list[bytes]:
    """Real validation images if the split exists, otherwise random-noise PNGs."""
    samples: list[bytes] = []
    if VAL_CSV.is_file():
        import pandas as pd

        for rel in pd.read_csv(VAL_CSV)["image_path"].head(count):
            samples.append((PROJECT_ROOT / rel).read_bytes())
    rng = np.random.default_rng(0)
    while len(samples) < count:
        noise = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(noise).save(buffer, format="PNG")
        samples.append(buffer.getvalue())
    return samples


def export_graph(model: torch.nn.Module, size: int, path: Path) -> str:
    """Write the ONNX graph. Returns which exporter produced it."""
    dummy = torch.zeros(1, 3, size, size)
    common = {"input_names": ["input"], "output_names": ["logits"]}
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:  # torch < 2.5 has only one exporter
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            **common,
        )
        return "torchscript"
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            opset_version=17,
            dynamo=False,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            **common,
        )
        return "torchscript"
    except Exception as exc:  # the legacy exporter is deprecated and may disappear; the new one needs onnxscript
        logger.warning("Legacy exporter failed (%s: %s). Trying the dynamo exporter.", type(exc).__name__, exc)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        opset_version=18,
        dynamo=True,
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        **common,
    )
    return "dynamo"


def finalise(path: Path, info: ModelInfo) -> None:
    """Embed the metadata, validate, and rewrite as ONE self-contained file (no external weight sidecar)."""
    proto = onnx.load(str(path))  # also reads a weights sidecar if the exporter wrote one
    onnx.helper.set_model_props(proto, info.to_metadata())
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path), save_as_external_data=False)
    Path(str(path) + ".data").unlink(missing_ok=True)


def check_parity(model: torch.nn.Module, onnx_path: Path, images: list[bytes], size: int, tolerance: float) -> dict:
    """ONNX Runtime fed by the API's preprocessing must match PyTorch fed by the training preprocessing."""
    served = np.concatenate([preprocess_image(data, size, NO_PIXEL_LIMIT) for data in images])
    transform = eval_transform(size)
    trained = torch.stack([transform(Image.open(io.BytesIO(data)).convert("RGB")) for data in images])
    preprocess_diff = float(np.abs(trained.numpy() - served).max())

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model(trained).numpy()
    batched = session.run(["logits"], {"input": served})[0]
    one_by_one = np.concatenate([session.run(["logits"], {"input": served[i : i + 1]})[0] for i in range(len(images))])
    logit_diff = float(max(np.abs(reference - batched).max(), np.abs(reference - one_by_one).max()))

    problems = []
    if preprocess_diff > 1e-6:
        problems.append(f"API preprocessing differs from training preprocessing by {preprocess_diff:.2e}")
    if logit_diff > tolerance:
        problems.append(f"ONNX logits differ from PyTorch by {logit_diff:.2e} (tolerance {tolerance:.0e})")
    if not (reference.argmax(1) == batched.argmax(1)).all():
        problems.append("ONNX and PyTorch disagree on the predicted class")
    if problems:
        raise ExportError("; ".join(problems))
    return {
        "samples": len(images),
        "max_abs_logit_diff": logit_diff,
        "max_abs_preprocess_diff": preprocess_diff,
        "tolerance": tolerance,
    }


def benchmark_latency(onnx_path: Path, size: int, runs: int = 30) -> dict:
    """Single-image latency on one CPU thread: a conservative figure for small hosts."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    batch = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
    for _ in range(5):
        session.run(None, {"input": batch})
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {"input": batch})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "p50": round(float(np.percentile(timings, 50)), 2),
        "p95": round(float(np.percentile(timings, 95)), 2),
        "runs": runs,
    }


def export(
    checkpoint: Path = CHECKPOINT_PATH,
    onnx_path: Path = ONNX_PATH,
    *,
    results_dir: Path = RESULTS_DIR,
    normal_class: str | None = None,
    sample_images: list[bytes] | None = None,
    tolerance: float = 1e-3,
) -> dict:
    model, ckpt = load_checkpoint(checkpoint)
    size: int = ckpt["image_size"]
    now = utc_now()
    commit = git_short_sha()
    info = ModelInfo(
        model_version=f"{now:%Y%m%dT%H%M%SZ}-{commit}",
        arch=ckpt["arch"],
        class_names=tuple(ckpt["class_names"]),
        input_size=size,
        normal_class=normal_class,
        mean=tuple(ckpt["mean"]),
        std=tuple(ckpt["std"]),
        created_at=now.isoformat(),
        git_commit=commit,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    staging = onnx_path.with_name(f".{onnx_path.name}.{os.getpid()}.tmp")
    try:
        exporter = export_graph(model, size, staging)
        finalise(staging, info)
        parity = check_parity(model, staging, sample_images or default_samples(size), size, tolerance)
        latency = benchmark_latency(staging, size)
        os.replace(staging, onnx_path)  # only a verified file ever appears at the final path
    finally:
        staging.unlink(missing_ok=True)
        Path(str(staging) + ".data").unlink(missing_ok=True)

    report = {
        "model_version": info.model_version,
        "arch": info.arch,
        "onnx_path": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "exporter": exporter,
        "parity": parity,
        "latency_ms_batch1_one_thread": latency,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Exported %s (%.1f MB, version %s) | parity diff %.1e | latency p50 %.1f ms, p95 %.1f ms (1 CPU thread)",
        onnx_path,
        report["size_mb"],
        info.model_version,
        parity["max_abs_logit_diff"],
        latency["p50"],
        latency["p95"],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the trained checkpoint to a self-describing ONNX file.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=ONNX_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--normal-class", default=None, help="name of the 'normal' class, if the model has one")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="max allowed logit difference vs PyTorch")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        export(
            args.checkpoint,
            args.output,
            results_dir=args.results_dir,
            normal_class=args.normal_class,
            tolerance=args.tolerance,
        )
    except (FileNotFoundError, ExportError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Next: python -m src.deploy.publish_model --model %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
