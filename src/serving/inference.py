"""ONNX Runtime inference: image bytes in, class and confidence out (steps 3 and 4 of startup).

Preprocessing must match training exactly. The model takes RGB pixels in [0, 1] because the
mean/std normalisation is baked into the graph at export time, so this module only has to
decode, convert to RGB, resize bilinearly (the same PIL resize torchvision uses) and divide by 255.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.serving.contract import ModelContractError, ModelInfo

logger = logging.getLogger(__name__)

# PIL reports JPEG variants as "JPEG" and "MPO" (multi-picture JPEG, written by some phones/cameras).
ALLOWED_FORMATS = frozenset({"JPEG", "MPO", "PNG", "BMP", "WEBP"})


class InvalidImageError(ValueError):
    """The uploaded bytes are not an image this service can classify. Safe to show to the caller."""


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    is_defective: bool
    needs_review: bool
    inference_ms: float


def preprocess_image(data: bytes, size: int, max_pixels: int) -> np.ndarray:
    """Decode image bytes into a float32 array of shape (1, 3, size, size) with values in [0, 1]."""
    try:
        img = Image.open(io.BytesIO(data))  # lazy: reads the header only
        if img.format not in ALLOWED_FORMATS:
            raise InvalidImageError(f"unsupported image format {img.format!r}; send JPEG, PNG, BMP or WebP")
        width, height = img.size
        if width * height > max_pixels:
            raise InvalidImageError(f"image is {width}x{height}, above the limit of {max_pixels:,} pixels")
        if img.mode in {"I", "F"} or img.mode.startswith("I;16"):
            raise InvalidImageError("16/32-bit images are not supported; send an 8-bit image")
        img.load()  # decode now, so truncated or corrupt files fail here with a clear message
        rgb = img.convert("RGB")
    except InvalidImageError:
        raise
    except Exception as exc:  # PIL raises many unrelated types for bad input (OSError, SyntaxError, ...)
        raise InvalidImageError(f"could not decode the image ({type(exc).__name__})") from exc

    resized = rgb.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0  # H x W x C
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))[None]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis, in float64."""
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxClassifier:
    """One ONNX Runtime session shared by all requests (InferenceSession.run is thread-safe)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threads: int = 1,
        low_confidence_threshold: float = 0.6,
        max_image_pixels: int = 12_000_000,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Idle worker threads would otherwise spin and burn CPU, which hurts on small shared hosts.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        try:
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # ORT raises InvalidProtobuf, Fail, NoSuchFile, ...
            raise ModelContractError(f"onnxruntime could not load {model_path}: {exc}") from exc

        self.info = ModelInfo.from_metadata(self._session.get_modelmeta().custom_metadata_map)
        self._input_name, self._output_name = self._check_graph()
        self._low_confidence_threshold = low_confidence_threshold
        self._max_image_pixels = max_image_pixels

    def _check_graph(self) -> tuple[str, str]:
        """Fail at startup, not on the first request, if the graph does not match its metadata."""
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ModelContractError(f"expected 1 input and at least 1 output, found {len(inputs)} and {len(outputs)}")
        shape = inputs[0].shape  # e.g. ['batch', 3, 224, 224]; dynamic dims are strings or None
        size = self.info.input_size
        static_ok = all(dim in (size, None) or isinstance(dim, str) for dim in shape[2:])
        if len(shape) != 4 or shape[1] != 3 or not static_ok or inputs[0].type != "tensor(float)":
            raise ModelContractError(
                f"input must be float [batch, 3, {size}, {size}], the graph declares {inputs[0].type} {shape}"
            )
        out_shape = outputs[0].shape
        if len(out_shape) != 2 or (isinstance(out_shape[1], int) and out_shape[1] != self.info.num_classes):
            raise ModelContractError(
                f"output must be [batch, {self.info.num_classes}] logits, the graph declares {out_shape}"
            )
        return inputs[0].name, outputs[0].name

    def warmup(self) -> None:
        """Run once so the first real request is not slower, and confirm the output makes sense."""
        size = self.info.input_size
        logits = self._run(np.zeros((1, 3, size, size), dtype=np.float32))
        if logits.shape != (1, self.info.num_classes) or not np.isfinite(logits).all():
            raise ModelContractError(f"warm-up run returned {logits.shape} (finite: {np.isfinite(logits).all()})")

    def _run(self, batch: np.ndarray) -> np.ndarray:
        return self._session.run([self._output_name], {self._input_name: batch})[0]

    def predict(self, image_bytes: bytes) -> Prediction:
        batch = preprocess_image(image_bytes, self.info.input_size, self._max_image_pixels)
        started = time.perf_counter()
        logits = self._run(batch)
        inference_ms = (time.perf_counter() - started) * 1000

        probs = softmax(logits)[0]
        top = int(np.argmax(probs))
        name = self.info.class_names[top]
        confidence = float(probs[top])
        return Prediction(
            predicted_class=name,
            confidence=confidence,
            probabilities={cls: float(p) for cls, p in zip(self.info.class_names, probs, strict=True)},
            is_defective=name != self.info.normal_class,
            needs_review=confidence < self._low_confidence_threshold,
            inference_ms=inference_ms,
        )
