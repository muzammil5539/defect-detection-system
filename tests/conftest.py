"""Shared fixtures.

Two stand-ins keep the whole suite fast and offline:
  * a tiny ONNX model built with the onnx helpers (no torch needed): class i fires when colour
    channel i is brightest, so a red image is "red", a green one "green", and so on;
  * a local S3 server (moto) in place of IDrive e2.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from src.serving.contract import ModelInfo
from src.serving.inference import OnnxClassifier
from src.serving.settings import Settings, StorageSettings
from src.serving.storage import make_s3_client

CLASSES = ("red", "green", "blue")
SIZE = 32


def build_tiny_model(
    path: Path,
    *,
    classes: tuple[str, ...] = CLASSES,
    size: int = SIZE,
    version: str = "test-1",
    normal_class: str | None = None,
    with_metadata: bool = True,
    logits_dim: int | None = None,
) -> Path:
    """Write a valid ONNX classifier: GlobalAveragePool -> Flatten -> Gemm, dynamic batch."""
    n_out = logits_dim or len(classes)
    weight = np.eye(3, n_out, dtype=np.float32) * 10.0
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "w", "b"], ["logits"]),
        ],
        "tiny_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_out])],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(np.zeros(n_out, np.float32), "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    if with_metadata:
        info = ModelInfo(
            model_version=version,
            arch="tiny",
            class_names=classes,
            input_size=size,
            normal_class=normal_class,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            created_at="2026-01-01T00:00:00Z",
            git_commit="abc1234",
        )
        helper.set_model_props(model, info.to_metadata())
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def solid_image(color: tuple[int, int, int], size: tuple[int, int] = (64, 48), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("models") / "best_model.onnx")


@pytest.fixture(scope="session")
def classifier(tiny_model: Path) -> OnnxClassifier:
    return OnnxClassifier(tiny_model, threads=1, low_confidence_threshold=0.6)


@pytest.fixture
def local_settings(tiny_model: Path, tmp_path: Path):
    """Settings that read the model from a local file; overrides are env-style strings."""

    def make(**overrides: str) -> Settings:
        env = {"MODEL_PATH": str(tiny_model), "MODEL_CACHE_DIR": str(tmp_path / "cache"), "ORT_THREADS": "1"}
        env.update(overrides)
        return Settings.from_env(env)

    return make


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterator[str]:
    """A local S3-compatible server, standing in for IDrive e2."""
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0, verbose=False)  # port 0: the OS picks a free port
    server.start()
    _, port = server.get_host_and_port()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def storage_cfg(s3_endpoint: str) -> StorageSettings:
    """A fresh, empty bucket per test, so tests cannot see each other's objects."""
    cfg = StorageSettings(
        endpoint_url=s3_endpoint,
        bucket=f"test-{uuid.uuid4().hex[:12]}",
        access_key_id="test",
        secret_access_key="test",
    )
    make_s3_client(cfg).create_bucket(Bucket=cfg.bucket)
    return cfg


def put_model(cfg: StorageSettings, model_path: Path, *, sha256: str | None = "auto", key: str | None = None) -> str:
    """Upload a model the way publish_model does. sha256=None stores no checksum; 'auto' computes it."""
    data = model_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest() if sha256 == "auto" else sha256
    metadata = {"sha256": digest} if digest else {}
    make_s3_client(cfg).put_object(Bucket=cfg.bucket, Key=key or cfg.model_key, Body=data, Metadata=metadata)
    return hashlib.sha256(data).hexdigest()
