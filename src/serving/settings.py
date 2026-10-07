"""Runtime settings for the inference service, read from environment variables.

Everything the container needs arrives through the environment, so the same image runs
locally, in CI and on any Docker host. Secrets are never logged: use Settings.describe().
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import PROJECT_ROOT

DEFAULT_MODEL_KEY = "models/best_model.onnx"
ADDRESSING_STYLES = ("path", "virtual", "auto")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    """Required settings are missing or malformed. The message lists every problem at once."""


@dataclass(frozen=True)
class StorageSettings:
    """Where the model object lives in the S3-compatible bucket (IDrive e2)."""

    endpoint_url: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    model_key: str = DEFAULT_MODEL_KEY
    region: str = "us-east-1"
    addressing_style: str = "path"  # path-style works on every S3-compatible endpoint, with no wildcard DNS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> StorageSettings:
        reader = _EnvReader(env)
        storage = _read_storage(reader)
        reader.raise_if_errors()
        assert storage is not None  # no errors, so every required value was present
        return storage


@dataclass(frozen=True)
class Settings:
    storage: StorageSettings | None  # None when MODEL_PATH points at a local file
    model_path: Path | None
    cache_dir: Path
    max_upload_bytes: int
    max_image_pixels: int
    low_confidence_threshold: float
    ort_threads: int
    max_concurrent_inferences: int
    api_key: str | None = field(default=None, repr=False)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        reader = _EnvReader(env)

        model_path_text = reader.text("MODEL_PATH")
        model_path = Path(model_path_text) if model_path_text else None
        # With a local model file the bucket settings are not needed at all.
        storage = None if model_path else _read_storage(reader)
        max_upload_mb = reader.number("MAX_UPLOAD_MB", 10.0, minimum=0.1, maximum=100.0)

        settings = cls(
            storage=storage,
            model_path=model_path,
            cache_dir=Path(reader.text("MODEL_CACHE_DIR") or PROJECT_ROOT / "model_cache"),
            max_upload_bytes=int(max_upload_mb * 1024 * 1024),
            # 12 MP (e.g. 4000x3000): decoding costs about 3 bytes per pixel, twice, per concurrent request
            max_image_pixels=reader.integer("MAX_IMAGE_PIXELS", 12_000_000, minimum=1_000),
            low_confidence_threshold=reader.number("LOW_CONFIDENCE_THRESHOLD", 0.6, minimum=0.0, maximum=1.0),
            ort_threads=reader.integer("ORT_THREADS", _default_threads(), minimum=1, maximum=64),
            max_concurrent_inferences=reader.integer("MAX_CONCURRENT_INFERENCES", 2, minimum=1, maximum=64),
            api_key=reader.text("API_KEY"),
            log_level=reader.choice("LOG_LEVEL", "INFO", LOG_LEVELS),
        )
        reader.raise_if_errors()
        return settings

    def describe(self) -> dict[str, object]:
        """Loggable summary: no keys, no secrets."""
        if self.model_path:
            source = f"local file {self.model_path}"
        elif self.storage:
            source = f"bucket {self.storage.bucket!r} key {self.storage.model_key!r}"
        else:
            source = "none configured"
        return {
            "model_source": source,
            "endpoint": self.storage.endpoint_url if self.storage else None,
            "cache_dir": str(self.cache_dir),
            "max_upload_mb": round(self.max_upload_bytes / 1024 / 1024, 2),
            "low_confidence_threshold": self.low_confidence_threshold,
            "ort_threads": self.ort_threads,
            "max_concurrent_inferences": self.max_concurrent_inferences,
            "api_key_required": self.api_key is not None,
        }


def _default_threads() -> int:
    """CPUs this process may use (respects cpusets), capped: ONNX Runtime stops scaling early."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS/Windows
        available = os.cpu_count() or 1
    return max(1, min(available, 4))


def _read_storage(reader: _EnvReader) -> StorageSettings | None:
    endpoint = reader.text("E2_ENDPOINT_URL", required=True)
    bucket = reader.text("E2_BUCKET", required=True)
    access_key = reader.text("E2_ACCESS_KEY_ID", required=True)
    secret_key = reader.text("E2_SECRET_ACCESS_KEY", required=True)
    model_key = (reader.text("E2_MODEL_KEY") or DEFAULT_MODEL_KEY).lstrip("/")
    region = reader.text("E2_REGION") or "us-east-1"
    style = reader.choice("E2_ADDRESSING_STYLE", "path", ADDRESSING_STYLES)
    if endpoint and not endpoint.startswith(("https://", "http://")):
        reader.errors.append("E2_ENDPOINT_URL must start with https:// (copy it from the e2 dashboard)")
    if not (endpoint and bucket and access_key and secret_key):
        return None
    return StorageSettings(endpoint.rstrip("/"), bucket, access_key, secret_key, model_key, region, style)


class _EnvReader:
    """Reads and validates variables, collecting every problem so the user fixes them in one go."""

    def __init__(self, env: Mapping[str, str] | None) -> None:
        self._env = os.environ if env is None else env
        self.errors: list[str] = []

    def text(self, name: str, *, required: bool = False) -> str | None:
        value = (self._env.get(name) or "").strip()
        if not value and required:
            self.errors.append(f"{name} is required")
        return value or None

    def integer(self, name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
        raw = self.text(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            self.errors.append(f"{name} must be an integer, got {raw!r}")
            return default
        if value < minimum or (maximum is not None and value > maximum):
            self.errors.append(f"{name} must be between {minimum} and {maximum or 'infinity'}, got {value}")
            return default
        return value

    def number(self, name: str, default: float, *, minimum: float, maximum: float) -> float:
        raw = self.text(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            self.errors.append(f"{name} must be a number, got {raw!r}")
            return default
        if not minimum <= value <= maximum:
            self.errors.append(f"{name} must be between {minimum} and {maximum}, got {value}")
            return default
        return value

    def choice(self, name: str, default: str, options: tuple[str, ...]) -> str:
        raw = self.text(name)
        if raw is None:
            return default
        if raw.upper() in options:
            return raw.upper()
        if raw.lower() in options:
            return raw.lower()
        self.errors.append(f"{name} must be one of {list(options)}, got {raw!r}")
        return default

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ConfigError("invalid configuration: " + "; ".join(self.errors))
