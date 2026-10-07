"""Fetch the model from the S3-compatible bucket (IDrive e2): step 2 of the startup sequence.

HEAD the object -> reuse the cached copy if it is the same object -> otherwise GET it, hashing
while streaming, compare with the sha256 the publisher stored on the object, and move the file
into place atomically. Only HeadObject and GetObject are used, so any S3-compatible store works.

Check your keys and endpoint from your own machine, before deploying:
    python -m src.serving.storage        (reads the same E2_* variables the container reads)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    HTTPClientError,
    IncompleteReadError,
)
from botocore.exceptions import ConnectionError as BotoConnectionError

from src.serving.settings import ConfigError, Settings, StorageSettings

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (1.0, 3.0)
_TRANSIENT_CODES = {"SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable", "Throttling"}
_NETWORK_ERRORS = (EndpointConnectionError, BotoConnectionError, HTTPClientError, IncompleteReadError)


class ModelFetchError(RuntimeError):
    """The model could not be fetched or failed verification. The message says what to check."""


class _Transient(Exception):
    """Worth another attempt: network blip, 5xx, or a transfer that arrived damaged."""


@dataclass(frozen=True)
class FetchedModel:
    path: Path
    sha256: str
    size: int
    etag: str
    from_cache: bool
    bucket: str
    key: str

    def describe(self) -> dict[str, object]:
        return {
            "kind": "bucket",
            "bucket": self.bucket,
            "key": self.key,
            "etag": self.etag,
            "sha256": self.sha256,
            "size_bytes": self.size,
            "from_cache": self.from_cache,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_s3_client(cfg: StorageSettings) -> Any:
    """boto3 client for an S3-compatible endpoint. Also used by the publish script."""
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name=cfg.region,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": cfg.addressing_style},
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
            # boto3 >= 1.36 attaches CRC checksums to every request by default, and several
            # S3-compatible stores reject them. "when_required" restores plain S3 behaviour;
            # integrity is covered by the sha256 we verify ourselves.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def fetch_model(
    cfg: StorageSettings,
    cache_dir: Path,
    *,
    client: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchedModel:
    """Return a verified local copy of the model object, downloading it only when it changed."""
    client = client or make_s3_client(cfg)
    cache_dir = _writable_dir(cache_dir)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _fetch_once(client, cfg, cache_dir)
        except _Transient as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)]
            logger.warning("Model fetch attempt %d/%d failed: %s. Retrying in %.0fs", attempt, MAX_ATTEMPTS, exc, delay)
            sleep(delay)
        except OSError as exc:  # disk full, permissions: retrying cannot help
            raise ModelFetchError(f"cannot write the model into {cache_dir}: {exc}") from exc
    raise ModelFetchError(f"gave up after {MAX_ATTEMPTS} attempts: {last_error}") from last_error


def _fetch_once(client: Any, cfg: StorageSettings, cache_dir: Path) -> FetchedModel:
    dest = cache_dir / Path(cfg.model_key).name
    meta_path = dest.with_name(dest.name + ".meta.json")

    head = _call(client.head_object, cfg, "HeadObject", Bucket=cfg.bucket, Key=cfg.model_key)
    etag = str(head.get("ETag", "")).strip('"')
    size = int(head["ContentLength"])
    remote_sha = (head.get("Metadata") or {}).get("sha256")

    cached = _valid_cache(dest, meta_path, cfg, etag=etag, size=size, remote_sha=remote_sha)
    if cached is not None:
        logger.info("Cached model is up to date (%s), skipping download", _human(size))
        return cached

    logger.info("Downloading %s from bucket %r (%s)", cfg.model_key, cfg.bucket, _human(size))
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        response = _call(client.get_object, cfg, "GetObject", Bucket=cfg.bucket, Key=cfg.model_key)
        digest = hashlib.sha256()
        written = 0
        try:
            with tmp.open("wb") as fh:
                for chunk in _stream(response["Body"]):
                    fh.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
        finally:
            response["Body"].close()

        if written != int(response["ContentLength"]):
            raise _Transient(f"truncated download: got {written} of {response['ContentLength']} bytes")
        sha = digest.hexdigest()
        recorded = (response.get("Metadata") or {}).get("sha256")
        if recorded and sha != recorded:
            raise _Transient(f"sha256 mismatch: downloaded {sha[:12]}..., bucket record says {recorded[:12]}...")
        if not recorded:
            logger.warning(
                "The object has no sha256 metadata, so only its size was verified. "
                "Upload with `python -m src.deploy.publish_model` to record one."
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)

    new_etag = str(response.get("ETag", etag)).strip('"')
    meta = {"etag": new_etag, "sha256": sha, "size": written, "key": cfg.model_key, "at": datetime.now(UTC).isoformat()}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return FetchedModel(dest, sha, written, new_etag, False, cfg.bucket, cfg.model_key)


def _stream(body: Any) -> Iterator[bytes]:
    """Chunks of the response body. A dropped connection becomes a retryable error here, so that
    only genuine disk errors (raised by the caller's write) are treated as permanent."""
    try:
        yield from body.iter_chunks(CHUNK_BYTES)
    except (*_NETWORK_ERRORS, OSError) as exc:  # reset, timeout, truncated stream, ...
        raise _Transient(f"download interrupted ({type(exc).__name__})") from exc


def _valid_cache(
    dest: Path, meta_path: Path, cfg: StorageSettings, *, etag: str, size: int, remote_sha: str | None
) -> FetchedModel | None:
    """The cached file counts only if it is provably the object that is in the bucket now."""
    if not (dest.is_file() and meta_path.is_file() and dest.stat().st_size == size):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not etag or meta.get("etag") != etag:
        return None
    local_sha = sha256_file(dest)  # about 0.1 s for a 50 MB model; guards against a damaged cache
    if local_sha != meta.get("sha256") or (remote_sha and local_sha != remote_sha):
        return None
    return FetchedModel(dest, local_sha, size, etag, True, cfg.bucket, cfg.model_key)


def _call(fn: Callable[..., Any], cfg: StorageSettings, operation: str, **kwargs: Any) -> Any:
    """Run one S3 call and sort failures into 'retry' (_Transient) and 'fix your config' (ModelFetchError)."""
    try:
        return fn(**kwargs)
    except ClientError as exc:
        raise _translate(exc, cfg, operation) from exc
    except _NETWORK_ERRORS as exc:
        raise _Transient(f"{operation}: {type(exc).__name__}") from exc
    except BotoCoreError as exc:  # bad endpoint URL, bad parameters, ...: not fixable by retrying
        raise ModelFetchError(f"{operation} failed: {exc}") from exc


def _translate(exc: ClientError, cfg: StorageSettings, operation: str) -> Exception:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
    if status >= 500 or code in _TRANSIENT_CODES:
        return _Transient(f"{operation} returned {status} {code}".strip())
    if status == 404 or code in {"NoSuchKey", "NoSuchBucket"}:
        return ModelFetchError(
            f"{operation}: nothing at bucket {cfg.bucket!r}, key {cfg.model_key!r}. Check E2_BUCKET and "
            "E2_MODEL_KEY, and that a model was published (python -m src.deploy.publish_model)."
        )
    if status in {400, 401, 403} or code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
        return ModelFetchError(
            f"{operation}: access denied ({status} {code}). Check E2_ACCESS_KEY_ID / E2_SECRET_ACCESS_KEY, "
            f"that the key may read bucket {cfg.bucket!r}, and that E2_ENDPOINT_URL is the endpoint of the "
            f"bucket's region (E2_REGION={cfg.region!r}, E2_ADDRESSING_STYLE={cfg.addressing_style!r})."
        )
    return ModelFetchError(f"{operation} failed: {status} {code} {error.get('Message', '')}".strip())


def _writable_dir(path: Path) -> Path:
    """The cache directory, or a temp directory when the preferred one is read-only."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.touch()
        probe.unlink()
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "defect_model_cache"
        logger.warning("Cache directory %s is not writable (%s); using %s", path, exc, fallback)
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except OSError as inner:
            raise ModelFetchError(f"no writable directory for the model cache: {inner}") from inner
        return fallback


def _human(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def main() -> int:
    """Fetch the model exactly as the container would, then load it and print what the API will see."""
    from src.serving.inference import OnnxClassifier  # imported here: keeps `import storage` light
    from src.utils.config import setup_logging

    setup_logging()
    try:
        settings = Settings.from_env()
        if settings.storage is None:
            logger.error("MODEL_PATH is set, so there is nothing to fetch. Unset it to test the bucket.")
            return 2
        fetched = fetch_model(settings.storage, settings.cache_dir)
        info = OnnxClassifier(fetched.path, threads=1).info
    except (ConfigError, ModelFetchError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(json.dumps({"source": fetched.describe(), "model": info.to_metadata()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
