"""Publish a trained ONNX model to the bucket, promote an older version, or list versions.

    python -m src.deploy.publish_model                    upload models/best_model.onnx
    python -m src.deploy.publish_model --model PATH       upload a different file
    python -m src.deploy.publish_model --dry-run          check the file and print the plan, change nothing
    python -m src.deploy.publish_model --list             show the published versions
    python -m src.deploy.publish_model --promote VERSION  make an older (or any) version current again

Layout in the bucket (with the default E2_MODEL_KEY):
    models/best_model.onnx                       current: the one object every container start downloads
    models/archive/<version>/best_model.onnx     immutable copy of each published version, for rollback

Every object carries its sha256 as user metadata; the service verifies it after downloading.
Only plain S3 calls are used (PutObject, GetObject, HeadObject, ListObjectsV2).

Containers read the current model when they start, so after publishing, redeploy or restart the
service. Use a key with WRITE access here, and give the container a separate read-only key.
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from src.serving.contract import ModelContractError, ModelInfo
from src.serving.inference import OnnxClassifier
from src.serving.settings import DEFAULT_MODEL_KEY, ConfigError, StorageSettings
from src.serving.storage import make_s3_client, sha256_file
from src.utils.config import ONNX_PATH, setup_logging

logger = logging.getLogger(__name__)

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # ends up in an S3 key, so keep it boring


class PublishError(RuntimeError):
    """Publishing was refused or failed verification. The message says why."""


@dataclass(frozen=True)
class LocalModel:
    path: Path
    info: ModelInfo
    sha256: str
    size: int

    @property
    def version(self) -> str:
        return self.info.model_version


@dataclass(frozen=True)
class RemoteVersion:
    version: str
    size: int
    last_modified: datetime
    sha256: str | None
    is_current: bool


def archive_prefix(model_key: str) -> str:
    """'models/best_model.onnx' -> 'models/archive/'."""
    folder = posixpath.dirname(model_key)
    return f"{folder}/archive/" if folder else "archive/"


def archive_key(model_key: str, version: str) -> str:
    return f"{archive_prefix(model_key)}{version}/{posixpath.basename(model_key)}"


def inspect_local_model(path: Path) -> LocalModel:
    """Refuse to publish a file the service would refuse to start with: load and warm it up here."""
    if not path.is_file():
        raise PublishError(f"{path} does not exist. Train and export first: python -m src.models.export_onnx")
    classifier = OnnxClassifier(path)  # checks metadata and graph shapes
    classifier.warmup()  # and runs it once
    info = classifier.info
    if not _VERSION_PATTERN.match(info.model_version):
        raise PublishError(f"model_version {info.model_version!r} must match {_VERSION_PATTERN.pattern}")
    return LocalModel(path=path, info=info, sha256=sha256_file(path), size=path.stat().st_size)


def remote_sha256(client: Any, bucket: str, key: str) -> str | None:
    """The sha256 recorded on an object, or None when the object is missing or has no record."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None
        raise
    return (head.get("Metadata") or {}).get("sha256")


def upload(client: Any, bucket: str, key: str, path: Path, *, sha256: str, version: str) -> None:
    """Upload one object with its checksum, then read it back and confirm the record."""
    logger.info("Uploading %s (%.1f MB) to %s", path.name, path.stat().st_size / 1024 / 1024, key)
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"Metadata": {"sha256": sha256, "model-version": version}, "ContentType": "application/octet-stream"},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    recorded = (head.get("Metadata") or {}).get("sha256")
    if head["ContentLength"] != path.stat().st_size or recorded != sha256:
        raise PublishError(f"verification failed for {key}: the stored object does not match the local file")


def publish(client: Any, cfg: StorageSettings, local: LocalModel, *, dry_run: bool = False) -> None:
    archived = archive_key(cfg.model_key, local.version)
    logger.info("Model version %s, sha256 %s...", local.version, local.sha256[:12])
    if dry_run:
        logger.info("Dry run. Would upload %s to:\n  %s (archive)\n  %s (current)", local.path, archived, cfg.model_key)
        return

    existing = remote_sha256(client, cfg.bucket, archived)
    if existing is None:
        upload(client, cfg.bucket, archived, local.path, sha256=local.sha256, version=local.version)
    elif existing == local.sha256:
        logger.info("Version %s is already archived, skipping that upload", local.version)
    else:
        raise PublishError(
            f"version {local.version} is already published with different content. Versions are immutable; "
            "export again to get a new version."
        )

    if remote_sha256(client, cfg.bucket, cfg.model_key) == local.sha256:
        logger.info("%s already holds this model", cfg.model_key)
    else:
        upload(client, cfg.bucket, cfg.model_key, local.path, sha256=local.sha256, version=local.version)
    logger.info("Done. Redeploy or restart the service so it downloads the new current model.")


def promote(client: Any, cfg: StorageSettings, version: str) -> None:
    """Copy an archived version over the current key (rollback). Containers pick it up on restart."""
    if not _VERSION_PATTERN.match(version):
        raise PublishError(f"{version!r} is not a valid version name")
    source = archive_key(cfg.model_key, version)
    expected = remote_sha256(client, cfg.bucket, source)
    if expected is None:
        known = ", ".join(row.version for row in list_versions(client, cfg)) or "none"
        raise PublishError(f"version {version!r} is not in the archive (or has no sha256). Published versions: {known}")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / posixpath.basename(cfg.model_key)
        client.download_file(cfg.bucket, source, str(local))
        if sha256_file(local) != expected:
            raise PublishError(f"archived copy of {version} is damaged (sha256 mismatch); not promoting it")
        upload(client, cfg.bucket, cfg.model_key, local, sha256=expected, version=version)
    logger.info("Version %s is now current. Redeploy or restart the service to pick it up.", version)


def list_versions(client: Any, cfg: StorageSettings) -> list[RemoteVersion]:
    """Archived versions, newest first. The one whose checksum equals the current object is marked."""
    prefix = archive_prefix(cfg.model_key)
    filename = posixpath.basename(cfg.model_key)
    current_sha = remote_sha256(client, cfg.bucket, cfg.model_key)
    rows: list[RemoteVersion] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=cfg.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            parts = obj["Key"][len(prefix) :].split("/")
            if len(parts) != 2 or parts[1] != filename:
                continue
            sha = remote_sha256(client, cfg.bucket, obj["Key"])
            is_current = sha is not None and sha == current_sha
            rows.append(RemoteVersion(parts[0], obj["Size"], obj["LastModified"], sha, is_current))
    return sorted(rows, key=lambda row: row.last_modified, reverse=True)


def _print_versions(rows: list[RemoteVersion]) -> None:
    if not rows:
        print("No versions published yet.")
        return
    print(f"  {'version':<34}{'size':>10}  {'published (UTC)':<20}  sha256")
    for row in rows:
        marker = "*" if row.is_current else " "
        stamp = row.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{marker} {row.version:<34}{row.size / 1024 / 1024:>8.1f}MB  {stamp:<20}  {(row.sha256 or '-')[:12]}")
    print("\n* = current (what a restarted container will download)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the trained ONNX model to the bucket.")
    parser.add_argument("--model", type=Path, default=ONNX_PATH, help="ONNX file to publish (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="check the file and show the plan; no network")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="list published versions")
    action.add_argument("--promote", metavar="VERSION", help="make an archived version current again")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    try:
        if args.dry_run and not (args.list or args.promote):
            local = inspect_local_model(args.model)
            model_key = os.environ.get("E2_MODEL_KEY") or DEFAULT_MODEL_KEY
            publish(None, StorageSettings("https://offline", "<bucket>", "", "", model_key), local, dry_run=True)
            return 0

        cfg = StorageSettings.from_env()
        client = make_s3_client(cfg)
        if args.list:
            _print_versions(list_versions(client, cfg))
        elif args.promote:
            promote(client, cfg, args.promote)
        else:
            publish(client, cfg, inspect_local_model(args.model))
    except (ConfigError, PublishError, ModelContractError) as exc:
        logger.error("%s", exc)
        return 1
    except ClientError as exc:
        error = exc.response.get("Error", {})
        logger.error(
            "The bucket refused the request: %s %s. Check the key has write access to this bucket.",
            error.get("Code"),
            error.get("Message", ""),
        )
        return 1
    except (BotoCoreError, OSError) as exc:
        logger.error("Could not reach the bucket: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
