"""The contract between the exporter and the inference service.

The ONNX file describes itself in its `metadata_props`, so one object in the bucket is enough:
the service never needs a second file and can never pair a model with the wrong class list.
The exporter (writer) and the service (reader) both import this module, so the keys and their
meaning are defined exactly once.

  input   float32 [batch, 3, input_size, input_size], RGB, pixel values scaled to [0, 1].
          Mean/std normalisation is part of the graph, so callers must not apply it.
  output  float32 [batch, num_classes] logits, in the order of `class_names`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA_VERSION = "1"
REQUIRED_KEYS = ("schema_version", "model_version", "arch", "class_names", "input_size")


class ModelContractError(ValueError):
    """The ONNX file does not follow the contract (missing metadata, wrong shapes, ...)."""


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    arch: str
    class_names: tuple[str, ...]
    input_size: int
    normal_class: str | None = None  # name of the "normal" class if the model has one, else None
    mean: tuple[float, ...] = ()  # informational: the normalisation already baked into the graph
    std: tuple[float, ...] = ()
    created_at: str = ""
    git_commit: str = ""
    schema_version: str = SCHEMA_VERSION

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_metadata(self) -> dict[str, str]:
        """ONNX metadata_props only hold strings, so structured values are stored as JSON."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "arch": self.arch,
            "class_names": json.dumps(list(self.class_names)),
            "input_size": str(self.input_size),
            "normal_class": self.normal_class or "",
            "mean": json.dumps(list(self.mean)),
            "std": json.dumps(list(self.std)),
            "created_at": self.created_at,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_metadata(cls, meta: Mapping[str, str]) -> ModelInfo:
        missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
        if missing:
            raise ModelContractError(
                f"ONNX metadata is missing {missing}. Export the model with `python -m src.models.export_onnx`."
            )
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ModelContractError(
                f"model metadata schema {meta['schema_version']!r} is not supported (this server reads "
                f"{SCHEMA_VERSION!r}); deploy a server version that matches the model."
            )
        try:
            class_names = tuple(json.loads(meta["class_names"]))
            input_size = int(meta["input_size"])
            mean = tuple(float(v) for v in json.loads(meta.get("mean") or "[]"))
            std = tuple(float(v) for v in json.loads(meta.get("std") or "[]"))
        except (ValueError, TypeError) as exc:
            raise ModelContractError(f"ONNX metadata is malformed: {exc}") from exc

        if not class_names or not all(isinstance(name, str) and name for name in class_names):
            raise ModelContractError("class_names must be a non-empty list of non-empty strings")
        if len(set(class_names)) != len(class_names):
            raise ModelContractError("class_names contains duplicates")
        if input_size < 16:
            raise ModelContractError(f"input_size {input_size} is implausibly small")
        normal_class = meta.get("normal_class") or None
        if normal_class is not None and normal_class not in class_names:
            raise ModelContractError(f"normal_class {normal_class!r} is not one of class_names")

        return cls(
            model_version=meta["model_version"],
            arch=meta["arch"],
            class_names=class_names,
            input_size=input_size,
            normal_class=normal_class,
            mean=mean,
            std=std,
            created_at=meta.get("created_at", ""),
            git_commit=meta.get("git_commit", ""),
            schema_version=meta["schema_version"],
        )
