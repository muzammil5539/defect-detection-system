"""Request and response models: they also drive the OpenAPI docs served at /docs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": "scratches",
                "confidence": 0.9731,
                "is_defective": True,
                "needs_review": False,
                "probabilities": {
                    "crazing": 0.0042,
                    "inclusion": 0.0051,
                    "patches": 0.0038,
                    "pitted_surface": 0.0044,
                    "rolled-in_scale": 0.0094,
                    "scratches": 0.9731,
                },
                "model_version": "20261007T201500Z-3fa9c21",
                "inference_ms": 14.2,
            }
        }
    )

    predicted_class: str = Field(description="Most likely class.")
    confidence: float = Field(ge=0, le=1, description="Softmax probability of the predicted class.")
    is_defective: bool = Field(
        description="False only when the model has a 'normal' class and predicted it. "
        "A model trained on defect classes alone always answers true."
    )
    needs_review: bool = Field(description="True when confidence is below the configured threshold: send to a human.")
    probabilities: dict[str, float] = Field(description="Probability of every class.")
    model_version: str
    inference_ms: float = Field(description="Time spent inside ONNX Runtime, excluding upload and decoding.")


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelResponse(BaseModel):
    model_version: str
    arch: str
    class_names: list[str]
    input_size: int
    normal_class: str | None
    created_at: str
    git_commit: str
    source: dict[str, Any] = Field(description="Where the model file came from (bucket, key, sha256, cache hit).")


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
