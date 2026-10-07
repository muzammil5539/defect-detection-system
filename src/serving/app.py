"""FastAPI service: loads the ONNX model once at startup, then serves /predict.

Startup runs before the port opens (the lifespan below), so a container that cannot get a valid
model exits instead of accepting traffic: the host's health check never passes and a platform
with rolling deploys keeps serving the previous release.

Run locally:   python -m src.serving.app          (docs at http://127.0.0.1:8000/docs)
In Docker:     uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

import anyio
import anyio.to_thread
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.serving.contract import ModelContractError
from src.serving.inference import InvalidImageError, OnnxClassifier
from src.serving.schemas import ErrorResponse, HealthResponse, ModelResponse, PredictionResponse
from src.serving.settings import ConfigError, Settings
from src.serving.storage import ModelFetchError, fetch_model
from src.utils.config import setup_logging

logger = logging.getLogger(__name__)

MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and headers around the file in a multipart body
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HTTP_ERROR_CODES = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}


class ApiError(Exception):
    """An error the caller can act on: becomes a JSON body with a stable `code`."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    return JSONResponse(status_code=status_code, content=body)


def _request_id(request: Request) -> str:
    """Honour a caller's X-Request-ID for tracing, but only if it is safe to put in logs."""
    supplied = request.headers.get("x-request-id", "")
    return supplied if _REQUEST_ID_PATTERN.match(supplied) else uuid.uuid4().hex[:16]


def _megabytes(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:g} MB"


def load_classifier(settings: Settings) -> tuple[OnnxClassifier, dict[str, object]]:
    """Startup steps 2 and 3: get the model file (bucket or local path), load it, warm it up."""
    if settings.model_path is not None:
        path, source = settings.model_path, {"kind": "local", "path": str(settings.model_path)}
    else:
        assert settings.storage is not None  # Settings guarantees one of the two sources
        fetched = fetch_model(settings.storage, settings.cache_dir)
        path, source = fetched.path, fetched.describe()
    classifier = OnnxClassifier(
        path,
        threads=settings.ort_threads,
        low_confidence_threshold=settings.low_confidence_threshold,
        max_image_pixels=settings.max_image_pixels,
    )
    classifier.warmup()
    return classifier, source


def create_app(settings: Settings | None = None, classifier: OnnxClassifier | None = None) -> FastAPI:
    """Build the app. Tests inject `settings` and a ready `classifier`; production passes neither."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            cfg = settings or Settings.from_env()
        except ConfigError as exc:
            setup_logging()
            logger.error("Startup failed: %s", exc)
            raise
        setup_logging(getattr(logging, cfg.log_level))
        logger.info("Starting with %s", cfg.describe())
        app.state.settings = cfg
        app.state.limiter = anyio.CapacityLimiter(cfg.max_concurrent_inferences)
        started = time.perf_counter()
        try:
            if classifier is not None:
                app.state.classifier, app.state.model_source = classifier, {"kind": "injected"}
            else:
                app.state.classifier, app.state.model_source = load_classifier(cfg)
        except (ModelFetchError, ModelContractError) as exc:
            logger.error("Startup failed: %s", exc)
            raise
        info = app.state.classifier.info
        logger.info(
            "Model %s ready (%s, %d classes, %dpx input) in %.1fs",
            info.model_version,
            info.arch,
            info.num_classes,
            info.input_size,
            time.perf_counter() - started,
        )
        yield

    app = FastAPI(
        title="Defect detection API",
        version="1.0.0",
        description="Upload an image of a product surface; get the predicted defect class and a confidence score.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = _request_id(request)
        request.state.request_id = request_id
        started = time.perf_counter()
        limit = request.app.state.settings.max_upload_bytes
        declared = request.headers.get("content-length", "")
        if request.method == "POST" and declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD_BYTES:
            # Refuse before reading the body, so an oversized upload costs us almost nothing.
            message = f"the image must be smaller than {_megabytes(limit)}"
            response = _error_response(413, "file_too_large", message, request_id)
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("Unhandled error request_id=%s", request_id)
                response = _error_response(500, "internal_error", "unexpected server error", request_id)
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s -> %d in %.1f ms request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return _error_response(exc.status_code, exc.code, exc.message, request.state.request_id)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        code = _HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return _error_response(exc.status_code, code, str(exc.detail), request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        problems = [
            f"{'.'.join(str(part) for part in err['loc'] if part != 'body')}: {err['msg']}" for err in exc.errors()[:3]
        ]
        return _error_response(422, "validation_error", "; ".join(problems), request.state.request_id)

    def get_classifier(request: Request) -> OnnxClassifier:
        ready = getattr(request.app.state, "classifier", None)
        if ready is None:  # not reachable once startup succeeded; kept as a safety net
            raise ApiError(503, "model_not_ready", "the model is not loaded")
        return ready

    def require_api_key(request: Request) -> None:
        expected = request.app.state.settings.api_key
        if expected is None:
            return
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise ApiError(401, "unauthorized", "missing or invalid X-API-Key header")

    async def read_upload(upload: UploadFile, limit: int) -> bytes:
        """Read at most `limit` bytes. Content-Length can lie or be absent, so count what arrives."""
        chunks: list[bytes] = []
        total = 0
        while chunk := await upload.read(64 * 1024):
            total += len(chunk)
            if total > limit:
                raise ApiError(413, "file_too_large", f"the image must be smaller than {_megabytes(limit)}")
            chunks.append(chunk)
        return b"".join(chunks)

    error_docs = {
        400: {"model": ErrorResponse, "description": "Empty or unreadable image."},
        401: {"model": ErrorResponse, "description": "API key required or wrong."},
        413: {"model": ErrorResponse, "description": "Image larger than the upload limit."},
        422: {"model": ErrorResponse, "description": "The multipart field `file` is missing."},
    }

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    # HEAD too: many uptime monitors probe with HEAD, and a 405 there would read as "down".
    @app.api_route(
        "/health",
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        tags=["ops"],
        summary="Liveness and readiness",
    )
    def health(model: OnnxClassifier = Depends(get_classifier)) -> HealthResponse:
        # The port only opens after the model has loaded, so a 200 here also means "ready".
        return HealthResponse(status="ok", model_version=model.info.model_version)

    @app.get("/model", response_model=ModelResponse, tags=["ops"], summary="Which model is serving")
    def model_info(request: Request, model: OnnxClassifier = Depends(get_classifier)) -> ModelResponse:
        info = model.info
        return ModelResponse(
            model_version=info.model_version,
            arch=info.arch,
            class_names=list(info.class_names),
            input_size=info.input_size,
            normal_class=info.normal_class,
            created_at=info.created_at,
            git_commit=info.git_commit,
            source=request.app.state.model_source,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        summary="Classify one image",
        responses=error_docs,
        dependencies=[Depends(require_api_key)],
    )
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="The image: JPEG, PNG, BMP or WebP."),
        model: OnnxClassifier = Depends(get_classifier),
    ) -> PredictionResponse:
        cfg: Settings = request.app.state.settings
        data = await read_upload(file, cfg.max_upload_bytes)
        if not data:
            raise ApiError(400, "empty_file", "the uploaded file is empty")
        try:
            # CPU-bound work goes to a thread so the event loop keeps answering /health. The limiter
            # caps concurrent inferences, which protects small hosts from thrashing; excess requests wait.
            result = await anyio.to_thread.run_sync(model.predict, data, limiter=request.app.state.limiter)
        except InvalidImageError as exc:
            raise ApiError(400, "invalid_image", str(exc)) from exc

        logger.info(
            "prediction class=%s confidence=%.3f review=%s inference_ms=%.1f request_id=%s",
            result.predicted_class,
            result.confidence,
            result.needs_review,
            result.inference_ms,
            request.state.request_id,
        )
        return PredictionResponse(
            predicted_class=result.predicted_class,
            confidence=round(result.confidence, 4),
            is_defective=result.is_defective,
            needs_review=result.needs_review,
            probabilities={name: round(p, 4) for name, p in result.probabilities.items()},
            model_version=model.info.model_version,
            inference_ms=round(result.inference_ms, 2),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        access_log=False,  # the request middleware logs every request with its id
    )
