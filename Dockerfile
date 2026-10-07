# Inference image. It contains code only: the model is downloaded from the bucket at startup,
# so a new model never needs a rebuild, and the image stays small.
#
#   docker build -t defect-api .
#   docker run --rm -p 8000:8000 --env-file .env defect-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/app/model_cache

# Non-root user. UID 1000 is what Hugging Face Spaces expects and is harmless on other hosts.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dependencies first, so this layer is rebuilt only when requirements-serving.txt changes.
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Only what the service imports: no training code, no data, no weights.
COPY src/__init__.py src/__init__.py
COPY src/utils src/utils
COPY src/serving src/serving
RUN mkdir -p "$MODEL_CACHE_DIR" && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# The port opens only after the model is loaded (see src/serving/app.py), so this also means "ready".
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Shell form so ${PORT} expands (most hosts inject it); exec so uvicorn gets SIGTERM and drains requests.
# One worker: every worker would hold its own copy of the model, and the thread pool already runs
# ONNX Runtime outside the GIL.
CMD ["sh", "-c", "exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log --timeout-graceful-shutdown 20"]
