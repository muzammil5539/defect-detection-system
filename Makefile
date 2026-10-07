# The workflow, one target per stage:  data -> train -> evaluate -> export -> publish
#
#   make pipeline                              everything up to and including publishing the model
#   make train ARGS="--arch mobilenet_v3_large --epochs 20"     extra command-line arguments
#   make train PYTHON="uv run python"          use uv instead of the active interpreter
PYTHON ?= python
IMAGE ?= defect-api
ARGS ?=

.PHONY: pipeline data train evaluate export publish versions check-bucket serve test lint docker-build docker-run

pipeline: data train evaluate export publish

data:
	$(PYTHON) -m src.data.make_dataset $(ARGS)

train:
	$(PYTHON) -m src.models.train $(ARGS)

evaluate:
	$(PYTHON) -m src.models.evaluate $(ARGS)

export:
	$(PYTHON) -m src.models.export_onnx $(ARGS)

# Needs the E2_* variables with a WRITE key. Redeploy the service afterwards.
publish:
	$(PYTHON) -m src.deploy.publish_model $(ARGS)

versions:
	$(PYTHON) -m src.deploy.publish_model --list

# Fetch the model exactly as the container would: checks your endpoint, bucket and keys.
check-bucket:
	$(PYTHON) -m src.serving.storage

# Run the API locally. Set MODEL_PATH=models/best_model.onnx to skip the bucket.
serve:
	$(PYTHON) -m src.serving.app

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE)
