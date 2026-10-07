import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.contract import ModelContractError
from src.serving.settings import ConfigError, Settings
from src.serving.storage import ModelFetchError
from tests.conftest import CLASSES, build_tiny_model, put_model, solid_image


@pytest.fixture
def client(local_settings, classifier):
    with TestClient(create_app(local_settings(), classifier)) as test_client:
        yield test_client


def post_image(client: TestClient, data: bytes, content_type: str = "image/png", **kwargs):
    return client.post("/predict", files={"file": ("sample.png", data, content_type)}, **kwargs)


class TestPredict:
    def test_returns_class_and_confidence(self, client):
        response = post_image(client, solid_image((255, 0, 0)))
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] == "red"
        assert body["confidence"] > 0.99
        assert body["is_defective"] is True
        assert body["needs_review"] is False
        assert set(body["probabilities"]) == set(CLASSES)
        assert body["model_version"] == "test-1"
        assert body["inference_ms"] >= 0

    def test_jpeg_is_accepted(self, client):
        response = post_image(client, solid_image((0, 0, 255), fmt="JPEG"), "image/jpeg")
        assert response.json()["predicted_class"] == "blue"

    def test_the_declared_content_type_is_not_trusted(self, client):
        response = post_image(client, solid_image((0, 255, 0)), "application/octet-stream")
        assert response.status_code == 200
        assert response.json()["predicted_class"] == "green"

    def test_ambiguous_image_is_flagged_for_review(self, client):
        assert post_image(client, solid_image((128, 128, 128))).json()["needs_review"] is True

    def test_missing_file_field_is_a_validation_error(self, client):
        response = client.post("/predict")
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert "file" in error["message"]

    def test_empty_file_is_rejected(self, client):
        response = post_image(client, b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "empty_file"

    def test_non_image_is_rejected(self, client):
        response = post_image(client, b"just some text", "text/plain")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_image"

    def test_oversized_upload_is_refused_up_front(self, local_settings, classifier):
        app = create_app(local_settings(MAX_UPLOAD_MB="0.1"), classifier)
        with TestClient(app) as small_limit:
            response = post_image(small_limit, b"x" * 400_000)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"

    def test_upload_limit_is_also_enforced_while_reading(self, local_settings, classifier):
        # Within limit + multipart slack, so the Content-Length pre-check lets it through.
        app = create_app(local_settings(MAX_UPLOAD_MB="0.1"), classifier)
        with TestClient(app) as small_limit:
            response = post_image(small_limit, b"x" * 120_000)
        assert response.status_code == 413


class TestRequestIds:
    def test_every_response_has_a_request_id(self, client):
        assert len(client.get("/health").headers["x-request-id"]) == 16

    def test_a_safe_caller_supplied_id_is_echoed(self, client):
        response = client.get("/health", headers={"x-request-id": "trace-123.abc"})
        assert response.headers["x-request-id"] == "trace-123.abc"

    def test_an_unsafe_id_is_replaced(self, client):
        response = client.get("/health", headers={"x-request-id": "bad id; <script>"})
        assert response.headers["x-request-id"] != "bad id; <script>"

    def test_errors_carry_the_same_id(self, client):
        response = client.post("/predict", headers={"x-request-id": "abc-1"})
        assert response.json()["error"]["request_id"] == "abc-1"


class TestAuth:
    def test_api_key_is_enforced_on_predict_only(self, local_settings, classifier):
        app = create_app(local_settings(API_KEY="letmein"), classifier)
        with TestClient(app) as guarded:
            image = solid_image((255, 0, 0))
            assert post_image(guarded, image).status_code == 401
            assert post_image(guarded, image, headers={"x-api-key": "wrong"}).status_code == 401
            assert post_image(guarded, image, headers={"x-api-key": "letmein"}).status_code == 200
            assert guarded.get("/health").status_code == 200  # probes must work without a key

    def test_open_by_default(self, client):
        assert post_image(client, solid_image((255, 0, 0))).status_code == 200


class TestOps:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok", "model_version": "test-1"}

    def test_health_answers_head_requests_for_uptime_monitors(self, client):
        response = client.head("/health")
        assert response.status_code == 200
        assert response.content == b""

    def test_model_info(self, client):
        body = client.get("/model").json()
        assert body["class_names"] == list(CLASSES)
        assert body["input_size"] == 32
        assert body["arch"] == "tiny"
        assert body["normal_class"] is None
        assert body["source"] == {"kind": "injected"}

    def test_root_redirects_to_the_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/docs"

    def test_unknown_route_uses_the_error_envelope(self, client):
        response = client.get("/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_wrong_method_uses_the_error_envelope(self, client):
        response = client.get("/predict")
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"

    def test_openapi_lists_the_endpoints(self, client):
        assert {"/predict", "/health", "/model"} <= set(client.get("/openapi.json").json()["paths"])


class TestStartup:
    """The container must refuse to start, not start without a model."""

    @staticmethod
    def bucket_settings(cfg, tmp_path) -> Settings:
        return Settings.from_env(
            {
                "E2_ENDPOINT_URL": cfg.endpoint_url,
                "E2_BUCKET": cfg.bucket,
                "E2_ACCESS_KEY_ID": cfg.access_key_id,
                "E2_SECRET_ACCESS_KEY": cfg.secret_access_key,
                "MODEL_CACHE_DIR": str(tmp_path / "cache"),
                "ORT_THREADS": "1",
            }
        )

    def test_loads_the_model_from_the_bucket(self, storage_cfg, tiny_model, tmp_path):
        sha = put_model(storage_cfg, tiny_model)
        with TestClient(create_app(self.bucket_settings(storage_cfg, tmp_path))) as booted:
            source = booted.get("/model").json()["source"]
            assert source["kind"] == "bucket"
            assert source["key"] == "models/best_model.onnx"
            assert source["sha256"] == sha
            assert post_image(booted, solid_image((255, 0, 0))).json()["predicted_class"] == "red"

    def test_refuses_to_start_when_the_bucket_has_no_model(self, storage_cfg, tmp_path):
        with (
            pytest.raises(ModelFetchError, match="E2_MODEL_KEY"),
            TestClient(create_app(self.bucket_settings(storage_cfg, tmp_path))),
        ):
            pass

    def test_refuses_to_start_when_the_file_breaks_the_contract(self, storage_cfg, tmp_path):
        bad = build_tiny_model(tmp_path / "bad.onnx", with_metadata=False)
        put_model(storage_cfg, bad)
        with (
            pytest.raises(ModelContractError, match="missing"),
            TestClient(create_app(self.bucket_settings(storage_cfg, tmp_path))),
        ):
            pass

    def test_refuses_to_start_without_configuration(self, monkeypatch):
        for name in ("MODEL_PATH", "E2_ENDPOINT_URL", "E2_BUCKET", "E2_ACCESS_KEY_ID", "E2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ConfigError, match="E2_BUCKET"), TestClient(create_app()):
            pass
