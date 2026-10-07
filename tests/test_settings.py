from pathlib import Path

import pytest

from src.serving.settings import ConfigError, Settings

BUCKET_ENV = {
    "E2_ENDPOINT_URL": "https://s3.example.com",
    "E2_BUCKET": "models-bucket",
    "E2_ACCESS_KEY_ID": "AKIA-EXAMPLE",
    "E2_SECRET_ACCESS_KEY": "s3cret-value",
}


def test_bucket_settings_with_defaults():
    settings = Settings.from_env(BUCKET_ENV)
    assert settings.storage is not None
    assert settings.storage.model_key == "models/best_model.onnx"
    assert settings.storage.addressing_style == "path"
    assert settings.model_path is None
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.api_key is None


def test_every_missing_variable_is_reported_at_once():
    with pytest.raises(ConfigError) as err:
        Settings.from_env({})
    for name in BUCKET_ENV:
        assert name in str(err.value)


def test_blank_values_count_as_missing():
    with pytest.raises(ConfigError, match="E2_BUCKET"):
        Settings.from_env({**BUCKET_ENV, "E2_BUCKET": "   "})


def test_model_path_makes_bucket_settings_optional():
    settings = Settings.from_env({"MODEL_PATH": "/data/model.onnx"})
    assert settings.storage is None
    assert settings.model_path == Path("/data/model.onnx")


def test_endpoint_must_be_a_url():
    with pytest.raises(ConfigError, match="https://"):
        Settings.from_env({**BUCKET_ENV, "E2_ENDPOINT_URL": "s3.example.com"})


def test_trailing_slashes_are_normalised():
    env = {**BUCKET_ENV, "E2_ENDPOINT_URL": "https://s3.example.com/", "E2_MODEL_KEY": "/models/x.onnx"}
    storage = Settings.from_env(env).storage
    assert storage is not None
    assert storage.endpoint_url == "https://s3.example.com"
    assert storage.model_key == "models/x.onnx"


def test_malformed_numbers_and_choices_are_all_reported():
    env = {
        **BUCKET_ENV,
        "MAX_UPLOAD_MB": "lots",
        "LOW_CONFIDENCE_THRESHOLD": "2",
        "E2_ADDRESSING_STYLE": "sideways",
        "ORT_THREADS": "0",
    }
    with pytest.raises(ConfigError) as err:
        Settings.from_env(env)
    for name in ("MAX_UPLOAD_MB", "LOW_CONFIDENCE_THRESHOLD", "E2_ADDRESSING_STYLE", "ORT_THREADS"):
        assert name in str(err.value)


def test_secrets_never_appear_in_repr_or_describe():
    settings = Settings.from_env({**BUCKET_ENV, "API_KEY": "topsecret"})
    text = repr(settings) + str(settings.describe())
    for secret in ("AKIA-EXAMPLE", "s3cret-value", "topsecret"):
        assert secret not in text
    assert settings.describe()["api_key_required"] is True
