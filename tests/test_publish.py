from unittest.mock import MagicMock

import pytest

from src.deploy import publish_model as pm
from src.serving.contract import ModelContractError
from src.serving.settings import StorageSettings
from src.serving.storage import fetch_model, make_s3_client, sha256_file
from tests.conftest import build_tiny_model, put_model, solid_image


@pytest.fixture
def client(storage_cfg):
    return make_s3_client(storage_cfg)


def make_version(tmp_path, version: str):
    """A valid model file whose embedded model_version is `version` (each in its own folder)."""
    folder = tmp_path / version
    folder.mkdir()
    return build_tiny_model(folder / "best_model.onnx", version=version)


class TestKeys:
    def test_archive_layout(self):
        assert pm.archive_prefix("models/best_model.onnx") == "models/archive/"
        assert pm.archive_key("models/best_model.onnx", "v1") == "models/archive/v1/best_model.onnx"

    def test_key_without_a_folder(self):
        assert pm.archive_key("best_model.onnx", "v1") == "archive/v1/best_model.onnx"


class TestPublish:
    def test_uploads_the_archive_copy_and_the_current_object(self, client, storage_cfg, tmp_path):
        path = make_version(tmp_path, "v1")
        local = pm.inspect_local_model(path)
        pm.publish(client, storage_cfg, local)

        sha = sha256_file(path)
        for key in ("models/best_model.onnx", "models/archive/v1/best_model.onnx"):
            head = client.head_object(Bucket=storage_cfg.bucket, Key=key)
            assert head["Metadata"]["sha256"] == sha
            assert head["Metadata"]["model-version"] == "v1"
            assert head["ContentLength"] == path.stat().st_size

    def test_what_is_published_is_what_a_container_downloads(self, client, storage_cfg, tmp_path):
        path = make_version(tmp_path, "v1")
        pm.publish(client, storage_cfg, pm.inspect_local_model(path))
        fetched = fetch_model(storage_cfg, tmp_path / "cache", sleep=lambda _s: None)
        assert fetched.path.read_bytes() == path.read_bytes()

    def test_publishing_twice_uploads_nothing_the_second_time(self, client, storage_cfg, tmp_path):
        local = pm.inspect_local_model(make_version(tmp_path, "v1"))
        pm.publish(client, storage_cfg, local)
        spy = MagicMock(wraps=client)
        pm.publish(spy, storage_cfg, local)
        assert spy.upload_file.call_count == 0

    def test_a_new_version_becomes_current_and_the_old_one_stays_archived(self, client, storage_cfg, tmp_path):
        v1, v2 = make_version(tmp_path, "v1"), make_version(tmp_path, "v2")
        pm.publish(client, storage_cfg, pm.inspect_local_model(v1))
        pm.publish(client, storage_cfg, pm.inspect_local_model(v2))

        current = client.get_object(Bucket=storage_cfg.bucket, Key="models/best_model.onnx")["Body"].read()
        assert current == v2.read_bytes()
        archived = client.get_object(Bucket=storage_cfg.bucket, Key="models/archive/v1/best_model.onnx")["Body"].read()
        assert archived == v1.read_bytes()

    def test_dry_run_changes_nothing(self, client, storage_cfg, tmp_path):
        pm.publish(client, storage_cfg, pm.inspect_local_model(make_version(tmp_path, "v1")), dry_run=True)
        assert "Contents" not in client.list_objects_v2(Bucket=storage_cfg.bucket)

    def test_a_version_cannot_be_overwritten_with_different_content(self, client, storage_cfg, tmp_path):
        pm.publish(client, storage_cfg, pm.inspect_local_model(make_version(tmp_path, "v1")))
        other = build_tiny_model(tmp_path / "other.onnx", version="v1", classes=("red", "green", "blue", "grey"))
        with pytest.raises(pm.PublishError, match="immutable"):
            pm.publish(client, storage_cfg, pm.inspect_local_model(other))


class TestValidation:
    def test_a_model_the_service_would_reject_is_not_published(self, tmp_path):
        bad = build_tiny_model(tmp_path / "bad.onnx", with_metadata=False)
        with pytest.raises(ModelContractError, match="missing"):
            pm.inspect_local_model(bad)

    def test_missing_file_has_a_helpful_message(self, tmp_path):
        with pytest.raises(pm.PublishError, match="export_onnx"):
            pm.inspect_local_model(tmp_path / "nope.onnx")

    def test_version_names_must_be_safe_for_object_keys(self, tmp_path):
        path = build_tiny_model(tmp_path / "m.onnx", version="../escape")
        with pytest.raises(pm.PublishError, match="must match"):
            pm.inspect_local_model(path)


class TestPromoteAndList:
    def published(self, client, cfg, tmp_path):
        v1, v2 = make_version(tmp_path, "v1"), make_version(tmp_path, "v2")
        pm.publish(client, cfg, pm.inspect_local_model(v1))
        pm.publish(client, cfg, pm.inspect_local_model(v2))
        return v1, v2

    def test_promote_rolls_back_to_an_older_version(self, client, storage_cfg, tmp_path):
        v1, _ = self.published(client, storage_cfg, tmp_path)
        pm.promote(client, storage_cfg, "v1")

        body = client.get_object(Bucket=storage_cfg.bucket, Key="models/best_model.onnx")["Body"].read()
        assert body == v1.read_bytes()
        head = client.head_object(Bucket=storage_cfg.bucket, Key="models/best_model.onnx")
        assert head["Metadata"]["sha256"] == sha256_file(v1)
        assert head["Metadata"]["model-version"] == "v1"

    def test_promote_unknown_version_lists_the_known_ones(self, client, storage_cfg, tmp_path):
        self.published(client, storage_cfg, tmp_path)
        with pytest.raises(pm.PublishError, match="Published versions: .*v1"):
            pm.promote(client, storage_cfg, "v9")

    def test_promote_refuses_a_damaged_archive_copy(self, client, storage_cfg, tmp_path):
        path = make_version(tmp_path, "v1")
        put_model(storage_cfg, path, sha256="0" * 64, key="models/archive/v1/best_model.onnx")
        with pytest.raises(pm.PublishError, match="damaged"):
            pm.promote(client, storage_cfg, "v1")

    def test_list_marks_the_current_version(self, client, storage_cfg, tmp_path):
        self.published(client, storage_cfg, tmp_path)
        rows = pm.list_versions(client, storage_cfg)
        assert {row.version for row in rows} == {"v1", "v2"}
        assert [row.version for row in rows if row.is_current] == ["v2"]

        pm.promote(client, storage_cfg, "v1")
        assert [row.version for row in pm.list_versions(client, storage_cfg) if row.is_current] == ["v1"]

    def test_list_on_an_empty_bucket(self, client, storage_cfg):
        assert pm.list_versions(client, storage_cfg) == []


class TestCli:
    @pytest.fixture
    def e2_env(self, monkeypatch, storage_cfg: StorageSettings):
        monkeypatch.setenv("E2_ENDPOINT_URL", storage_cfg.endpoint_url)
        monkeypatch.setenv("E2_BUCKET", storage_cfg.bucket)
        monkeypatch.setenv("E2_ACCESS_KEY_ID", storage_cfg.access_key_id)
        monkeypatch.setenv("E2_SECRET_ACCESS_KEY", storage_cfg.secret_access_key)

    def test_publish_then_list_then_rollback(self, e2_env, tmp_path, capsys, client, storage_cfg):
        v1, v2 = make_version(tmp_path, "v1"), make_version(tmp_path, "v2")
        assert pm.main(["--model", str(v1)]) == 0
        assert pm.main(["--model", str(v2)]) == 0

        assert pm.main(["--list"]) == 0
        listing = capsys.readouterr().out
        assert "* v2" in listing and "  v1" in listing

        assert pm.main(["--promote", "v1"]) == 0
        body = client.get_object(Bucket=storage_cfg.bucket, Key="models/best_model.onnx")["Body"].read()
        assert body == v1.read_bytes()

    def test_dry_run_needs_no_credentials(self, monkeypatch, tmp_path):
        for name in ("E2_ENDPOINT_URL", "E2_BUCKET", "E2_ACCESS_KEY_ID", "E2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert pm.main(["--dry-run", "--model", str(make_version(tmp_path, "v1"))]) == 0

    def test_missing_credentials_are_reported(self, monkeypatch, tmp_path, caplog):
        for name in ("E2_ENDPOINT_URL", "E2_BUCKET", "E2_ACCESS_KEY_ID", "E2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert pm.main(["--model", str(make_version(tmp_path, "v1"))]) == 1

    def test_unreadable_model_exits_non_zero(self, e2_env, tmp_path):
        assert pm.main(["--model", str(tmp_path / "missing.onnx")]) == 1


def test_published_model_serves_predictions(client, storage_cfg, tmp_path):
    """The whole hand-off in one test: publish -> container-style fetch -> predict."""
    from src.serving.inference import OnnxClassifier

    path = make_version(tmp_path, "v1")
    pm.publish(client, storage_cfg, pm.inspect_local_model(path))
    fetched = fetch_model(storage_cfg, tmp_path / "cache", sleep=lambda _s: None)
    assert OnnxClassifier(fetched.path).predict(solid_image((0, 0, 255))).predicted_class == "blue"
