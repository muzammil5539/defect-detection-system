"""Storage tests run against a local S3 server (moto); failure modes use small stub clients."""

import errno
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from src.serving.storage import MAX_ATTEMPTS, ModelFetchError, fetch_model, make_s3_client
from tests.conftest import build_tiny_model, put_model


def no_sleep(_seconds: float) -> None:
    """Retries must not slow the suite down."""


def client_error(status: int, code: str, operation: str = "HeadObject") -> ClientError:
    response = {"Error": {"Code": code, "Message": "boom"}, "ResponseMetadata": {"HTTPStatusCode": status}}
    return ClientError(response, operation)


class FailingClient:
    """Raises `error` from head_object every time and counts the calls."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def head_object(self, **_kwargs):
        self.calls += 1
        raise self.error


def test_client_is_configured_for_s3_compatible_stores(storage_cfg):
    client = make_s3_client(storage_cfg)
    assert client.meta.config.request_checksum_calculation == "when_required"
    assert client.meta.config.response_checksum_validation == "when_required"
    assert client.meta.config.s3["addressing_style"] == "path"


class TestDownload:
    def test_downloads_and_verifies_the_model(self, storage_cfg, tiny_model, tmp_path):
        sha = put_model(storage_cfg, tiny_model)
        fetched = fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)

        assert fetched.path.read_bytes() == tiny_model.read_bytes()
        assert fetched.sha256 == sha
        assert fetched.from_cache is False
        assert not list((tmp_path / "cache").glob("*.part")), "temporary download file left behind"

    def test_second_start_reuses_the_cache(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        spy = MagicMock(wraps=make_s3_client(storage_cfg))

        fetch_model(storage_cfg, tmp_path / "cache", client=spy, sleep=no_sleep)
        second = fetch_model(storage_cfg, tmp_path / "cache", client=spy, sleep=no_sleep)

        assert second.from_cache is True
        assert spy.get_object.call_count == 1  # downloaded once
        assert spy.head_object.call_count == 2  # but checked against the bucket both times

    def test_republished_model_replaces_the_cache(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)

        # Same size, different bytes: only the ETag tells them apart.
        newer = build_tiny_model(tmp_path / "newer.onnx", version="test-2")
        put_model(storage_cfg, newer)
        refetched = fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)

        assert refetched.from_cache is False
        assert refetched.path.read_bytes() == newer.read_bytes()

    def test_damaged_cache_file_is_downloaded_again(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        first = fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)
        damaged = bytearray(first.path.read_bytes())
        damaged[-1] ^= 0xFF  # same size, one flipped byte
        first.path.write_bytes(bytes(damaged))

        again = fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)

        assert again.from_cache is False
        assert again.path.read_bytes() == tiny_model.read_bytes()

    def test_object_without_checksum_still_loads_but_warns(self, storage_cfg, tiny_model, tmp_path, caplog):
        put_model(storage_cfg, tiny_model, sha256=None)
        with caplog.at_level("WARNING"):
            fetched = fetch_model(storage_cfg, tmp_path / "cache", sleep=no_sleep)
        assert fetched.path.read_bytes() == tiny_model.read_bytes()
        assert "no sha256 metadata" in caplog.text

    def test_unwritable_cache_dir_falls_back_to_a_temp_dir(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        blocker = tmp_path / "a_file"
        blocker.write_text("not a directory")
        fetched = fetch_model(storage_cfg, blocker / "cache", sleep=no_sleep)
        assert fetched.path.read_bytes() == tiny_model.read_bytes()
        assert fetched.path.parent.name == "defect_model_cache"


class TestFailures:
    def test_missing_object_fails_at_once_with_a_hint(self, storage_cfg, tmp_path):
        sleeps: list[float] = []
        with pytest.raises(ModelFetchError, match="E2_MODEL_KEY"):
            fetch_model(storage_cfg, tmp_path / "cache", sleep=sleeps.append)
        assert sleeps == [], "a permanent error must not be retried"

    def test_checksum_mismatch_is_retried_then_fails_and_leaves_nothing_behind(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model, sha256="0" * 64)
        sleeps: list[float] = []
        with pytest.raises(ModelFetchError, match="sha256 mismatch"):
            fetch_model(storage_cfg, tmp_path / "cache", sleep=sleeps.append)

        assert len(sleeps) == MAX_ATTEMPTS - 1
        cache = tmp_path / "cache"
        assert not (cache / "best_model.onnx").exists(), "a model that failed verification must never be cached"
        assert not list(cache.glob("*.part"))

    def test_access_denied_fails_at_once_and_explains(self, storage_cfg, tmp_path):
        stub = FailingClient(client_error(403, "403"))
        with pytest.raises(ModelFetchError, match="access denied.*E2_ENDPOINT_URL"):
            fetch_model(storage_cfg, tmp_path / "cache", client=stub, sleep=no_sleep)
        assert stub.calls == 1

    @pytest.mark.parametrize("error", [client_error(503, "ServiceUnavailable"), client_error(500, "InternalError")])
    def test_server_errors_are_retried(self, storage_cfg, tmp_path, error):
        stub = FailingClient(error)
        with pytest.raises(ModelFetchError, match=f"gave up after {MAX_ATTEMPTS} attempts"):
            fetch_model(storage_cfg, tmp_path / "cache", client=stub, sleep=no_sleep)
        assert stub.calls == MAX_ATTEMPTS

    def test_connection_errors_are_retried(self, storage_cfg, tmp_path):
        stub = FailingClient(EndpointConnectionError(endpoint_url="https://down.example.com"))
        with pytest.raises(ModelFetchError, match="EndpointConnectionError"):
            fetch_model(storage_cfg, tmp_path / "cache", client=stub, sleep=no_sleep)
        assert stub.calls == MAX_ATTEMPTS

    def test_recovers_when_a_transient_error_clears(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        real = make_s3_client(storage_cfg)

        class FailsOnce:
            def __init__(self) -> None:
                self.heads = 0

            def head_object(self, **kwargs):
                self.heads += 1
                if self.heads == 1:
                    raise client_error(503, "ServiceUnavailable")
                return real.head_object(**kwargs)

            def get_object(self, **kwargs):
                return real.get_object(**kwargs)

        flaky = FailsOnce()
        fetched = fetch_model(storage_cfg, tmp_path / "cache", client=flaky, sleep=no_sleep)
        assert fetched.path.read_bytes() == tiny_model.read_bytes()
        assert flaky.heads == 2

    def test_connection_dropped_mid_download_is_retried(self, storage_cfg, tiny_model, tmp_path):
        put_model(storage_cfg, tiny_model)
        real = make_s3_client(storage_cfg)

        class DroppingBody:
            def iter_chunks(self, _size):
                yield b"partial"
                raise ConnectionResetError("peer reset the connection")

            def close(self):
                pass

        class DropsMidStream:
            def __init__(self) -> None:
                self.gets = 0

            def head_object(self, **kwargs):
                return real.head_object(**kwargs)

            def get_object(self, **kwargs):
                self.gets += 1
                response = real.get_object(**kwargs)
                response["Body"] = DroppingBody()
                return response

        stub = DropsMidStream()
        with pytest.raises(ModelFetchError, match="download interrupted"):
            fetch_model(storage_cfg, tmp_path / "cache", client=stub, sleep=no_sleep)

        assert stub.gets == MAX_ATTEMPTS
        cache = tmp_path / "cache"
        assert not (cache / "best_model.onnx").exists()
        assert not list(cache.glob("*.part")), "a half-written download must not be left behind"

    def test_a_full_disk_is_not_retried_and_says_so(self, storage_cfg, tiny_model, tmp_path, monkeypatch):
        put_model(storage_cfg, tiny_model)
        real_open = Path.open

        def open_failing_for_downloads(self, *args, **kwargs):
            if self.name.endswith(".part"):
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_failing_for_downloads)
        sleeps: list[float] = []
        with pytest.raises(ModelFetchError, match="cannot write the model"):
            fetch_model(storage_cfg, tmp_path / "cache", sleep=sleeps.append)
        assert sleeps == [], "retrying cannot help when the disk is full"
