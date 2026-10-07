import io

import numpy as np
import pytest
from PIL import Image

from src.serving.contract import ModelContractError
from src.serving.inference import InvalidImageError, OnnxClassifier, preprocess_image, softmax
from tests.conftest import CLASSES, build_tiny_model, solid_image

MAX_PIXELS = 10_000_000


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPreprocess:
    def test_shape_dtype_and_range(self):
        batch = preprocess_image(solid_image((255, 0, 0)), 32, MAX_PIXELS)
        assert batch.shape == (1, 3, 32, 32)
        assert batch.dtype == np.float32
        assert batch[0, 0].min() == 1.0  # red channel full
        assert batch[0, 1].max() == 0.0  # green channel empty

    @pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P", "1"])
    def test_other_colour_modes_become_rgb(self, mode):
        batch = preprocess_image(png_bytes(Image.new(mode, (20, 20))), 32, MAX_PIXELS)
        assert batch.shape == (1, 3, 32, 32)

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "WEBP"])
    def test_supported_formats(self, fmt):
        assert preprocess_image(solid_image((10, 20, 30), fmt=fmt), 32, MAX_PIXELS).shape == (1, 3, 32, 32)

    def test_grayscale_is_replicated_across_channels(self):
        batch = preprocess_image(png_bytes(Image.new("L", (16, 16), 100)), 16, MAX_PIXELS)
        assert np.array_equal(batch[0, 0], batch[0, 1])
        assert np.array_equal(batch[0, 1], batch[0, 2])

    def test_garbage_is_rejected(self):
        with pytest.raises(InvalidImageError, match="decode"):
            preprocess_image(b"definitely not an image", 32, MAX_PIXELS)

    def test_truncated_file_is_rejected(self):
        data = solid_image((1, 2, 3), size=(300, 300), fmt="JPEG")
        with pytest.raises(InvalidImageError):
            preprocess_image(data[: len(data) // 2], 32, MAX_PIXELS)

    def test_unsupported_format_is_rejected(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="GIF")
        with pytest.raises(InvalidImageError, match="unsupported image format 'GIF'"):
            preprocess_image(buffer.getvalue(), 32, MAX_PIXELS)

    def test_pixel_limit_is_enforced_before_decoding(self):
        with pytest.raises(InvalidImageError, match="above the limit"):
            preprocess_image(solid_image((0, 0, 0), size=(100, 100)), 32, max_pixels=5_000)

    def test_sixteen_bit_images_are_rejected_not_misread(self):
        image = Image.fromarray(np.zeros((8, 8), dtype=np.uint16))
        with pytest.raises(InvalidImageError, match="8-bit"):
            preprocess_image(png_bytes(image), 32, MAX_PIXELS)


def test_softmax_is_stable_for_huge_logits():
    probs = softmax(np.array([[1000.0, 1000.0, 0.0]]))
    assert np.isfinite(probs).all()
    assert probs[0].tolist() == pytest.approx([0.5, 0.5, 0.0])


class TestPredict:
    @pytest.mark.parametrize(
        ("color", "expected"),
        [((255, 0, 0), "red"), ((0, 255, 0), "green"), ((0, 0, 255), "blue")],
    )
    def test_picks_the_dominant_channel(self, classifier, color, expected):
        prediction = classifier.predict(solid_image(color))
        assert prediction.predicted_class == expected
        assert prediction.confidence > 0.99
        assert prediction.needs_review is False
        assert set(prediction.probabilities) == set(CLASSES)
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
        assert prediction.inference_ms >= 0

    def test_ambiguous_image_is_flagged_for_review(self, classifier):
        prediction = classifier.predict(solid_image((128, 128, 128)))
        assert prediction.confidence == pytest.approx(1 / 3, abs=1e-3)
        assert prediction.needs_review is True

    def test_model_without_normal_class_always_reports_defective(self, classifier):
        assert classifier.predict(solid_image((255, 0, 0))).is_defective is True

    def test_normal_class_decides_is_defective(self, tmp_path):
        path = build_tiny_model(tmp_path / "m.onnx", normal_class="green")
        clf = OnnxClassifier(path)
        assert clf.predict(solid_image((0, 255, 0))).is_defective is False
        assert clf.predict(solid_image((255, 0, 0))).is_defective is True

    def test_threshold_is_configurable(self, tiny_model):
        strict = OnnxClassifier(tiny_model, low_confidence_threshold=1.0)
        assert strict.predict(solid_image((255, 0, 0))).needs_review is True


class TestLoading:
    def test_warmup_passes_on_a_valid_model(self, classifier):
        classifier.warmup()

    def test_model_without_metadata_is_rejected(self, tmp_path):
        path = build_tiny_model(tmp_path / "m.onnx", with_metadata=False)
        with pytest.raises(ModelContractError, match="missing"):
            OnnxClassifier(path)

    def test_non_onnx_file_is_rejected(self, tmp_path):
        path = tmp_path / "m.onnx"
        path.write_bytes(b"this is not a model")
        with pytest.raises(ModelContractError, match="could not load"):
            OnnxClassifier(path)

    def test_output_width_must_match_the_class_list(self, tmp_path):
        path = build_tiny_model(tmp_path / "m.onnx", logits_dim=5)
        with pytest.raises(ModelContractError, match="output must be"):
            OnnxClassifier(path)
