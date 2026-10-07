# ruff: noqa: E402
"""Training-side tests: model factory, preprocessing parity, and the train -> export -> serve hand-off.

They skip automatically when torch is not installed (the serving image and the serving CI job do not need it).
"""

import io

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

# Guards first: in the serving-only environment these imports must skip the file, not break collection.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pd = pytest.importorskip("pandas")

from src.deploy.publish_model import inspect_local_model
from src.models.build import ARCHS, build_model, load_checkpoint
from src.models.dataset import RandomRotate90, eval_transform
from src.models.evaluate import compute_metrics, evaluate, expected_calibration_error
from src.models.export_onnx import ExportError, export, export_graph
from src.models.train import TrainConfig, class_weights, fit
from src.serving.inference import OnnxClassifier, preprocess_image
from src.utils.config import CLASS_NAMES

pytestmark = pytest.mark.torch
SIZE = 64


def noise_png(seed: int, size: int = SIZE, brightness: int = 120) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = np.clip(rng.normal(brightness, 15, (size, size, 3)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


class TestModel:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_architecture_outputs_one_logit_per_class(self, arch):
        model = build_model(arch, len(CLASS_NAMES), [0.5] * 3, [0.2] * 3, pretrained=False).eval()
        assert model(torch.rand(2, 3, SIZE, SIZE)).shape == (2, len(CLASS_NAMES))

    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_architecture_exports_to_onnx_that_matches_pytorch(self, arch, tmp_path):
        """Operators such as hard-swish and squeeze-excite must survive the export, not only ResNet's."""
        model = build_model(arch, len(CLASS_NAMES), [0.5] * 3, [0.2] * 3, pretrained=False).eval()
        path = tmp_path / f"{arch}.onnx"
        export_graph(model, SIZE, path)

        pixels = torch.rand(3, 3, SIZE, SIZE)  # a batch of 3 also exercises the dynamic batch axis
        with torch.no_grad():
            expected = model(pixels).numpy()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        produced = session.run(["logits"], {"input": pixels.numpy()})[0]
        np.testing.assert_allclose(produced, expected, rtol=1e-3, atol=1e-3)

    def test_unknown_architecture_is_rejected(self):
        with pytest.raises(ValueError, match="unknown architecture"):
            build_model("vgg19", 6, [0.5] * 3, [0.2] * 3, pretrained=False)

    def test_normalisation_is_part_of_the_model(self):
        model = build_model("resnet18", 6, [0.5] * 3, [0.2] * 3, pretrained=False).eval()
        pixels = torch.rand(2, 3, SIZE, SIZE)
        assert torch.allclose(model(pixels), model.backbone((pixels - 0.5) / 0.2))


class TestPreprocessing:
    def test_training_eval_transform_equals_the_api_preprocessing(self):
        """The most common train/serve skew bug: these two must be bit-identical."""
        buffer = io.BytesIO()
        rng = np.random.default_rng(1)
        Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)).save(buffer, format="JPEG")
        data = buffer.getvalue()

        trained = eval_transform(32)(Image.open(io.BytesIO(data)).convert("RGB")).numpy()
        served = preprocess_image(data, 32, 10**9)[0]
        assert np.array_equal(trained, served)

    def test_rotation_augmentation_keeps_every_pixel(self):
        image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8))
        for _ in range(8):
            rotated = RandomRotate90()(image)
            assert np.array_equal(np.sort(np.asarray(rotated).ravel()), np.sort(np.asarray(image).ravel()))


class TestClassWeights:
    def test_balanced_classes_get_equal_weight(self):
        assert torch.allclose(class_weights([0, 1, 2] * 4, 3), torch.ones(3))

    def test_rare_classes_weigh_more(self):
        weights = class_weights([0] * 9 + [1], 2)
        assert weights[1] > weights[0]
        assert weights[1].item() == pytest.approx(5.0)

    def test_an_absent_class_does_not_produce_infinity(self):
        assert torch.isfinite(class_weights([0, 0, 1], 3)).all()


class TestMetrics:
    def test_false_positives_and_negatives_are_counted_per_class(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        probs = np.eye(2)[y_pred] * 0.9 + 0.05
        metrics = compute_metrics(y_true, probs, ["a", "b"])

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["per_class"]["a"]["precision"] == pytest.approx(2 / 3)
        assert metrics["per_class"]["a"]["false_negatives"] == 1  # one "a" was called "b"
        assert metrics["per_class"]["a"]["false_positives"] == 1  # one "b" was called "a"
        assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
        assert {(c["true"], c["predicted"]) for c in metrics["top_confusions"]} == {("a", "b"), ("b", "a")}

    def test_calibration_error_is_zero_when_confidence_matches_accuracy(self):
        assert expected_calibration_error(np.array([1.0, 1.0]), np.array([True, True])) == 0.0
        assert expected_calibration_error(np.array([1.0, 1.0]), np.array([False, False])) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A real (tiny) training run on synthetic images: 6 classes that differ in brightness."""
    root = tmp_path_factory.mktemp("training")
    frames = {}
    for split, per_class in (("train", 4), ("val", 2)):
        rows = []
        for index, name in enumerate(CLASS_NAMES):
            for n in range(per_class):
                path = root / f"{split}_{name}_{n}.png"
                path.write_bytes(noise_png(seed=index * 100 + n, brightness=30 + index * 35))
                rows.append({"image_path": str(path), "label": name, "label_idx": index})
        frames[split] = root / f"{split}.csv"
        pd.DataFrame(rows).to_csv(frames[split], index=False)
    stats = root / "stats.json"
    stats.write_text('{"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}', encoding="utf-8")

    cfg = TrainConfig(
        arch="resnet18",
        epochs=2,
        batch_size=8,
        image_size=SIZE,
        pretrained=False,
        workers=0,
        train_csv=frames["train"],
        val_csv=frames["val"],
        norm_stats=stats,
        output=root / "model.pt",
    )
    summary = fit(cfg, results_dir=root / "results")
    return {"cfg": cfg, "summary": summary, "root": root, "samples": [noise_png(seed=900 + i) for i in range(4)]}


class TestTrainExportServe:
    def test_training_writes_a_checkpoint_and_history(self, trained):
        assert trained["cfg"].output.is_file()
        history = pd.read_csv(trained["root"] / "results" / "training_history.csv")
        assert list(history["epoch"]) == [1, 2]
        assert (trained["root"] / "results" / "learning_curves.png").is_file()

    def test_checkpoint_rebuilds_the_same_model(self, trained):
        model, ckpt = load_checkpoint(trained["cfg"].output)
        assert ckpt["class_names"] == list(CLASS_NAMES)
        assert ckpt["image_size"] == SIZE
        assert model(torch.rand(1, 3, SIZE, SIZE)).shape == (1, len(CLASS_NAMES))

    def test_export_matches_pytorch_and_is_one_self_describing_file(self, trained):
        onnx_path = trained["root"] / "exported" / "best_model.onnx"
        report = export(
            trained["cfg"].output, onnx_path, results_dir=trained["root"] / "results", sample_images=trained["samples"]
        )

        assert report["parity"]["max_abs_logit_diff"] < 1e-3
        assert report["parity"]["max_abs_preprocess_diff"] == 0.0
        assert sorted(p.name for p in onnx_path.parent.iterdir()) == ["best_model.onnx"], "no sidecar or temp files"

        classifier = OnnxClassifier(onnx_path)
        assert classifier.info.class_names == tuple(CLASS_NAMES)
        assert classifier.info.input_size == SIZE
        assert classifier.predict(trained["samples"][0]).predicted_class in CLASS_NAMES

    def test_exported_model_passes_the_publish_checks(self, trained):
        onnx_path = trained["root"] / "for_publish" / "best_model.onnx"
        export(
            trained["cfg"].output, onnx_path, results_dir=trained["root"] / "results", sample_images=trained["samples"]
        )
        local = inspect_local_model(onnx_path)  # raises if the service would refuse this file
        assert local.version.endswith(local.info.git_commit), "the version names the commit that exported it"

    def test_a_failed_parity_check_leaves_nothing_behind(self, trained):
        onnx_path = trained["root"] / "rejected" / "best_model.onnx"
        with pytest.raises(ExportError, match="differ"):
            export(
                trained["cfg"].output,
                onnx_path,
                results_dir=trained["root"] / "results",
                sample_images=trained["samples"],
                tolerance=-1.0,
            )
        assert not onnx_path.exists()
        assert list(onnx_path.parent.iterdir()) == []

    def test_a_model_with_a_normal_class_reports_it(self, trained):
        onnx_path = trained["root"] / "normal" / "best_model.onnx"
        export(
            trained["cfg"].output,
            onnx_path,
            results_dir=trained["root"] / "results",
            sample_images=trained["samples"],
            normal_class=CLASS_NAMES[0],
        )
        assert OnnxClassifier(onnx_path).info.normal_class == CLASS_NAMES[0]

    def test_evaluation_writes_every_artifact(self, trained):
        results = trained["root"] / "eval"
        metrics = evaluate(trained["cfg"].output, trained["cfg"].val_csv, results, split="val", workers=0)

        assert metrics["n"] == 2 * len(CLASS_NAMES)
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        for name in ("metrics.json", "classification_report.txt", "confusion_matrix.png", "errors.csv"):
            assert (results / f"val_{name}").is_file(), name
