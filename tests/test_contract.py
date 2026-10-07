import pytest

from src.serving.contract import ModelContractError, ModelInfo


def make_info(**overrides) -> ModelInfo:
    fields = {"model_version": "v1", "arch": "resnet18", "class_names": ("a", "b"), "input_size": 64}
    fields.update(overrides)
    return ModelInfo(**fields)


def test_metadata_round_trip():
    info = make_info(normal_class="a", mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), created_at="t", git_commit="abc")
    assert ModelInfo.from_metadata(info.to_metadata()) == info


def test_model_without_normal_class_round_trips_to_none():
    assert ModelInfo.from_metadata(make_info().to_metadata()).normal_class is None


def test_missing_required_keys_are_named():
    meta = make_info().to_metadata()
    del meta["class_names"], meta["arch"]
    with pytest.raises(ModelContractError, match=r"arch.*class_names"):
        ModelInfo.from_metadata(meta)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", "99", "not supported"),
        ("class_names", "not json", "malformed"),
        ("class_names", "5", "malformed"),
        ("class_names", "[]", "non-empty"),
        ("class_names", '["a", "a"]', "duplicates"),
        ("input_size", "ten", "malformed"),
        ("input_size", "4", "implausibly small"),
        ("normal_class", "zzz", "not one of"),
    ],
)
def test_invalid_values_are_rejected(key, value, message):
    meta = make_info().to_metadata()
    meta[key] = value
    with pytest.raises(ModelContractError, match=message):
        ModelInfo.from_metadata(meta)
