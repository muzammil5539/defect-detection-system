# ruff: noqa: E402
"""Dataset preparation: the splits must be stratified, reproducible and free of leaks."""

import pytest
from PIL import Image

# Guards first: in the serving-only environment these imports must skip the file, not break collection.
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from src.data.make_dataset import drop_duplicate_images, find_cross_split_duplicates, split_train_val
from src.utils.config import CLASS_NAMES


def make_frame(per_class: int = 20) -> pd.DataFrame:
    rows = [
        {"image_path": f"{name}/{n:03d}.jpg", "label": name, "label_idx": index}
        for index, name in enumerate(CLASS_NAMES)
        for n in range(per_class)
    ]
    return pd.DataFrame(rows)


def test_split_is_stratified_disjoint_and_complete():
    frame = make_frame(20)
    train, val = split_train_val(frame, val_fraction=0.15)

    assert val["label"].value_counts().to_dict() == dict.fromkeys(CLASS_NAMES, 3)
    assert train["label"].value_counts().to_dict() == dict.fromkeys(CLASS_NAMES, 17)
    assert not set(train["image_path"]) & set(val["image_path"])
    assert set(train["image_path"]) | set(val["image_path"]) == set(frame["image_path"])


def test_split_is_reproducible():
    first = split_train_val(make_frame())
    second = split_train_val(make_frame())
    assert first[0].equals(second[0]) and first[1].equals(second[1])


def write_image(path, color):
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


def test_identical_files_are_dropped_before_splitting(tmp_path):
    a = write_image(tmp_path / "a.png", (10, 20, 30))
    b = write_image(tmp_path / "b.png", (10, 20, 30))  # same pixels, same bytes as a
    c = write_image(tmp_path / "c.png", (200, 20, 30))
    frame = pd.DataFrame({"image_path": [a, b, c], "label": "x", "label_idx": 0})

    kept = drop_duplicate_images(frame)

    assert list(kept["image_path"]) == [a, c]


def test_a_file_shared_between_splits_is_reported(tmp_path):
    a = write_image(tmp_path / "a.png", (1, 2, 3))
    copy = write_image(tmp_path / "copy.png", (1, 2, 3))
    other = write_image(tmp_path / "other.png", (9, 9, 9))
    splits = {
        "train": pd.DataFrame({"image_path": [a], "label": "x", "label_idx": 0}),
        "test": pd.DataFrame({"image_path": [copy, other], "label": "x", "label_idx": 0}),
    }

    found = find_cross_split_duplicates(splits)

    assert len(found) == 1 and "copy.png" in found[0]


def test_clean_splits_report_nothing(tmp_path):
    a = write_image(tmp_path / "a.png", (1, 2, 3))
    b = write_image(tmp_path / "b.png", (4, 5, 6))
    splits = {
        "train": pd.DataFrame({"image_path": [a], "label": "x", "label_idx": 0}),
        "val": pd.DataFrame({"image_path": [b], "label": "x", "label_idx": 0}),
    }
    assert find_cross_split_duplicates(splits) == []
