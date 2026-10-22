import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data_loader import load_dataset, split_train_test  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    df, report = load_dataset()
    return df, report


@pytest.fixture(scope="session")
def split(dataset):
    return split_train_test(dataset[0])


@pytest.fixture(scope="session")
def quick_artifact(tmp_path_factory):
    """Train a fast model into a temp location (does not touch the real artefact)."""
    from src.predict import load_artifact
    from src.train_model import train_and_save

    path = tmp_path_factory.mktemp("models") / "m.joblib"
    train_and_save(path, quick=True, save_reports=False)
    return path, load_artifact(path)
