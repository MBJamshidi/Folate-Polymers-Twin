import os

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def results_dir():
    """The reference results shipped with the repository."""
    return os.path.join(REPO, "results")


@pytest.fixture(scope="session")
def target():
    from folate_twin import load_target
    return load_target()
