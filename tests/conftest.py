import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from scoreapp.paths import setup_cache_env  # noqa: E402

setup_cache_env()


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("synth")
