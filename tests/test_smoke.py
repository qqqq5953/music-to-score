import scoreapp
from scoreapp.paths import CACHE_DIR, PROJECT_ROOT


def test_package_imports():
    assert scoreapp.__version__
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert CACHE_DIR.name == ".cache" or CACHE_DIR.is_absolute()
