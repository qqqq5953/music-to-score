"""專案路徑與快取位置。

模型（Demucs、beat_this）預設會下載到使用者家目錄的各種快取資料夾。這裡把它們全部導向
專案內的 `.cache/`，讓整個專案資料夾刪掉就乾淨了。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("SCOREAPP_CACHE", PROJECT_ROOT / ".cache"))
OUTPUT_DIR = Path(os.environ.get("SCOREAPP_OUTPUT", PROJECT_ROOT / "output"))


def setup_cache_env() -> None:
    """在載入 torch / huggingface 之前呼叫，把模型快取放進專案資料夾。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
    # MPS 不支援的運算自動退回 CPU，不要直接當掉。
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
