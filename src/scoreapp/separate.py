"""分軌：Demucs → bass.wav（貝斯）+ no_bass.wav（去貝斯伴奏）。

預設模型 htdemucs_ft 是「4 個模型的組合」，每個模型專門負責一種音源。我們只需要貝斯，
所以只跑負責貝斯的那個子模型，伴奏 = 原曲 − 貝斯，速度快 4 倍。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

DEFAULT_MODEL = "htdemucs_ft"

ProgressFn = Callable[[float], None]


@dataclass
class Stems:
    bass: Path
    no_bass: Path


def pick_device(prefer: str | None = None) -> str:
    """MPS（Apple GPU）可用就用 MPS，有 CUDA 就用 CUDA，否則 CPU。"""
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _bass_submodel(model, stem: str = "bass"):
    """如果是「每個子模型負責一種音源」的組合（例如 htdemucs_ft），只取負責 stem 的那個。"""
    from demucs.apply import BagOfModels

    if not isinstance(model, BagOfModels):
        return model
    idx = model.sources.index(stem)
    for sub, weights in zip(model.models, model.weights):
        others = [w for i, w in enumerate(weights) if i != idx]
        if weights[idx] > 0 and all(w == 0 for w in others):
            return sub
    return model


def load_model(name: str = DEFAULT_MODEL):
    from demucs.pretrained import get_model

    model = get_model(name)
    model.eval()
    return model


def separate_array(
    mix: np.ndarray,
    sample_rate: int,
    model,
    device: str,
    stem: str = "bass",
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """mix: (samples, channels) float32。回傳同形狀的 stem 音訊。"""
    import torch
    from demucs.apply import apply_model

    if sample_rate != model.samplerate:
        raise ValueError(f"取樣率必須是 {model.samplerate}")
    wav = torch.from_numpy(np.ascontiguousarray(mix.T, dtype=np.float32))
    if wav.shape[0] == 1:
        wav = wav.repeat(model.audio_channels, 1)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    length = wav.shape[-1]

    def on_chunk(d: dict) -> None:
        if progress and d.get("state") == "end":
            progress(min(1.0, d["segment_offset"] / length))

    sub = _bass_submodel(model, stem)
    out = apply_model(sub, wav[None], device=device, shifts=1, split=True, overlap=0.25,
                      progress=False, callback=on_chunk)[0]
    out = out * std + mean
    idx = list(sub.sources).index(stem)
    return out[idx].cpu().numpy().T


def separate_bass(
    mix_wav: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    progress: ProgressFn | None = None,
) -> Stems:
    mix, sr = sf.read(str(mix_wav), dtype="float32", always_2d=True)
    model = load_model(model_name)
    device = pick_device(device)
    log.info("Demucs %s 使用裝置：%s", model_name, device)
    try:
        bass = separate_array(mix, sr, model, device, progress=progress)
    except Exception as exc:  # MPS 偶爾會遇到不支援的運算
        if device == "cpu":
            raise
        log.warning("在 %s 上分軌失敗（%s），改用 CPU", device, exc)
        bass = separate_array(mix, sr, model, "cpu", progress=progress)

    if bass.shape[1] != mix.shape[1]:
        bass = np.repeat(bass.mean(axis=1, keepdims=True), mix.shape[1], axis=1)
    no_bass = mix - bass
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = Stems(out_dir / "bass.wav", out_dir / "no_bass.wav")
    write_wav(stems.bass, bass, sr)
    write_wav(stems.no_bass, no_bass, sr)
    return stems


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
