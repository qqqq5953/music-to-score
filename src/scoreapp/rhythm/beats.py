"""節拍、downbeat、拍號、tempo map。

偵測器：beat_this（優先）→ librosa（備援，例如 beat_this 的模型下載不了時）。
偵測結果再整理成 BeatGrid：
- 拍號：downbeat 之間拍數的眾數（一首歌只用一個拍號）。
- 第一拍：找出和 downbeat 最吻合的相位。
- 時間 0 到第一個偵測到的拍之間補上拍（弱起小節），讓 MIDI 的第 0 tick = 音訊的第 0 秒。
- 結尾補到完整的小節。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scoreapp.model import BeatGrid

log = logging.getLogger(__name__)


@dataclass
class BeatDetection:
    beats: np.ndarray  # 秒
    downbeats: np.ndarray  # 秒（可能是空的）
    source: str  # "beat_this" / "librosa"


def detect_with_beat_this(
    y: np.ndarray, sr: int, device: str | None = None, checkpoint: str | None = "final0"
) -> BeatDetection:
    from beat_this.inference import Audio2Beats

    from scoreapp.separate import pick_device

    device = pick_device(device)
    try:
        tracker = Audio2Beats(checkpoint_path=checkpoint, device=device, dbn=False)
        beats, downbeats = tracker(y, sr)
    except Exception:
        if device == "cpu":
            raise
        tracker = Audio2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)
        beats, downbeats = tracker(y, sr)
    return BeatDetection(np.asarray(beats, float), np.asarray(downbeats, float), "beat_this")


def detect_with_librosa(y: np.ndarray, sr: int, beats_per_bar: int = 4) -> BeatDetection:
    """librosa 只會給拍點；downbeat 用低頻起音強度找最可能的相位。"""
    import librosa

    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, trim=False)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    if len(beats) < 2:
        return BeatDetection(beats, np.zeros(0), "librosa")

    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=150, n_mels=16)
    low_times = librosa.times_like(low, sr=sr, hop_length=hop)
    strength = np.interp(beats, low_times, low)
    scores = [strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    return BeatDetection(beats, beats[phase::beats_per_bar], "librosa")


def detect_beats(audio_path: Path, use_beat_this: bool = True) -> BeatDetection:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if use_beat_this:
        try:
            return detect_with_beat_this(y, sr)
        except Exception as exc:
            log.warning("beat_this 無法使用（%s），改用 librosa", exc)
    return detect_with_librosa(y, sr)


def _clean_beats(beats: np.ndarray) -> np.ndarray:
    beats = np.unique(np.asarray(beats, float))
    beats = beats[beats >= 0]
    if len(beats) < 2:
        return beats
    keep = [beats[0]]
    for b in beats[1:]:
        if b - keep[-1] > 0.15:  # 比 400 BPM 還快的是重複偵測
            keep.append(b)
    return np.array(keep)


def estimate_meter(beats: np.ndarray, downbeats: np.ndarray, default: int = 4) -> tuple[int, int]:
    """回傳（每小節拍數, 第一個 downbeat 在 beats 裡的相位）。"""
    if len(downbeats) < 2 or len(beats) < 4:
        return default, 0
    idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    idx = np.unique(idx)
    counts = np.diff(idx)
    counts = counts[(counts >= 2) & (counts <= 7)]
    bpb = int(np.bincount(counts).argmax()) if len(counts) else default
    votes = np.bincount(idx % bpb, minlength=bpb)
    return bpb, int(np.argmax(votes))


def build_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
    beats_per_bar: int | None = None,
) -> BeatGrid:
    beats = _clean_beats(beats)
    if len(beats) < 2:
        # 偵測不到節拍：假設 120 BPM
        beats = np.arange(0.0, max(duration, 2.0) + 0.5, 0.5)
        downbeats = beats[::4]
    bpb, phase = estimate_meter(beats, np.asarray(downbeats, float), beats_per_bar or 4)
    if beats_per_bar:
        bpb = beats_per_bar

    # 開頭：0 秒到第一拍之間補 n 拍
    first_iv = float(np.median(np.diff(beats[: min(len(beats), 9)])))
    n_pre = int(round(beats[0] / first_iv))
    if beats[0] <= 0.02:
        beats = beats.copy()
        beats[0] = 0.0
        n_pre = 0
    elif n_pre == 0:
        n_pre = 1
    pre = np.linspace(0.0, beats[0], n_pre + 1)[:-1] if n_pre else np.zeros(0)
    times = np.concatenate([pre, beats])
    downbeat_index = (phase + n_pre) % bpb

    # 結尾：用最後幾拍的平均拍長延伸到音訊結束，並補滿最後一小節
    last_iv = float(np.median(np.diff(beats[-min(len(beats), 9) :])))
    extra = []
    t = times[-1]
    while t < duration or (len(times) + len(extra) - 1 - downbeat_index) % bpb != 0:
        t += last_iv
        extra.append(t)
    times = np.concatenate([times, extra])
    return BeatGrid(times=times, beats_per_bar=bpb, downbeat_index=downbeat_index)


def analyze_rhythm(audio_path: Path, duration: float, use_beat_this: bool = True) -> tuple[BeatGrid, str]:
    det = detect_beats(audio_path, use_beat_this=use_beat_this)
    grid = build_grid(det.beats, det.downbeats, duration)
    log.info("節拍：%s，%d/4，約 %.1f BPM，弱起 %d 拍", det.source, grid.beats_per_bar, grid.tempo_bpm(), grid.pickup_beats)
    return grid, det.source
