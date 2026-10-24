"""測試用的合成音訊：已知音高與節奏的貝斯音、鼓組。"""

from __future__ import annotations

import numpy as np

SR = 44100


def midi_to_hz(p: float) -> float:
    return 440.0 * 2 ** ((p - 69) / 12)


def pluck(pitch: int, dur: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """類似撥弦貝斯的聲音：多個泛音，高泛音衰減較快。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = midi_to_hz(pitch)
    out = np.zeros(n)
    for h in range(1, 9):
        if h * f0 > sr / 2 - 1000:
            break
        out += (1.0 / h**1.1) * np.exp(-t * (1.5 + 1.2 * h)) * np.sin(2 * np.pi * h * f0 * t + 0.3 * h)
    env = np.minimum(1.0, t / 0.004)
    release = int(0.02 * sr)
    if n > release:
        env[-release:] *= np.linspace(1, 0, release)
    return amp * out * env / np.max(np.abs(out) + 1e-9)


def kick(sr: int = SR, amp: float = 0.8) -> np.ndarray:
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    freq = 50 + 90 * np.exp(-t * 30)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return amp * np.sin(phase) * np.exp(-t * 18)


def noise_hit(dur: float, decay: float, sr: int = SR, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)  # 粗略高通
    return amp * x * np.exp(-t * decay) / 3


def render_notes(notes: list[tuple[int, float, float]], total: float, sr: int = SR) -> np.ndarray:
    """notes: (pitch, onset 秒, duration 秒)"""
    y = np.zeros(int(total * sr) + sr)
    for pitch, onset, dur in notes:
        s = int(onset * sr)
        seg = pluck(pitch, dur, sr)
        y[s : s + len(seg)] += seg
    return y[: int(total * sr)]


def render_drums(bpm: float, bars: int, beats_per_bar: int = 4, start: float = 0.0, sr: int = SR) -> np.ndarray:
    beat = 60.0 / bpm
    total = start + bars * beats_per_bar * beat + 1.0
    y = np.zeros(int(total * sr) + sr)
    k, sn, hh = kick(sr), noise_hit(0.2, 20, sr, 0.5, 1), noise_hit(0.05, 80, sr, 0.15, 2)
    for i in range(bars * beats_per_bar):
        t0 = start + i * beat
        s = int(t0 * sr)
        pos = i % beats_per_bar
        if pos == 0 or (beats_per_bar == 4 and pos == 2):
            y[s : s + len(k)] += k * (1.0 if pos == 0 else 0.6)
        if beats_per_bar == 4 and pos in (1, 3):
            y[s : s + len(sn)] += sn
        for half in (0, 0.5):
            s2 = int((t0 + half * beat) * sr)
            y[s2 : s2 + len(hh)] += hh
    return y[: int(total * sr)]


# 一段已知的貝斯線：（音高, 起點拍, 長度拍），4/4，兩小節重複兩次
BASS_RIFF = [
    (28, 0.0, 1.0),   # E1 四分
    (28, 1.0, 0.5),   # E1 八分
    (31, 1.5, 0.5),   # G1
    (33, 2.0, 1.0),   # A1 四分
    (35, 3.0, 0.5),   # B1
    (38, 3.5, 0.5),   # D2
    (40, 4.0, 0.75),  # E2 附點八分
    (40, 4.75, 0.25), # E2 十六分
    (43, 5.0, 1.0),   # G2
    (45, 6.0, 0.5),   # A2
    (43, 6.5, 0.5),   # G2
    (38, 7.0, 1.0),   # D2
]


def riff_events(bpm: float, start: float = 0.0, repeats: int = 2, gap: float = 0.03):
    """回傳 (pitch, onset 秒, dur 秒) 以及對應的 (pitch, start_beat, dur_beats)。"""
    beat = 60.0 / bpm
    secs, beats = [], []
    for r in range(repeats):
        for p, b, d in BASS_RIFF:
            bb = b + 8 * r
            secs.append((p, start + bb * beat, d * beat - gap))
            beats.append((p, bb, d))
    return secs, beats
