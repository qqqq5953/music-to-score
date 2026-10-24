"""單聲部轉譜（貝斯）：Basic Pitch + 後處理。

Basic Pitch 是多音模型，直接用在貝斯音軌上會有幾個問題，這裡逐一處理：
1. 泛音被當成音符（基音 +12、+19、+24 半音）→ 有更低、夠大聲、同時發聲的基音就刪掉。
2. 八度錯誤 → 用 pYIN 的 f0 和頻譜（低八度候選的奇數泛音有沒有能量）一起判斷。
3. 低音的起音時間會偏 30～50ms，還常被切成「短碎片 + 主音」→ 用起音包絡修正並合併碎片。
4. 大鼓漏進貝斯音軌的假音符：短、低、沒有穩定音高 → 刪掉。
5. 最後整理成單聲部，限制在 4 弦貝斯的音域（E1～G4）。

注意：Basic Pitch 的 melodia_trick 預設關閉（我們不使用 Melodia 相關演算法）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from scoreapp.model import Line, NoteEvent

log = logging.getLogger(__name__)

HARMONIC_INTERVALS = (12, 19, 24, 28, 31)


@dataclass
class MonoParams:
    min_pitch: int = 28  # E1：4 弦貝斯最低音
    max_pitch: int = 67  # G4：G 弦第 24 格
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_sec: float = 0.06
    melodia_trick: bool = False
    octave_correction: bool = True
    refine_onsets: bool = True
    onset_search_before: float = 0.12  # 往前找起音的範圍（秒）
    onset_search_after: float = 0.08
    merge_gap_sec: float = 0.03  # 同音高、間隔這麼小且沒有新起音 → 合併
    recover_missed: bool = True  # 用起音包絡 + pYIN 補回 Basic Pitch 漏掉的音
    recover_min_strength: float = 0.4
    kick_filter: bool = True
    kick_max_sec: float = 0.13
    kick_max_pitch: int = 40  # E2 以下才可能是大鼓（大鼓基頻約 40～80Hz）


@dataclass
class F0Track:
    """pYIN 的結果（MIDI 音高，無聲的格子是 NaN）。"""

    times: np.ndarray
    midi: np.ndarray
    voiced_prob: np.ndarray

    def span(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        i0, i1 = np.searchsorted(self.times, [t0, t1])
        return self.midi[i0:i1], self.voiced_prob[i0:i1]


@dataclass
class AudioFeatures:
    sr: int
    f0: F0Track
    onset_times: np.ndarray
    onset_env: np.ndarray  # 正規化到 ~0..1
    spec: np.ndarray  # |STFT|，(freq, frames)，頻率解析度高、時間解析度低（判斷八度用）
    spec_freqs: np.ndarray
    spec_times: np.ndarray
    hspec: np.ndarray  # 時間解析度較高的 |STFT|（判斷是否重新撥弦用）
    hspec_freqs: np.ndarray
    hspec_times: np.ndarray


# ---------------------------------------------------------------- Basic Pitch


@lru_cache(maxsize=1)
def _basic_pitch_model():
    import basic_pitch
    from basic_pitch.inference import Model

    return Model(basic_pitch.build_icassp_2022_model_path(basic_pitch.FilenameSuffix.onnx))


def run_basic_pitch(wav_path: Path, params: MonoParams) -> list[NoteEvent]:
    from basic_pitch.inference import predict

    _, _, events = predict(
        str(wav_path),
        _basic_pitch_model(),
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_note_length=params.min_note_sec * 1000,
        melodia_trick=params.melodia_trick,
        multiple_pitch_bends=False,
    )
    notes = [
        NoteEvent(int(p), float(s), float(e), int(np.clip(round(30 + 97 * a), 1, 127)))
        for s, e, p, a, _ in events
    ]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


# ---------------------------------------------------------------- 音訊特徵


def analyze_audio(y: np.ndarray, sr: int) -> AudioFeatures:
    """y：單聲道。計算 pYIN f0、起音包絡、頻譜。"""
    import librosa

    # pYIN：貝斯在 420Hz 以下，降到 11025Hz 算比較快
    y11 = librosa.resample(y, orig_sr=sr, target_sr=11025) if sr != 11025 else y
    hop = 128
    f0, _, vprob = librosa.pyin(
        y11, fmin=35.0, fmax=420.0, sr=11025, frame_length=1024, hop_length=hop,
        resolution=0.25,  # 預設 0.1 慢 6 倍以上，對四捨五入到半音的結果沒有差別
        fill_na=np.nan,
    )
    f0_times = librosa.times_like(f0, sr=11025, hop_length=hop)
    midi = librosa.hz_to_midi(f0)

    # 起音包絡：短視窗（512 點 ≈ 23ms）的 log-mel 頻譜通量，時間解析度約 3ms。
    # 峰值會比真正的起音早約 0.3 個視窗長，補回去。
    y22 = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    n_fft, ohop = 512, 64
    power = np.abs(librosa.stft(y22, n_fft=n_fft, hop_length=ohop)) ** 2
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=22050, n_mels=48, fmin=30, fmax=4000))
    env = np.concatenate([[0.0], np.maximum(0.0, np.diff(logmel, axis=1)).mean(axis=0)])
    scale = np.percentile(env, 99) if np.any(env > 0) else 1.0
    env = env / (scale + 1e-9)
    env_times = librosa.frames_to_time(np.arange(len(env)), sr=22050, hop_length=ohop) + 0.3 * n_fft / 22050

    # 頻譜：11025Hz、n_fft 4096（解析度約 2.7Hz，分得出 E1 和 F1）
    # 只保留 1500Hz 以下（貝斯基音與前幾個泛音），省記憶體。
    spec_freqs = librosa.fft_frequencies(sr=11025, n_fft=4096)
    keep = spec_freqs < 1500
    spec = np.abs(librosa.stft(y11, n_fft=4096, hop_length=256))[keep].astype(np.float32)
    spec_times = librosa.times_like(spec, sr=11025, hop_length=256)

    # 時間解析度較高的頻譜：22050Hz、n_fft 1024（46ms）、hop 64（2.9ms）
    hspec_freqs = librosa.fft_frequencies(sr=22050, n_fft=1024)
    hkeep = hspec_freqs < 1500
    hspec = np.abs(librosa.stft(y22, n_fft=1024, hop_length=64))[hkeep].astype(np.float32)
    hspec_times = librosa.times_like(hspec, sr=22050, hop_length=64)
    return AudioFeatures(sr, F0Track(f0_times, midi, vprob), env_times, env,
                         spec, spec_freqs[keep], spec_times, hspec, hspec_freqs[hkeep], hspec_times)


def _band_energy(feat: AudioFeatures, t0: float, t1: float, hz: float) -> float:
    i0, i1 = np.searchsorted(feat.spec_times, [t0, max(t1, t0 + 0.03)])
    if i1 <= i0 or hz >= feat.spec_freqs[-1]:
        return 0.0
    lo, hi = hz * 2 ** (-0.4 / 12), hz * 2 ** (0.4 / 12)
    j0, j1 = np.searchsorted(feat.spec_freqs, [lo, hi])
    j1 = max(j1, j0 + 1)
    return float(feat.spec[j0:j1, i0:i1].max(axis=0).mean())


def _hz(p: float) -> float:
    return 440.0 * 2 ** ((p - 69) / 12)


def odd_harmonic_ratio(feat: AudioFeatures, pitch: int, t0: float, t1: float) -> float:
    """「pitch 是真的基音」的證據：奇數泛音（1、3 倍）能量 / 偶數泛音（2、4 倍）能量。

    如果真正的音高高一個八度，pitch 的 1、3 倍頻率上就沒有能量，比值會很小。
    """
    f = _hz(pitch)
    odd = _band_energy(feat, t0, t1, f) + 0.7 * _band_energy(feat, t0, t1, 3 * f)
    even = _band_energy(feat, t0, t1, 2 * f) + 0.7 * _band_energy(feat, t0, t1, 4 * f)
    return odd / (even + 1e-9)


# ---------------------------------------------------------------- 後處理步驟


def _overlap(a: NoteEvent, b: NoteEvent) -> float:
    return min(a.offset, b.offset) - max(a.onset, b.onset)


def remove_harmonics(notes: list[NoteEvent]) -> list[NoteEvent]:
    """同時有更低、夠大聲的音，且音程是泛音關係 → 這個音是泛音，刪掉。"""
    keep = []
    for n in notes:
        is_harmonic = False
        for m in notes:
            if m is n or n.pitch - m.pitch not in HARMONIC_INTERVALS:
                continue
            ov = _overlap(n, m)
            if ov > 0.5 * min(n.duration, m.duration) and m.velocity >= 0.7 * n.velocity:
                is_harmonic = True
                break
        if not is_harmonic:
            keep.append(n)
    return keep


def correct_octaves(notes: list[NoteEvent], feat: AudioFeatures) -> list[NoteEvent]:
    for n in notes:
        midi, vprob = feat.f0.span(n.onset, n.offset)
        voiced = midi[~np.isnan(midi)]
        if len(midi) == 0 or len(voiced) < 0.5 * len(midi):
            continue
        diff = float(np.median(voiced)) - n.pitch
        octaves = int(round(diff / 12))
        if octaves == 0 or abs(diff - 12 * octaves) > 0.7 or abs(octaves) > 2:
            continue
        if octaves < 0:
            # pYIN 說比較低：每往下一個八度，候選的奇數泛音都要有能量才相信
            while octaves < 0 and odd_harmonic_ratio(feat, n.pitch - 12, n.onset, n.offset) > 0.12:
                n.pitch -= 12
                octaves += 1
        elif odd_harmonic_ratio(feat, n.pitch, n.onset, n.offset) < 0.3:
            # pYIN 說比較高：原本的音高沒有奇數泛音才相信（只移一個八度）
            n.pitch += 12
    return notes


def clamp_range(notes: list[NoteEvent], params: MonoParams) -> list[NoteEvent]:
    """低於 E1 的移高八度（4 弦彈得到）；高於 G4 的多半是泛音或其他樂器漏音，刪掉。"""
    out = []
    for n in notes:
        while n.pitch < params.min_pitch:
            n.pitch += 12
        if n.pitch <= params.max_pitch:
            out.append(n)
    return out


def harmonic_reattack(feat: AudioFeatures, pitch: int, t: float, not_before: float = -np.inf) -> float:
    """在 t 附近，這個音高各泛音（2～8 倍）能量變化倍數的中位數：之後 / 之前。

    真的重新撥弦，泛音都會重新變大（> 1）；持續中的音只會繼續衰減（< 1）。
    不看基音，因為大鼓的能量就在那附近；取中位數，大鼓的擊槌聲只讓少數頻帶變大，影響不了。
    「之前」的視窗不會早於 not_before（前一個音的起音之後），前一個音太短就回傳 1。
    """
    f0 = _hz(pitch)
    b0, b1, a0, a1 = np.searchsorted(feat.hspec_times, [max(t - 0.035, not_before), t - 0.012, t + 0.012, t + 0.035])
    if b1 <= b0 or a1 <= a0:
        return 1.0
    bin_hz = feat.hspec_freqs[1] - feat.hspec_freqs[0]
    ratios = []
    for h in range(2, 9):
        j = int(round(h * f0 / bin_hz))
        if j + 1 >= len(feat.hspec_freqs):
            break
        band = feat.hspec[max(j - 1, 0) : j + 2]
        before, after = band[:, b0:b1].max(axis=0).mean(), band[:, a0:a1].max(axis=0).mean()
        ratios.append(after / (before + 1e-9))
    return float(np.median(ratios)) if ratios else 1.0


def _onset_peak(feat: AudioFeatures, t0: float, t1: float, center: float | None = None) -> tuple[float, float]:
    """回傳 [t0, t1] 之間起音包絡的峰值（時間, 強度）。

    有 center 的話，離 center 越遠的峰值打越多折扣（遠處的鼓聲不要蓋過近處比較弱的起音）。
    """
    i0, i1 = np.searchsorted(feat.onset_times, [t0, t1])
    if i1 <= i0:
        return (t0 + t1) / 2, 0.0
    env = feat.onset_env[i0:i1]
    score = env if center is None else env * np.exp(-np.abs(feat.onset_times[i0:i1] - center) / 0.08)
    i = i0 + int(np.argmax(score))
    return float(feat.onset_times[i]), float(feat.onset_env[i])


def _noise_floor(feat: AudioFeatures, t0: float, t1: float) -> float:
    """附近起音包絡的中位數。鼓的殘留（尤其小鼓的噪音）會讓整段包絡變高。"""
    i0, i1 = np.searchsorted(feat.onset_times, [t0 - 0.1, t1 + 0.1])
    return float(np.median(feat.onset_env[i0:i1])) if i1 > i0 else 0.0


def _is_clear_peak(feat: AudioFeatures, t0: float, t1: float, strength: float, minimum: float) -> bool:
    return strength > max(minimum, 3.0 * _noise_floor(feat, t0, t1))


def refine_onsets(notes: list[NoteEvent], feat: AudioFeatures, params: MonoParams) -> list[NoteEvent]:
    """把 onset 移到附近的起音峰值。

    搜尋範圍不會越過前一個音（已修正）的起音，也不會碰到下一個不同音高的音，避免兩個音搶同一個起音。
    """
    notes = sorted(notes, key=lambda n: n.onset)
    last = -np.inf
    for i, n in enumerate(notes):
        nxt = next((m.onset for m in notes[i + 1 :] if m.pitch != n.pitch and m.onset > n.onset + 0.04), np.inf)
        t0 = max(n.onset - params.onset_search_before, last + 0.04)
        t1 = min(n.onset + params.onset_search_after, nxt - 0.03)
        if t1 > t0:
            t, strength = _onset_peak(feat, t0, t1, center=n.onset)
            if _is_clear_peak(feat, t0, t1, strength, 0.15):
                # 起音在這個音快結束的地方 → 這段是提早出現的碎片，之後會和主音合併
                n.onset = t
                n.offset = max(n.offset, t + 0.03)
        last = max(last, n.onset)
    return notes


def merge_fragments(notes: list[NoteEvent], feat: AudioFeatures | None, params: MonoParams) -> list[NoteEvent]:
    """同音高、幾乎相連、第二段沒有明顯的新起音 → 是同一個音被切開，合併。"""
    notes = sorted(notes, key=lambda n: n.onset)
    out: list[NoteEvent] = []
    for n in notes:
        prev = next((m for m in reversed(out[-4:]) if m.pitch == n.pitch), None)
        if prev is not None and n.onset - prev.onset < 0.04:
            prev.offset = max(prev.offset, n.offset)
            prev.velocity = max(prev.velocity, n.velocity)
            continue
        if prev is not None and n.onset - prev.offset < params.merge_gap_sec:
            new_attack = feat is not None and harmonic_reattack(feat, n.pitch, n.onset, prev.onset + 0.04) > 1.3
            if not new_attack or prev.duration < 0.04:
                prev.offset = max(prev.offset, n.offset)
                prev.velocity = max(prev.velocity, n.velocity)
                continue
            if feat is not None and prev.duration < 0.12 and prev is out[-1]:
                # 短碎片自己沒有起音、後面的音才有 → 是主音提早出現的殘影，丟掉碎片
                _, own = _onset_peak(feat, prev.onset - 0.01, prev.onset + 0.01)
                _, main = _onset_peak(feat, n.onset - 0.01, n.onset + 0.01)
                if own < 0.5 * main:
                    out[-1] = n
                    continue
        out.append(n)
    return out


def validate_overlapping_onsets(notes: list[NoteEvent], feat: AudioFeatures) -> list[NoteEvent]:
    """前一個（不同音高的）音還在響時就冒出來的音，要有證據才算數。

    Basic Pitch 常在下一個音真正開始前 100～200ms 就冒出一段小聲的「假開頭」，
    單聲部整理時會把前一個音截斷。
    - pYIN 聽到的還是前一個音 → 刪掉。
    - pYIN 聽到新的音高（hammer-on、滑音沒有起音，但音高會變）→ 保留。
    - pYIN 不確定 → 有明顯的起音才保留。
    """
    notes = sorted(notes, key=lambda n: n.onset)
    out: list[NoteEvent] = []
    for n in notes:
        prev = out[-1] if out else None
        if prev is not None and prev.pitch != n.pitch and prev.offset > n.onset + 0.02:
            heard = _stable_pitch(feat, n.onset + 0.015, min(n.offset, n.onset + 0.08))
            if heard == prev.pitch:
                continue  # pYIN 聽到的還是前一個音
            _, strength = _onset_peak(feat, n.onset - 0.02, n.onset + 0.02)
            if heard != n.pitch and not _is_clear_peak(feat, n.onset, n.onset, strength, 0.15):
                continue
        out.append(n)
    return out


def _stable_pitch(feat: AudioFeatures, t0: float, t1: float) -> int | None:
    """[t0, t1] 之間 pYIN 有穩定音高就回傳（四捨五入的 MIDI 音高），否則 None。"""
    midi, _ = feat.f0.span(t0, t1)
    voiced = midi[~np.isnan(midi)]
    if len(midi) < 3 or len(voiced) < 0.6 * len(midi) or np.std(voiced) > 0.5:
        return None
    return int(round(float(np.median(voiced))))


def _onset_peaks(feat: AudioFeatures, min_strength: float, min_gap: float = 0.04) -> list[tuple[float, float]]:
    env, times = feat.onset_env, feat.onset_times
    idx = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > min_strength)) + 1
    peaks: list[tuple[float, float]] = []
    for i in idx:
        if peaks and times[i] - peaks[-1][0] < min_gap:
            if env[i] > peaks[-1][1]:
                peaks[-1] = (float(times[i]), float(env[i]))
            continue
        peaks.append((float(times[i]), float(env[i])))
    return peaks


def recover_missed_notes(notes: list[NoteEvent], feat: AudioFeatures, params: MonoParams) -> list[NoteEvent]:
    """Basic Pitch 偶爾會整個漏掉一個音（尤其是快速的音群）。

    在起音包絡的明顯峰值附近如果沒有音符：
    - 沒有音在響 → pYIN 之後有穩定音高就補一個音。
    - 落在某個音的中間 → 音高變了，或同音高但泛音重新變大（重新撥弦），就從這裡切開。
    大鼓的起音沒有穩定音高、也不會讓泛音變大，所以不會被補進來。
    """
    notes = sorted(notes, key=lambda n: n.onset)
    onsets = np.array([n.onset for n in notes]) if notes else np.zeros(0)
    added: list[NoteEvent] = []
    for t, strength in _onset_peaks(feat, params.recover_min_strength):
        if len(onsets) and np.min(np.abs(onsets - t)) < 0.05:
            continue
        if not _is_clear_peak(feat, t, t, strength, params.recover_min_strength):
            continue
        nxt = onsets[onsets > t + 0.03]
        nxt_t = float(nxt[0]) if len(nxt) else t + 0.5
        pitch = _stable_pitch(feat, t + 0.01, min(nxt_t, t + 0.12))
        if pitch is None or not (params.min_pitch <= pitch <= params.max_pitch):
            continue
        cur = next((n for n in notes if n.onset < t < n.offset), None)
        if cur is None:
            end = min(nxt_t, t + 0.5)
            if end - t >= params.min_note_sec:
                added.append(NoteEvent(pitch, t, end, int(np.clip(40 + 60 * min(strength, 1.0), 1, 127))))
            continue
        if cur.offset - t < params.min_note_sec or t - cur.onset < params.min_note_sec:
            continue
        if pitch != cur.pitch or harmonic_reattack(feat, pitch, t, cur.onset + 0.04) > 1.5:
            added.append(NoteEvent(pitch, t, cur.offset, cur.velocity))
            cur.offset = t
    if added:
        log.debug("補回 %d 個音", len(added))
    return sorted(notes + added, key=lambda n: n.onset)


def harmonicity(feat: AudioFeatures, pitch: int, t0: float, t1: float) -> float:
    """2、3 倍泛音的能量 / 基音的能量。撥弦的貝斯泛音很多，大鼓幾乎沒有。"""
    f0 = _hz(pitch)
    i0, i1 = np.searchsorted(feat.hspec_times, [t0, max(t1, t0 + 0.02)])
    bin_hz = feat.hspec_freqs[1] - feat.hspec_freqs[0]

    def band(h: int) -> float:
        j = int(round(h * f0 / bin_hz))
        return float(feat.hspec[max(j - 1, 0) : j + 2, i0:i1].max(axis=0).mean()) if i1 > i0 else 0.0

    return (band(2) + band(3)) / (band(1) + 1e-9)


def remove_kick_bleed(notes: list[NoteEvent], feat: AudioFeatures, params: MonoParams) -> list[NoteEvent]:
    """大鼓：很短、很低、沒有穩定音高（pYIN 無聲或音高往下滑），而且幾乎沒有泛音。"""
    out = []
    for n in notes:
        if n.duration < params.kick_max_sec and n.pitch <= params.kick_max_pitch:
            midi, _ = feat.f0.span(n.onset, n.offset)
            voiced = midi[~np.isnan(midi)]
            unstable = len(voiced) < 0.4 * max(len(midi), 1) or (len(voiced) > 2 and np.std(voiced) > 1.0)
            if unstable and harmonicity(feat, n.pitch, n.onset, n.offset) < 0.5:
                continue
        out.append(n)
    return out


def make_monophonic(notes: list[NoteEvent], feat: AudioFeatures | None = None) -> list[NoteEvent]:
    """同時起音的只留一個（優先選 pYIN 支持的，否則選最大聲的），重疊的部分把前一個音截斷。"""
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    out: list[NoteEvent] = []
    for n in notes:
        if out and n.onset - out[-1].onset < 0.04:
            prev = out[-1]
            if _prefer(n, prev, feat):
                out[-1] = n
            continue
        if out and out[-1].offset > n.onset:
            out[-1].offset = n.onset
        out.append(n)
    return out


def _prefer(a: NoteEvent, b: NoteEvent, feat: AudioFeatures | None) -> bool:
    """a 比 b 更可能是真正的音？"""
    if feat is not None:
        midi, _ = feat.f0.span(max(a.onset, b.onset), min(a.offset, b.offset) + 0.02)
        voiced = midi[~np.isnan(midi)]
        if len(voiced):
            ref = float(np.median(voiced))
            da, db = abs(a.pitch - ref), abs(b.pitch - ref)
            if abs(da - db) > 0.5:
                return da < db
    if a.velocity != b.velocity:
        return a.velocity > b.velocity
    return a.duration > b.duration


def postprocess(notes: list[NoteEvent], feat: AudioFeatures | None, params: MonoParams) -> list[NoteEvent]:
    notes = remove_harmonics(notes)
    if feat is not None and params.octave_correction:
        notes = correct_octaves(notes, feat)
    notes = clamp_range(notes, params)
    if feat is not None and params.refine_onsets:
        notes = refine_onsets(notes, feat, params)
    notes = merge_fragments(notes, feat, params)
    if feat is not None:
        notes = validate_overlapping_onsets(notes, feat)
        notes = merge_fragments(notes, feat, params)
    if feat is not None and params.recover_missed:
        notes = recover_missed_notes(notes, feat, params)
    if feat is not None and params.kick_filter:
        notes = remove_kick_bleed(notes, feat, params)
    notes = make_monophonic(notes, feat)
    return [n for n in notes if n.duration >= params.min_note_sec]


def transcribe_bass(wav_path: Path, params: MonoParams | None = None) -> Line:
    import librosa

    params = params or MonoParams()
    raw = run_basic_pitch(wav_path, params)
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    feat = analyze_audio(y, sr)
    notes = postprocess(raw, feat, params)
    log.info("Basic Pitch 原始音符 %d 個，整理後 %d 個", len(raw), len(notes))
    return Line(notes=notes, name="Bass", kind="melody", register="low")
