import numpy as np
import pytest
import soundfile as sf

from scoreapp.key import detect_key
from scoreapp.model import TICKS_PER_BEAT, BeatGrid, Line, NoteEvent
from scoreapp.rhythm import beats as B
from scoreapp.rhythm.quantize import QuantizeParams, quantize_line
from synth import SR, render_drums, render_notes, riff_events


def test_beatgrid_time_beat_roundtrip_and_extrapolation():
    g = BeatGrid(times=[0.0, 0.5, 1.0, 1.6, 2.2], beats_per_bar=4)
    assert g.time_to_beat(0.75) == pytest.approx(1.5)
    assert g.time_to_beat(1.9) == pytest.approx(3.5)
    assert g.time_to_beat(2.5) == pytest.approx(4.5)  # 用最後一拍的拍長外插
    for b in [0.0, 0.3, 1.5, 2.25, 3.9, 4.5]:
        assert g.time_to_beat(g.beat_to_time(b)) == pytest.approx(b)


def test_beatgrid_rejects_bad_times():
    with pytest.raises(ValueError):
        BeatGrid(times=[0.1, 0.5])
    with pytest.raises(ValueError):
        BeatGrid(times=[0.0, 0.5, 0.5])


def test_build_grid_adds_pickup_and_completes_last_bar():
    iv = 60 / 112
    beats = 0.37 + iv * np.arange(30)
    downbeats = beats[1::4]  # 第 2 個偵測到的拍才是第一拍 → 前面有弱起
    g = B.build_grid(beats, downbeats, duration=beats[-1] + 0.2)
    assert g.times[0] == 0.0
    assert g.beats_per_bar == 4
    # 每個偵測到的 downbeat 都要落在小節開頭
    starts = set(g.bar_start_beats())
    for d in downbeats:
        idx = int(np.argmin(np.abs(g.times - d)))
        assert abs(g.times[idx] - d) < 1e-9
        assert idx in starts
    assert (g.num_beats - g.downbeat_index) % 4 == 0
    assert g.times[-1] >= beats[-1] + 0.2
    assert g.tempo_bpm() == pytest.approx(112, abs=0.5)


def test_build_grid_three_four():
    iv = 0.5
    beats = 0.02 + iv * np.arange(24)
    g = B.build_grid(beats, beats[::3], duration=12.0)
    assert g.beats_per_bar == 3
    assert g.downbeat_index == 0
    assert g.times[1] == pytest.approx(0.52)  # 0.02 秒的開頭直接當成第一拍


def test_build_grid_without_beats_falls_back_to_120():
    g = B.build_grid(np.array([]), np.array([]), duration=5.0)
    assert g.tempo_bpm() == pytest.approx(120)


@pytest.fixture(scope="module")
def groove_wav(synth_dir):
    bpm, start = 112, 0.37
    secs, _ = riff_events(bpm, start=start, repeats=6)
    total = start + 48 * 60 / bpm + 1.0
    y = render_notes(secs, total)
    d = render_drums(bpm, 12, start=start)
    y[: min(len(y), len(d))] += 0.7 * d[: len(y)]
    path = synth_dir / "groove.wav"
    sf.write(path, y, SR)
    return path, bpm, start, total


def test_librosa_fallback_finds_tempo_and_downbeat(groove_wav):
    path, bpm, start, total = groove_wav
    det = B.detect_beats(path, use_beat_this=False)
    assert det.source == "librosa"
    g = B.build_grid(det.beats, det.downbeats, total)
    assert g.tempo_bpm() == pytest.approx(bpm, rel=0.03)
    # 第一個完整小節的第一拍 = 樂句開頭（0.37 秒，誤差 40ms 內）
    first_bar = g.times[g.downbeat_index]
    bar = 4 * 60 / bpm
    assert min(abs(first_bar - start - k * bar) for k in range(3)) < 0.04


def test_beat_this_code_path_runs_with_untrained_model(groove_wav):
    """雲端下載不了 beat_this 的權重；用未訓練的模型確認呼叫流程和 torchaudio 相容。"""
    import librosa

    y, sr = librosa.load(str(groove_wav[0]), sr=22050, duration=8.0)
    det = B.detect_with_beat_this(y, sr, device="cpu", checkpoint=None)
    assert det.source == "beat_this"
    assert det.beats.ndim == 1 and det.downbeats.ndim == 1


# ---------------------------------------------------------------- 量化

GRID = BeatGrid(times=np.arange(0, 41) * 0.5, beats_per_bar=4)  # 120 BPM，10 小節


def _q(events, params=None):
    line = Line([NoteEvent(p, on, off, v) for p, on, off, v in events])
    trip = quantize_line(line, GRID, params)
    return [(n.pitch, n.start_tick, n.end_tick) for n in line.notes], trip


def test_quantize_sixteenths_with_jitter():
    s = 0.125  # 16 分音符
    rng = np.random.default_rng(1)
    ev = [(40 + i % 3, i * s + rng.uniform(-0.02, 0.02), (i + 1) * s - 0.01, 90) for i in range(8)]
    got, trip = _q(ev)
    assert [g[1] for g in got] == [i * 3 for i in range(8)]
    assert [g[2] for g in got] == [(i + 1) * 3 for i in range(8)]
    assert trip == set()


def test_quantize_triplets():
    t = 0.5 / 3
    ev = [(40, 1.0 + i * t + 0.005, 1.0 + (i + 1) * t - 0.02, 90) for i in range(3)]
    got, trip = _q(ev)
    assert [g[1] for g in got] == [24, 28, 32]
    assert [g[2] for g in got] == [28, 32, 36]
    assert trip == {2}


def test_quantize_slightly_off_sixteenth_stays_straight():
    # 16 分音符的位置偏了一點，不應該被當成三連音
    got, trip = _q([(40, 0.0, 0.14, 90), (43, 0.15, 0.5, 90)])
    assert [g[1] for g in got] == [0, 3]
    assert trip == set()


def test_quantize_collisions_and_gap_fill_and_short_notes():
    ev = [
        (40, 0.00, 0.20, 90),
        (52, 0.01, 0.20, 50),  # 同一格、比較小聲 → 刪掉
        (43, 0.25, 0.62, 90),  # 結束在第 15 tick，和下一個音只差一個 16 分音符 → 補滿
        (45, 0.70, 0.72, 90),  # 太短 → 刪掉
        (47, 0.75, 0.95, 90),
        (48, 1.25, 1.40, 90),  # 前面空了 8 分音符 → 保留休止符
    ]
    got, _ = _q(ev)
    assert got == [(40, 0, 6), (43, 6, 18), (47, 18, 24), (48, 30, 33)]


def test_quantize_end_never_before_start():
    got, _ = _q([(40, 0.24, 0.30, 90)])  # 起點和終點都對齊到 6
    assert got == [(40, 6, 9)]


def test_quantize_keeps_seconds_untouched():
    line = Line([NoteEvent(40, 0.26, 0.74, 90)])
    quantize_line(line, GRID)
    n = line.notes[0]
    assert (n.onset, n.offset) == (0.26, 0.74)
    assert (n.start_tick, n.end_tick) == (6, 18)
    assert TICKS_PER_BEAT == 12


# ---------------------------------------------------------------- 調號


def _notes(pitches):
    return [NoteEvent(p, i * 0.5, i * 0.5 + 0.45) for i, p in enumerate(pitches)]


def test_detect_key_e_minor_bass_line():
    line = [40, 40, 43, 45, 47, 50, 52, 47, 45, 43, 40, 38, 40, 47, 40, 42] * 3
    assert detect_key(_notes(line)) == (1, "minor")


def test_detect_key_f_major():
    line = [41, 45, 48, 46, 41, 43, 45, 36, 41, 48, 46, 45, 43, 41, 40, 41] * 3
    assert detect_key(_notes(line)) == (-1, "major")


def test_detect_key_too_few_notes():
    assert detect_key(_notes([40, 42])) == (0, "major")
