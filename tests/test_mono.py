import numpy as np
import pytest
import soundfile as sf

from scoreapp.model import NoteEvent
from scoreapp.transcribe import mono
from synth import SR, kick, render_notes, riff_events


def N(p, on, off, v=90):
    return NoteEvent(p, on, off, v)


def test_remove_harmonics_drops_overtones_of_louder_lower_note():
    notes = [N(28, 0.0, 0.5, 100), N(40, 0.01, 0.4, 70), N(47, 0.0, 0.3, 60), N(52, 0.0, 0.5, 60), N(33, 0.6, 1.0)]
    kept = mono.remove_harmonics(notes)
    assert [n.pitch for n in kept] == [28, 33]


def test_remove_harmonics_keeps_loud_upper_note():
    # 上面的音比下面的大聲很多 → 不是泛音（例如和弦）
    kept = mono.remove_harmonics([N(28, 0, 0.5, 40), N(40, 0, 0.5, 110)])
    assert len(kept) == 2


def test_clamp_range():
    params = mono.MonoParams()
    out = mono.clamp_range([N(26, 0, 1), N(40, 1, 2), N(72, 2, 3)], params)
    assert [n.pitch for n in out] == [38, 40]  # D1 → D2；C5 刪除


def test_make_monophonic_truncates_overlaps_and_picks_one_of_simultaneous():
    out = mono.make_monophonic([N(40, 0.0, 1.0, 80), N(43, 0.5, 1.2, 80), N(45, 0.51, 0.9, 60)])
    assert [(n.pitch, n.onset, n.offset) for n in out] == [(40, 0.0, 0.5), (43, 0.5, 1.2)]


def test_merge_fragments_without_audio_joins_split_note():
    params = mono.MonoParams()
    out = mono.merge_fragments([N(28, 0.45, 0.55), N(28, 0.55, 1.0), N(28, 1.2, 1.5)], None, params)
    assert [(n.onset, n.offset) for n in out] == [(0.45, 1.0), (1.2, 1.5)]


def _check(line, truth, tol=0.015):
    got = [(n.pitch, n.onset) for n in line.notes]
    assert len(got) == len(truth), got
    for (p, on, _), (gp, gon) in zip(truth, got):
        assert gp == p
        assert abs(gon - on) < tol


@pytest.fixture(scope="module")
def riff_wav(synth_dir):
    secs, _ = riff_events(100, start=0.5)
    y = render_notes(secs, 11.0)
    k = kick(SR, 0.5)
    for b in range(0, 16, 2):  # 大鼓打在貝斯長音的中間（漏音最麻煩的情況）
        s = int((0.5 + b * 0.6 + 0.3) * SR)
        y[s : s + len(k)] += k
    path = synth_dir / "riff_kick.wav"
    sf.write(path, y, SR)
    return path, secs


def test_transcribe_synthetic_riff_with_kick_bleed(riff_wav):
    path, secs = riff_wav
    line = mono.transcribe_bass(path)
    _check(line, secs)
    assert all(28 <= n.pitch <= 67 for n in line.notes)


def test_transcribe_keeps_fast_repeated_notes(synth_dir):
    # 140 BPM 的八分音符重複同一個音，音和音之間只有 20ms 空隙
    beat = 60 / 140
    secs = [(33, 0.3 + i * beat / 2, beat / 2 - 0.02) for i in range(8)]
    secs += [(40, 0.3 + 4 * beat + i * beat / 4, beat / 4 - 0.015) for i in range(8)]
    path = synth_dir / "repeated.wav"
    sf.write(path, render_notes(secs, 4.0), SR)
    _check(mono.transcribe_bass(path), secs)


def test_transcribe_with_drum_bleed_and_noise(synth_dir):
    """模擬分軌不乾淨：鼓組（大鼓、小鼓噪音、hi-hat）漏進貝斯音軌，加上底噪。"""
    from synth import render_drums

    secs, _ = riff_events(112, start=0.37, repeats=7)
    total = secs[-1][1] + 1.5
    y = render_notes(secs, total)
    d = render_drums(112, 14, start=0.37)[: len(y)]
    y[: len(d)] += 0.25 * d
    y += 0.01 * np.random.default_rng(0).standard_normal(len(y))
    path = synth_dir / "drum_bleed.wav"
    sf.write(path, y, SR)

    notes = mono.transcribe_bass(path).notes
    onsets = np.array([n.onset for n in notes])
    hits = 0
    for p, on, _ in secs:
        j = int(np.argmin(np.abs(onsets - on)))
        hits += abs(onsets[j] - on) < 0.03 and notes[j].pitch == p
    assert hits / len(secs) >= 0.95
    assert len(notes) <= len(secs) * 1.05
