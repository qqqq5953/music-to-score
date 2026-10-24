"""端對端：合成的「貝斯 + 鼓」音檔 → 完整流程（CPU）。"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from scoreapp.model import TICKS_PER_BEAT
from scoreapp.pipelines.bass import BassOptions, run_bass_pipeline
from synth import SR, render_drums, render_notes, riff_events

BPM, START = 100, 0.9


@pytest.fixture(scope="module")
def song(synth_dir):
    secs, beats = riff_events(BPM, start=START, repeats=4)
    total = START + 32 * 60 / BPM + 1.0
    y = render_notes(secs, total)
    d = render_drums(BPM, 8, start=START)
    y[: len(d)] += 0.6 * d[: len(y)]
    path = synth_dir / "合成 riff.wav"
    sf.write(path, y, SR)
    return path, secs, beats


@pytest.fixture(scope="module")
def result(song, tmp_path_factory):
    path, _, _ = song
    progress = []
    res = run_bass_pipeline(
        str(path),
        tmp_path_factory.mktemp("out"),
        BassOptions(separate=False, use_beat_this=False),
        progress=lambda f, m: progress.append((f, m)),
    )
    res.progress = progress
    return res


def test_outputs_exist(result):
    assert result.out_dir.name == "合成 riff"
    for name in ["bass.musicxml", "bass.mid", "bass.wav", "mix.wav"]:
        assert result.files[name].exists()
    info = json.loads((result.out_dir / "analysis.json").read_text(encoding="utf-8"))
    assert info["time_signature"] == "4/4"
    assert info["tempo_bpm"] == pytest.approx(BPM, rel=0.03)
    fr = [f for f, _ in result.progress]
    assert fr == sorted(fr) and fr[-1] == 1.0


def test_quantized_notes_match_the_riff(result, song):
    _, _, beats = song
    grid = result.score.grid
    notes = result.score.lines[0].notes
    assert len(notes) == len(beats)
    first = grid.downbeat_index * TICKS_PER_BEAT  # 樂句從第一個完整小節開始
    for i, (n, (p, b, d)) in enumerate(zip(notes, beats)):
        assert n.pitch == p
        assert n.start_tick == first + round(b * TICKS_PER_BEAT)
        if i < len(beats) - 1:  # 最後一個音後面是外插的拍點，結尾允許有誤差
            assert n.end_tick == first + round((b + d) * TICKS_PER_BEAT)
        assert n.string is not None and n.fret is not None


def test_midi_aligns_with_audio(result, song):
    _, secs, _ = song
    pm = pretty_midi.PrettyMIDI(str(result.files["bass.mid"]))
    got = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    assert len(got) == len(secs)
    for g, (p, onset, _) in zip(got, secs):
        assert g.pitch == p
        assert abs(g.start - onset) < 0.02
    # MIDI 的小節線（downbeat）落在音訊的強拍上
    downbeats = pm.get_downbeats()
    bar = 4 * 60 / BPM
    assert any(abs(db - START) < 0.04 for db in downbeats)
    assert np.all(np.abs(np.diff(downbeats[1:5]) - bar) < 0.05)


def test_musicxml_is_tab_and_valid(result):
    from pathlib import Path

    lxml = pytest.importorskip("lxml.etree")
    xsd = Path(__file__).parent / "data" / "musicxml-4.0" / "musicxml.xsd"
    schema = lxml.XMLSchema(lxml.parse(str(xsd)))
    assert schema.validate(lxml.parse(str(result.files["bass.musicxml"]))), schema.error_log
    root = ET.parse(result.files["bass.musicxml"]).getroot()
    assert root.findtext("part/measure/attributes/clef/sign") == "TAB"
    assert root.findtext("work/work-title") == "合成 riff"


def test_pipeline_with_separation_step(tmp_path, monkeypatch):
    """用隨機權重的小 Demucs 模型走過分軌那條路（雲端下載不了真正的模型）。"""
    from demucs.pretrained import demucs_unittest

    from scoreapp import separate

    monkeypatch.setattr(separate, "load_model", lambda name: demucs_unittest().eval())
    secs, _ = riff_events(120, start=0.2, repeats=1)
    path = tmp_path / "short.flac"
    sf.write(path, render_notes(secs, 4.5), SR)
    res = run_bass_pipeline(str(path), tmp_path / "out", BassOptions(use_beat_this=False, device="cpu"))
    for name in ["bass.musicxml", "bass.mid", "bass.wav", "no_bass.wav", "mix.wav"]:
        assert res.files[name].exists(), name
    assert not list((tmp_path / "out").glob(".work-*"))  # 暫存資料夾要清掉
