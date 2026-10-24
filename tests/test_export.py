from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pretty_midi
import pytest

from scoreapp.export import musicxml as MX
from scoreapp.export.midi import score_to_midi, tempo_events
from scoreapp.export.tab import BASS_4_STANDARD, TabParams, assign_tab, candidates
from scoreapp.model import TICKS_PER_BEAT, BeatGrid, Line, NoteEvent, Score

XSD = Path(__file__).parent / "data" / "musicxml-4.0" / "musicxml.xsd"


def qnote(p, s, e, grid):
    n = NoteEvent(p, float(grid.tick_to_time(s)), float(grid.tick_to_time(e)), 90)
    n.start_tick, n.end_tick = s, e
    return n


# ---------------------------------------------------------------- TAB


def test_candidates_for_bass():
    assert candidates(28, TabParams()) == [(4, 0)]
    assert sorted(candidates(43, TabParams())) == [(1, 0), (2, 5), (3, 10), (4, 15)]
    assert candidates(27, TabParams()) == []


def _notes(pitches, dur=0.25, gap=0.0):
    return [NoteEvent(p, i * (dur + gap), i * (dur + gap) + dur) for i, p in enumerate(pitches)]


def test_assign_tab_every_note_valid():
    rng = np.random.default_rng(3)
    notes = _notes(list(rng.integers(28, 68, 200)))
    assign_tab(notes)
    for n in notes:
        assert n.string in BASS_4_STANDARD
        assert 0 <= n.fret <= 24
        assert BASS_4_STANDARD[n.string] + n.fret == n.pitch


def test_assign_tab_prefers_low_position_and_small_moves():
    # G 大調音階 G1-G2：應該都在 4 格以內，不會跳來跳去
    notes = assign_tab(_notes([31, 33, 35, 36, 38, 40, 42, 43]))
    fretted = [n.fret for n in notes if n.fret > 0]
    assert max(fretted) - min(fretted) <= 4
    assert max(n.fret for n in notes) <= 5


def test_assign_tab_stays_on_string_for_descending_line():
    notes = assign_tab(_notes([47, 45, 43]))
    assert [(n.string, n.fret) for n in notes] == [(1, 4), (1, 2), (1, 0)]


def test_assign_tab_high_passage_uses_high_position_not_jumps():
    notes = assign_tab(_notes([55, 57, 59, 60, 62, 60, 59, 57]))
    frets = [n.fret for n in notes]
    assert max(frets) - min(frets) <= 5  # 留在同一個把位附近


def test_assign_tab_shifts_out_of_range_by_octave():
    notes = assign_tab(_notes([26, 70]))
    assert [n.pitch for n in notes] == [38, 58]


def test_open_string_gives_time_to_shift():
    # 空弦長音之後才去高把位：前面的音不需要提早移到高把位
    notes = [NoteEvent(40, 0, 0.25), NoteEvent(38, 0.25, 2.25), NoteEvent(55, 2.25, 2.5), NoteEvent(57, 2.5, 2.75)]
    assign_tab(notes)
    assert (notes[0].string, notes[0].fret) == (2, 2)


# ---------------------------------------------------------------- 記譜時值拆分


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (0, 48, [(0, 48)]),
        (0, 36, [(0, 36)]),
        (12, 36, [(12, 12), (24, 12)]),  # 第 2 拍開始的二分音符要拆開（不跨小節中線）
        (24, 48, [(24, 24)]),
        (0, 18, [(0, 18)]),  # 附點四分
        (6, 18, [(6, 6), (12, 6)]),  # 切分：跨拍要拆
        (3, 12, [(3, 9)]),  # 16 分休止之後的附點 8 分
        (8, 12, [(8, 4)]),  # 三連音
        (0, 8, [(0, 8)]),
    ],
)
def test_split_span(start, end, expected):
    assert MX.split_span(start, end, 4, 48) == expected


def test_split_span_three_four():
    assert MX.split_span(0, 36, 3, 36) == [(0, 36)]
    assert MX.split_span(12, 36, 3, 36) == [(12, 24)]


def test_split_span_rejects_unrepresentable():
    with pytest.raises(ValueError):
        MX.split_span(0, 5, 4, 48)


# ---------------------------------------------------------------- MusicXML


@pytest.fixture
def rich_score():
    grid = BeatGrid(times=np.arange(0, 4 * 6 + 2) * 0.5, beats_per_bar=4, downbeat_index=1)
    spec = [
        (40, 0, 12), (28, 12, 30), (31, 30, 36), (33, 36, 60),
        (35, 60, 64), (36, 64, 68), (38, 68, 72),
        (40, 72, 78), (43, 78, 90), (45, 90, 96),
        (47, 96, 99), (45, 99, 102), (43, 102, 108),
        (40, 111, 120), (38, 120, 168),
        (55, 168, 176), (57, 176, 180), (60, 186, 192), (62, 192, 204), (64, 210, 216), (67, 216, 252),
    ]
    notes = [qnote(p, s, e, grid) for p, s, e in spec]
    assign_tab(notes)
    return Score([Line(notes)], grid, key_fifths=-2, key_mode="major", title="測試")


def test_musicxml_validates_against_schema(rich_score, tmp_path):
    lxml = pytest.importorskip("lxml.etree")
    path = MX.score_to_musicxml(rich_score, tmp_path / "bass.musicxml")
    schema = lxml.XMLSchema(lxml.parse(str(XSD)))
    doc = lxml.parse(str(path))
    assert schema.validate(doc), schema.error_log


def test_musicxml_tab_structure(rich_score, tmp_path):
    root = ET.parse(MX.score_to_musicxml(rich_score, tmp_path / "bass.musicxml")).getroot()
    attrs = root.find("part/measure/attributes")
    assert attrs.findtext("clef/sign") == "TAB"
    assert attrs.findtext("staff-details/staff-lines") == "4"
    tuning = [(t.findtext("tuning-step"), t.findtext("tuning-octave")) for t in attrs.findall("staff-details/staff-tuning")]
    assert tuning == [("E", "1"), ("A", "1"), ("D", "2"), ("G", "2")]
    assert root.findtext("part-list/score-part/score-instrument/instrument-name") == "Electric Bass"
    assert attrs.findtext("key/fifths") == "-2"
    assert root.find("part/measure").get("implicit") == "yes"

    step_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    open_pitch = BASS_4_STANDARD
    for note in root.iter("note"):
        if note.find("rest") is not None:
            continue
        assert note.findtext("type")
        assert note.findtext("stem") == "down"
        midi = (int(note.findtext("pitch/octave")) + 1) * 12 + step_to_pc[note.findtext("pitch/step")] + int(note.findtext("pitch/alter") or 0)
        string, fret = int(note.findtext("notations/technical/string")), int(note.findtext("notations/technical/fret"))
        assert open_pitch[string] + fret == midi


def test_musicxml_measure_durations_add_up(rich_score, tmp_path):
    root = ET.parse(MX.score_to_musicxml(rich_score, tmp_path / "bass.musicxml")).getroot()
    measures = root.findall("part/measure")
    totals = [sum(int(n.findtext("duration")) for n in m.findall("note")) for m in measures]
    assert totals[0] == 12  # 1 拍弱起
    assert all(t == 48 for t in totals[1:])


def test_musicxml_ties_triplets_and_beams(rich_score, tmp_path):
    root = ET.parse(MX.score_to_musicxml(rich_score, tmp_path / "bass.musicxml")).getroot()
    notes = list(root.iter("note"))
    ties = [n for n in notes if n.find("tie") is not None]
    assert ties, "跨拍 / 跨小節的音要有連結線"
    starts = sum(1 for n in notes for t in n.findall("tie") if t.get("type") == "start")
    stops = sum(1 for n in notes for t in n.findall("tie") if t.get("type") == "stop")
    assert starts == stops
    triplets = [n for n in notes if n.find("time-modification") is not None]
    assert len(triplets) == 5
    assert sum(1 for n in notes if n.find("notations/tuplet[@type='start']") is not None) == 2
    beams = [b.text for n in notes for b in n.findall("beam") if b.get("number") == "1"]
    assert beams.count("begin") == beams.count("end") > 0


def test_musicxml_quantized_line_from_random_notes_is_always_representable(tmp_path):
    """隨機產生的量化結果（16 分音符格和三連音格混合）都要能記成合法的譜。"""
    from scoreapp.rhythm.quantize import quantize_line

    rng = np.random.default_rng(7)
    grid = BeatGrid(times=np.cumsum(np.r_[0, rng.uniform(0.45, 0.55, 64)]), beats_per_bar=4, downbeat_index=3)
    t, notes = 0.1, []
    while t < grid.times[-1] - 1:
        d = rng.choice([0.08, 0.12, 0.17, 0.25, 0.33, 0.5, 0.75, 1.2])
        notes.append(NoteEvent(int(rng.integers(28, 60)), t, t + d * rng.uniform(0.6, 1.0), 90))
        t += d + rng.choice([0, 0, 0.1, 0.3])
    line = Line(notes)
    quantize_line(line, grid)
    assign_tab(line.notes)
    score = Score([line], grid)
    root = ET.parse(MX.score_to_musicxml(score, tmp_path / "r.musicxml")).getroot()
    totals = [sum(int(n.findtext("duration")) for n in m.findall("note")) for m in root.findall("part/measure")]
    assert totals[0] == 3 * TICKS_PER_BEAT and all(x == 48 for x in totals[1:])


def test_pitch_spelling():
    assert MX.pitch_parts(30, use_flats=False) == ("F", 1, 1)
    assert MX.pitch_parts(30, use_flats=True) == ("G", -1, 1)
    assert MX.pitch_parts(28, use_flats=True) == ("E", 0, 1)


# ---------------------------------------------------------------- MIDI


def _varying_grid():
    # 速度從 100 漸快到 130 BPM，2 拍弱起
    iv = 60 / np.linspace(100, 130, 42)
    return BeatGrid(times=np.r_[0, np.cumsum(iv)], beats_per_bar=4, downbeat_index=2)


def test_midi_tempo_map_and_raw_times_align_with_audio(tmp_path):
    grid = _varying_grid()
    rng = np.random.default_rng(0)
    onsets = np.sort(rng.uniform(0.2, grid.times[-1] - 1, 60))
    notes = [NoteEvent(int(rng.integers(28, 60)), float(o), float(o) + 0.2, 90) for o in onsets]
    path = score_to_midi(Score([Line(notes)], grid, title="t"), tmp_path / "bass.mid")

    pm = pretty_midi.PrettyMIDI(str(path))
    got = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    for n, g in zip(notes, got):
        assert g.pitch == n.pitch
        assert abs(g.start - n.onset) < 0.002  # 1 tick ≈ 1ms
    beats = pm.get_beats()
    assert np.allclose(beats[: len(grid.times)], grid.times[: len(beats)], atol=0.002)
    ts = [(t.numerator, t.denominator, round(t.time, 3)) for t in pm.time_signature_changes]
    assert ts[0][:2] == (2, 4) and ts[1][:2] == (4, 4)
    assert ts[1][2] == pytest.approx(grid.times[2], abs=0.002)
    assert pm.instruments[0].program == 33
    assert pm.instruments[0].name == "Bass"


def test_midi_quantized_uses_grid_positions(tmp_path):
    grid = _varying_grid()
    n = NoteEvent(40, 1.234, 1.5, 90)
    n.start_tick, n.end_tick = 30, 36
    path = score_to_midi(Score([Line([n])], grid), tmp_path / "q.mid", quantized=True)
    g = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes[0]
    assert g.start == pytest.approx(grid.tick_to_time(30), abs=0.002)
    assert g.end == pytest.approx(grid.tick_to_time(36), abs=0.002)


def test_tempo_events_merge_constant_tempo():
    grid = BeatGrid(times=np.arange(0, 9) * 0.5)
    assert tempo_events(grid) == [(0, 500000)]
