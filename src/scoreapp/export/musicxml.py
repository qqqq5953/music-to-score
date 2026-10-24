"""MusicXML 輸出（用 ElementTree 自己寫，music21 對 TAB 的支援不完整）。

階段一：一條線 → 4 弦貝斯 TAB 譜表（帶節奏符桿）。
- `<clef><sign>TAB</sign>`、`<staff-details>` 設 4 線 E-A-D-G 調音 → MuseScore 匯入成 TAB 譜表。
- 每個音帶 `<technical><string>/<fret>`，也帶 `<pitch>`（播放用）。
- 每個音都寫出 `<type>`、`<stem>`、`<beam>`、三連音 → 數字下方有符桿和符尾。
- 音高寫實際發聲的音高（E1 = 最低的空弦），不加移調，MuseScore 播放音高就是對的。

記譜規則（為了好讀）：
- 音符不跨拍，跨拍的用連結線（tie）拆開；但從拍頭開始、占滿整拍的音可以合成二分、附點二分、全音符。
- 每一拍內的 8 分、16 分音符用符槓連起來。
- 時值單位：每個四分音符 12 個 division（16 分音符 = 3，8 分三連音 = 4）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from scoreapp import __version__
from scoreapp.export.tab import BASS_4_STANDARD
from scoreapp.model import TICKS_PER_BEAT, NoteEvent, Score

DIV = TICKS_PER_BEAT

# 時值（division）→（type, 附點數, 是否三連音）
DURATIONS: dict[int, tuple[str, int, bool]] = {
    48: ("whole", 0, False),
    36: ("half", 1, False),
    24: ("half", 0, False),
    18: ("quarter", 1, False),
    12: ("quarter", 0, False),
    9: ("eighth", 1, False),
    8: ("quarter", 0, True),
    6: ("eighth", 0, False),
    4: ("eighth", 0, True),
    3: ("16th", 0, False),
}
BEAMABLE = {"eighth", "16th"}

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Piece:
    """樂譜上的一個符號（音符或休止符），長度一定在 DURATIONS 裡。"""

    start: int  # 小節內的位置（division）
    length: int
    note: NoteEvent | None  # None = 休止符
    tie_start: bool = False
    tie_stop: bool = False
    beam: list[tuple[int, str]] | None = None
    tuplet: str | None = None  # "start" / "stop"

    @property
    def kind(self) -> tuple[str, int, bool]:
        return DURATIONS[self.length]


def pitch_parts(midi: int, use_flats: bool) -> tuple[str, int, int]:
    """MIDI 音高 →（音名, 升降, 八度）。"""
    name = (FLAT_NAMES if use_flats else SHARP_NAMES)[midi % 12]
    octave = midi // 12 - 1
    alter = 1 if "#" in name else -1 if "b" in name else 0
    return name[0], alter, octave


# ---------------------------------------------------------------- 拆成可記譜的時值


def _merge_beats(beat: int, full: int, beats_per_bar: int) -> int:
    """從第 beat 拍開始、還剩 full 個整拍，可以合成幾拍的一個音符。"""
    if beats_per_bar == 4:
        if beat == 0 and full >= 4:
            return 4
        if beat == 0 and full >= 3:
            return 3
        if beat in (0, 2) and full >= 2:  # 二分音符不跨過小節中間
            return 2
    elif beats_per_bar == 3:
        if beat == 0 and full >= 3:
            return 3
        if beat in (0, 1) and full >= 2:
            return 2
    elif beats_per_bar == 2:
        if beat == 0 and full >= 2:
            return 2
    return 1


def split_span(start: int, end: int, beats_per_bar: int, measure_len: int) -> list[tuple[int, int]]:
    """把小節內的 [start, end) 拆成 (位置, 長度) 的列表，每段都是可記譜的時值。"""
    pieces = []
    pos = start
    full_bar = measure_len == beats_per_bar * DIV
    while pos < end:
        beat, off = divmod(pos, DIV)
        if off == 0 and end - pos >= DIV:
            n = _merge_beats(beat, (end - pos) // DIV, beats_per_bar) if full_bar else 1
            length = n * DIV
            # 從第 1、3 拍開始剛好一拍半 → 附點四分音符
            if n == 1 and end - pos == 18 and full_bar and beat % 2 == 0 and beat + 1 < beats_per_bar:
                length = 18
        else:
            length = min(end, (beat + 1) * DIV) - pos
        if length not in DURATIONS:
            raise ValueError(f"無法記譜的時值：位置 {pos}，長度 {length}")
        pieces.append((pos, length))
        pos += length
    return pieces


def layout_measure(notes: list[NoteEvent], m_start: int, m_len: int, beats_per_bar: int) -> list[Piece]:
    """notes：和這個小節有重疊的音（已量化、單聲部、依起點排序）。"""
    m_end = m_start + m_len
    spans: list[tuple[int, int, NoteEvent | None, bool, bool]] = []
    pos = m_start
    for n in notes:
        s, e = max(n.start_tick, m_start), min(n.end_tick, m_end)
        if e <= s:
            continue
        if s > pos:
            spans.append((pos, s, None, False, False))
        spans.append((s, e, n, n.start_tick < m_start, n.end_tick > m_end))
        pos = e
    if pos < m_end:
        spans.append((pos, m_end, None, False, False))

    pieces: list[Piece] = []
    for s, e, n, cont_before, cont_after in spans:
        parts = split_span(s - m_start, e - m_start, beats_per_bar, m_len)
        for i, (p, length) in enumerate(parts):
            piece = Piece(p, length, n)
            if n is not None:
                piece.tie_stop = i > 0 or cont_before
                piece.tie_start = i < len(parts) - 1 or cont_after
            pieces.append(piece)
    _add_beams_and_tuplets(pieces)
    return pieces


def _add_beams_and_tuplets(pieces: list[Piece]) -> None:
    by_beat: dict[int, list[Piece]] = {}
    for p in pieces:
        if p.start % DIV + p.length <= DIV:  # 只看在一拍之內的符號
            by_beat.setdefault(p.start // DIV, []).append(p)
    for group in by_beat.values():
        trip = [p for p in group if p.kind[2]]
        if trip:
            trip[0].tuplet = "start"
            trip[-1].tuplet = "stop" if len(trip) > 1 else "start"
            if len(trip) == 1:
                trip[0].tuplet = None  # 只有一個三連音符號 → 不畫括號（time-modification 仍保留）
        run: list[Piece] = []
        for p in group + [None]:  # type: ignore[list-item]
            if p is not None and p.note is not None and p.kind[0] in BEAMABLE:
                run.append(p)
                continue
            if len(run) >= 2:
                _beam_run(run)
            run = []


def _beam_run(run: list[Piece]) -> None:
    for i, p in enumerate(run):
        level1 = "begin" if i == 0 else "end" if i == len(run) - 1 else "continue"
        beams = [(1, level1)]
        if p.kind[0] == "16th":
            prev16 = i > 0 and run[i - 1].kind[0] == "16th"
            next16 = i < len(run) - 1 and run[i + 1].kind[0] == "16th"
            if prev16 and next16:
                beams.append((2, "continue"))
            elif next16:
                beams.append((2, "begin"))
            elif prev16:
                beams.append((2, "end"))
            else:
                beams.append((2, "forward hook" if i == 0 else "backward hook"))
        p.beam = beams


# ---------------------------------------------------------------- XML


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def _note_xml(measure: ET.Element, piece: Piece, use_flats: bool, tab: bool) -> None:
    typ, dots, triplet = piece.kind
    el = _sub(measure, "note")
    if piece.note is None:
        _sub(el, "rest")
    else:
        step, alter, octave = pitch_parts(piece.note.pitch, use_flats)
        pitch = _sub(el, "pitch")
        _sub(pitch, "step", step)
        if alter:
            _sub(pitch, "alter", alter)
        _sub(pitch, "octave", octave)
    _sub(el, "duration", piece.length)
    if piece.tie_stop:
        _sub(el, "tie", type="stop")
    if piece.tie_start:
        _sub(el, "tie", type="start")
    _sub(el, "voice", 1)
    _sub(el, "type", typ)
    for _ in range(dots):
        _sub(el, "dot")
    if triplet:
        tm = _sub(el, "time-modification")
        _sub(tm, "actual-notes", 3)
        _sub(tm, "normal-notes", 2)
    if piece.note is not None:
        _sub(el, "stem", "down" if tab else ("up" if piece.note.pitch < 50 else "down"))
    for number, value in piece.beam or []:
        _sub(el, "beam", value, number=str(number))

    notations = ET.Element("notations")
    if piece.tie_stop:
        _sub(notations, "tied", type="stop")
    if piece.tie_start:
        _sub(notations, "tied", type="start")
    if piece.tuplet:
        _sub(notations, "tuplet", type=piece.tuplet, bracket="yes")
    if tab and piece.note is not None and piece.note.string is not None:
        tech = _sub(notations, "technical")
        _sub(tech, "string", piece.note.string)
        _sub(tech, "fret", piece.note.fret)
    if len(notations):
        el.append(notations)


def _attributes(measure: ET.Element, score: Score, tab: bool, beats: int) -> None:
    attr = _sub(measure, "attributes")
    _sub(attr, "divisions", DIV)
    key = _sub(attr, "key")
    _sub(key, "fifths", score.key_fifths)
    _sub(key, "mode", score.key_mode)
    time = _sub(attr, "time")
    _sub(time, "beats", beats)
    _sub(time, "beat-type", 4)
    clef = _sub(attr, "clef")
    if tab:
        _sub(clef, "sign", "TAB")
        _sub(clef, "line", 5)
        details = _sub(attr, "staff-details")
        tuning = BASS_4_STANDARD
        _sub(details, "staff-lines", len(tuning))
        # staff-tuning 的 line 1 = 最下面那條線 = 最低的弦
        for line_no, string in enumerate(sorted(tuning, reverse=True), start=1):
            step, alter, octave = pitch_parts(tuning[string], use_flats=False)
            st = _sub(details, "staff-tuning", line=str(line_no))
            _sub(st, "tuning-step", step)
            if alter:
                _sub(st, "tuning-alter", alter)
            _sub(st, "tuning-octave", octave)
    else:
        _sub(clef, "sign", "F")
        _sub(clef, "line", 4)


def _tempo(measure: ET.Element, bpm: float) -> None:
    direction = _sub(measure, "direction", placement="above")
    dtype = _sub(direction, "direction-type")
    metro = _sub(dtype, "metronome", parentheses="no")
    _sub(metro, "beat-unit", "quarter")
    _sub(metro, "per-minute", int(round(bpm)))
    _sub(direction, "sound", tempo=f"{bpm:.1f}")


def score_to_musicxml(score: Score, path: Path, tab: bool = True) -> Path:
    """目前只輸出 score.lines[0]（階段一的貝斯）。"""
    line = score.lines[0]
    notes = sorted((n for n in line.notes if n.start_tick is not None), key=lambda n: n.start_tick)
    grid = score.grid
    bpb = grid.beats_per_bar
    use_flats = score.key_fifths < 0

    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", score.title)
    _sub(root, "movement-title", score.title)
    ident = _sub(root, "identification")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", f"music-to-score {__version__}")

    part_list = _sub(root, "part-list")
    sp = _sub(part_list, "score-part", id="P1")
    _sub(sp, "part-name", "Electric Bass")
    _sub(sp, "part-abbreviation", "E.Bass")
    si = _sub(sp, "score-instrument", id="P1-I1")
    _sub(si, "instrument-name", "Electric Bass")
    _sub(si, "instrument-sound", "pluck.bass.electric")
    mi = _sub(sp, "midi-instrument", id="P1-I1")
    _sub(mi, "midi-channel", 1)
    _sub(mi, "midi-program", 34)  # MusicXML 從 1 開始算：34 = Electric Bass (finger)

    part = _sub(root, "part", id="P1")
    bar_starts = grid.bar_start_beats()
    bar_starts.append(grid.num_beats)
    idx = 0
    for m_i, (b0, b1) in enumerate(zip(bar_starts[:-1], bar_starts[1:])):
        pickup = m_i == 0 and grid.pickup_beats > 0
        number = 0 if pickup else m_i + (0 if grid.pickup_beats else 1)
        measure = _sub(part, "measure", number=str(number))
        if pickup:
            measure.set("implicit", "yes")  # 弱起小節：拍號照寫完整的，MuseScore 會認得
        m_start, m_len = b0 * DIV, (b1 - b0) * DIV
        if not pickup:
            m_len = bpb * DIV  # 最後一小節不完整的話用休止符補滿
        if m_i == 0:
            _attributes(measure, score, tab, bpb)
            _tempo(measure, grid.tempo_bpm())

        while idx < len(notes) and notes[idx].end_tick <= m_start:
            idx += 1
        j = idx
        here = []
        while j < len(notes) and notes[j].start_tick < m_start + m_len:
            here.append(notes[j])
            j += 1
        pieces = layout_measure(here, m_start, m_len, bpb)
        if all(p.note is None for p in pieces) and not pickup:
            rest = _sub(measure, "note")
            _sub(rest, "rest", measure="yes")
            _sub(rest, "duration", m_len)
            _sub(rest, "voice", 1)
        else:
            for p in pieces:
                _note_xml(measure, p, use_flats, tab)

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    header = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + xml + "\n", encoding="utf-8")
    return path
