"""MIDI 輸出（type 1）：第 0 軌放 tempo map 和拍號，之後每條線一軌。

tempo map 每一拍一個速度事件，所以 MIDI 拉進 DAW 後，小節線會跟著原曲的速度變化，
而 MIDI 的第 0 tick 就是音訊的第 0 秒。

音符時間預設用「原始時間」（秒數換算成 tick），保留演奏的細節、和音訊完全對齊；
quantized=True 則用量化後的位置（和樂譜一樣）。
"""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np

from scoreapp.model import TICKS_PER_BEAT, BeatGrid, Line, Score

PPQ = 480
BASS_PROGRAM = 33  # GM：Electric Bass (finger)，從 0 開始算


def tempo_events(grid: BeatGrid, ppq: int = PPQ) -> list[tuple[int, int]]:
    """回傳 [(tick, 每拍微秒數)]，相同速度的連續拍合併。"""
    out: list[tuple[int, int]] = []
    for i, dt in enumerate(np.diff(grid.times)):
        tempo = int(round(dt * 1_000_000))
        if not out or out[-1][1] != tempo:
            out.append((i * ppq, tempo))
    return out


def _note_ticks(line: Line, grid: BeatGrid, quantized: bool, ppq: int) -> list[tuple[int, int, int, int]]:
    out = []
    for n in line.notes:
        if quantized and n.start_tick is not None and n.end_tick is not None:
            s = n.start_tick * ppq // TICKS_PER_BEAT
            e = n.end_tick * ppq // TICKS_PER_BEAT
        else:
            s = int(round(grid.time_to_beat(n.onset) * ppq))
            e = int(round(grid.time_to_beat(n.offset) * ppq))
        out.append((max(0, s), max(s + 1, e), n.pitch, int(np.clip(n.velocity, 1, 127))))
    return out


def _to_track(events: list[tuple[int, mido.Message | mido.MetaMessage]]) -> mido.MidiTrack:
    # 同一個 tick：先 note_off 再 note_on，避免同音高相連時被吃掉
    events.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))
    track = mido.MidiTrack()
    now = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def score_to_midi(score: Score, path: Path, quantized: bool = False, ppq: int = PPQ,
                  programs: list[int] | None = None) -> Path:
    grid = score.grid
    # 軌名用 UTF-8，中文歌名才寫得進去（Logic、Cubase 都讀得懂）
    mid = mido.MidiFile(type=1, ticks_per_beat=ppq, charset="utf-8")

    conductor: list[tuple[int, mido.MetaMessage]] = [(0, mido.MetaMessage("track_name", name=score.title))]
    for tick, tempo in tempo_events(grid, ppq):
        conductor.append((tick, mido.MetaMessage("set_tempo", tempo=tempo)))
    if grid.pickup_beats:
        conductor.append((0, mido.MetaMessage("time_signature", numerator=grid.pickup_beats, denominator=4)))
    conductor.append((grid.pickup_beats * ppq,
                      mido.MetaMessage("time_signature", numerator=grid.beats_per_bar, denominator=4)))
    key = _key_name(score.key_fifths, score.key_mode)
    if key:
        conductor.append((0, mido.MetaMessage("key_signature", key=key)))
    mid.tracks.append(_to_track(conductor))

    for idx, line in enumerate(score.lines):
        channel = idx if idx < 9 else idx + 1  # 跳過第 10 軌（鼓）
        channel %= 16
        program = programs[idx] if programs else BASS_PROGRAM
        ev: list[tuple[int, mido.Message | mido.MetaMessage]] = [
            (0, mido.MetaMessage("track_name", name=line.name)),
            (0, mido.Message("program_change", program=program, channel=channel)),
        ]
        for s, e, pitch, vel in _note_ticks(line, grid, quantized, ppq):
            ev.append((s, mido.Message("note_on", note=pitch, velocity=vel, channel=channel)))
            ev.append((e, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))
        mid.tracks.append(_to_track(ev))

    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


_MAJOR = {-7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F", 0: "C",
          1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#"}
_MINOR = {-7: "Abm", -6: "Ebm", -5: "Bbm", -4: "Fm", -3: "Cm", -2: "Gm", -1: "Dm", 0: "Am",
          1: "Em", 2: "Bm", 3: "F#m", 4: "C#m", 5: "G#m", 6: "D#m", 7: "A#m"}


def _key_name(fifths: int, mode: str) -> str | None:
    return (_MINOR if mode == "minor" else _MAJOR).get(fifths)
