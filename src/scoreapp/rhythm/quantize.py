"""量化：把音符對齊節拍格。

- 每一拍各自決定用「16 分音符格」（每拍 4 格）還是「8 分三連音格」（每拍 3 格）。
  三連音的誤差要明顯比較小才選三連音（triplet_bias），避免一般的不準被當成三連音。
- 音的結尾用結尾所在那一拍的格子對齊。
- 同一格只留一個音（單聲部），重疊的截斷，太短的空隙併進前一個音（樂譜比較好讀）。

量化只改 start_tick / end_tick，秒數（onset / offset）保留原樣給 MIDI 用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scoreapp.model import TICKS_PER_BEAT, BeatGrid, Line, NoteEvent

STRAIGHT = 4  # 每拍 4 格（16 分音符）
TRIPLET = 3  # 每拍 3 格（8 分三連音）


@dataclass
class QuantizeParams:
    triplets: bool = True
    triplet_bias: float = 0.6  # 三連音的誤差 < 直拍誤差 × 這個值，才用三連音
    min_note_sec: float = 0.05  # 比這個短的音直接刪掉
    fill_gap_ticks: int = 3  # 小於等於這個長度（3 = 一個 16 分音符）的空隙併進前一個音


def _choose_divisions(fracs: list[float], params: QuantizeParams) -> int:
    if not params.triplets or not fracs:
        return STRAIGHT
    f = np.array(fracs)
    err_s = np.abs(f * STRAIGHT - np.round(f * STRAIGHT)).sum() / STRAIGHT
    err_t = np.abs(f * TRIPLET - np.round(f * TRIPLET)).sum() / TRIPLET
    return TRIPLET if err_t < err_s * params.triplet_bias else STRAIGHT


def _snap(beat_pos: float, divisions: dict[int, int]) -> int:
    k = int(np.floor(beat_pos))
    div = divisions.get(k, STRAIGHT)
    step = TICKS_PER_BEAT // div
    return k * TICKS_PER_BEAT + int(round((beat_pos - k) * div)) * step


def quantize_line(line: Line, grid: BeatGrid, params: QuantizeParams | None = None) -> set[int]:
    """就地量化 line.notes（會刪掉太短或撞在一起的音）。回傳用三連音格的拍。"""
    params = params or QuantizeParams()
    notes = [n for n in line.notes if n.duration >= params.min_note_sec]
    if not notes:
        line.notes = []
        return set()
    on_b = grid.time_to_beat(np.array([n.onset for n in notes]))
    off_b = grid.time_to_beat(np.array([n.offset for n in notes]))

    per_beat: dict[int, list[float]] = {}
    for b in on_b:
        k = int(np.floor(b))
        per_beat.setdefault(k, []).append(float(b - k))
    divisions = {k: _choose_divisions(fr, params) for k, fr in per_beat.items()}

    for n, b0, b1 in zip(notes, on_b, off_b):
        n.start_tick = max(0, _snap(float(b0), divisions))
        n.end_tick = _snap(float(b1), divisions)

    # 同一格起音只留一個：比較大聲的，一樣大聲就留比較長的
    notes.sort(key=lambda n: (n.start_tick, -n.velocity, -(n.offset - n.onset)))
    unique: list[NoteEvent] = []
    for n in notes:
        if unique and unique[-1].start_tick == n.start_tick:
            continue
        unique.append(n)

    for i, n in enumerate(unique):
        nxt = unique[i + 1].start_tick if i + 1 < len(unique) else None
        beat = n.start_tick // TICKS_PER_BEAT
        unit = TICKS_PER_BEAT // divisions.get(beat, STRAIGHT)
        if n.end_tick <= n.start_tick:
            n.end_tick = n.start_tick + unit
        if nxt is not None:
            n.end_tick = min(n.end_tick, nxt)
            if 0 < nxt - n.end_tick <= params.fill_gap_ticks:
                n.end_tick = nxt
    line.notes = unique
    return {k for k, d in divisions.items() if d == TRIPLET}


def total_ticks(grid: BeatGrid) -> int:
    return grid.num_beats * TICKS_PER_BEAT
