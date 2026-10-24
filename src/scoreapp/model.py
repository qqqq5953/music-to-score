"""各階段共用的資料格式。

NoteEvent（一個音）→ Line（一條單聲部線）→ Score（多條線 + 節拍格 + 調號 + 拍號）。

時間有兩套座標：
- 秒（onset / offset）：和原曲音訊對齊，MIDI 用這套。
- tick（start_tick / end_tick）：量化後的位置，每拍 TICKS_PER_BEAT 個 tick，樂譜用這套。
  12 可以同時表示 16 分音符（3 tick）和 8 分三連音（4 tick）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TICKS_PER_BEAT = 12


@dataclass
class NoteEvent:
    pitch: int  # MIDI 音高（實際發聲的音高，E1 = 28）
    onset: float  # 秒
    offset: float  # 秒
    velocity: int = 80
    # 量化後才有
    start_tick: int | None = None
    end_tick: int | None = None
    # TAB 把位後才有；弦號跟 MusicXML 一樣：1 = 最高的弦（G），4 = 最低的弦（E）
    string: int | None = None
    fret: int | None = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Line:
    """一條單聲部線（任一時間最多一個音）。"""

    notes: list[NoteEvent]
    name: str = "Bass"
    kind: str = "melody"  # "melody"（旋律線）或 "pad"（和聲長音，階段二用）
    register: str | None = None  # "high" / "mid" / "low"（階段二用）


@dataclass
class BeatGrid:
    """節拍格。

    times[0] 一定是 0 秒（音訊開頭），times 是每一拍開始的時間。
    第 downbeat_index 拍是第一個完整小節的第一拍；在它之前的拍是弱起小節（pickup）。
    最後一個元素是結尾邊界，所以拍數 = len(times) - 1。
    """

    times: np.ndarray
    beats_per_bar: int = 4
    downbeat_index: int = 0

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if len(self.times) < 2 or abs(self.times[0]) > 1e-9:
            raise ValueError("BeatGrid.times 必須從 0 開始且至少有兩個點")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("BeatGrid.times 必須嚴格遞增")

    @property
    def num_beats(self) -> int:
        return len(self.times) - 1

    @property
    def pickup_beats(self) -> int:
        return self.downbeat_index

    def time_to_beat(self, t: float | np.ndarray) -> float | np.ndarray:
        """秒 → 拍（浮點數）。超出範圍時用頭尾的拍長外插。"""
        t_arr = np.asarray(t, dtype=float)
        times = self.times
        idx = np.clip(np.searchsorted(times, t_arr, side="right") - 1, 0, len(times) - 2)
        beat = idx + (t_arr - times[idx]) / (times[idx + 1] - times[idx])
        return float(beat) if np.ndim(t) == 0 else beat

    def beat_to_time(self, b: float | np.ndarray) -> float | np.ndarray:
        b_arr = np.asarray(b, dtype=float)
        times = self.times
        idx = np.clip(np.floor(b_arr).astype(int), 0, len(times) - 2)
        t = times[idx] + (b_arr - idx) * (times[idx + 1] - times[idx])
        return float(t) if np.ndim(b) == 0 else t

    def tick_to_time(self, tick: int | np.ndarray) -> float | np.ndarray:
        return self.beat_to_time(np.asarray(tick, dtype=float) / TICKS_PER_BEAT)

    def tempo_bpm(self) -> float:
        """代表速度（拍長中位數），給樂譜上的速度記號用。"""
        body = np.diff(self.times[self.downbeat_index :]) if self.num_beats > self.downbeat_index else np.diff(self.times)
        return float(60.0 / np.median(body))

    def bar_start_beats(self) -> list[int]:
        """每個小節開頭是第幾拍（含弱起小節）。最後一小節若不完整也算一個。"""
        starts = [0] if self.downbeat_index > 0 else []
        b = self.downbeat_index
        while b < self.num_beats:
            starts.append(b)
            b += self.beats_per_bar
        return starts


@dataclass
class Score:
    lines: list[Line]
    grid: BeatGrid
    key_fifths: int = 0  # 正數 = 升記號數量，負數 = 降記號數量
    key_mode: str = "major"
    title: str = "Untitled"

    @property
    def time_signature(self) -> tuple[int, int]:
        return (self.grid.beats_per_bar, 4)
