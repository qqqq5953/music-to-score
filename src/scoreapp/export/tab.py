"""TAB 把位指派：動態規劃，讓左手的移動最少。

每個音可以在幾條弦上彈（例如 G2 = E 弦 15 格、A 弦 10 格、D 弦 5 格、G 弦空弦）。
左手用「把位」h 表示（食指所在的格），一指一格可以按到 h～h+span 格，不用移動。
DP 的狀態是（弦, 格, 把位）；空弦不需要按，把位沿用前一個音的。
成本：
- 格數越高越貴（低把位比較好彈），12 格以上再加一點。
- 換把：只要換就有固定成本，再加上移動的格數；中間有休止符或空弦的話有時間換，比較便宜。
- 換弦：跨越的弦數。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scoreapp.model import NoteEvent

# MusicXML 的弦號：1 = 最高的弦。4 弦貝斯標準調音 E1-A1-D2-G2。
BASS_4_STANDARD: dict[int, int] = {1: 43, 2: 38, 3: 33, 4: 28}

State = tuple[int, int, int | None]  # （弦, 格, 把位）


@dataclass
class TabParams:
    tuning: dict[int, int] = field(default_factory=lambda: dict(BASS_4_STANDARD))
    max_fret: int = 24
    span: int = 3  # 一指一格：食指到小指差 3 格
    fret_cost: float = 0.03
    high_fret_cost: float = 0.05  # 12 格以上每格額外
    shift_cost: float = 1.0  # 換把的固定成本
    shift_per_fret: float = 0.15
    string_cost: float = 0.1  # 每跨一條弦


def candidates(pitch: int, params: TabParams) -> list[tuple[int, int]]:
    out = []
    for string, open_pitch in params.tuning.items():
        fret = pitch - open_pitch
        if 0 <= fret <= params.max_fret:
            out.append((string, fret))
    return out


def playable_range(params: TabParams) -> tuple[int, int]:
    return min(params.tuning.values()), max(params.tuning.values()) + params.max_fret


def _states(string: int, fret: int, prev_hand: int | None, p: TabParams) -> list[State]:
    if fret == 0:
        return [(string, 0, prev_hand)]
    return [(string, fret, h) for h in range(max(1, fret - p.span), fret + 1)]


def _note_cost(fret: int, p: TabParams) -> float:
    return fret * p.fret_cost + max(0, fret - 12) * p.high_fret_cost


def _transition_cost(prev: State, cur: State, free_time: float, p: TabParams) -> float:
    cost = abs(cur[0] - prev[0]) * p.string_cost
    h0, h1 = prev[2], cur[2]
    if h0 is not None and h1 is not None and h0 != h1:
        cost += (p.shift_cost + abs(h1 - h0) * p.shift_per_fret) / (1.0 + 4.0 * max(free_time, 0.0))
    return cost


def assign_tab(notes: list[NoteEvent], params: TabParams | None = None) -> list[NoteEvent]:
    """就地設定每個音的 string / fret。彈不到的音（超出音域）會先移八度。"""
    params = params or TabParams()
    lo, hi = playable_range(params)
    for n in notes:
        while n.pitch < lo:
            n.pitch += 12
        while n.pitch > hi:
            n.pitch -= 12
    if not notes:
        return notes
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset)

    # layers[k]：狀態 →（累積成本, 前一層的狀態）
    layers: list[dict[State, tuple[float, State | None]]] = []
    prev_note: NoteEvent | None = None
    for i in order:
        n = notes[i]
        layer: dict[State, tuple[float, State | None]] = {}
        for string, fret in candidates(n.pitch, params):
            base = _note_cost(fret, params)
            if not layers:
                for state in _states(string, fret, None, params):
                    layer[state] = (base, None)
                continue
            for pstate, (pcost, _) in layers[-1].items():
                # 換把的時間：前一個音放開之後；前一個是空弦的話，它一響起手就自由了
                free_from = prev_note.onset if pstate[1] == 0 else prev_note.offset
                for state in _states(string, fret, pstate[2], params):
                    total = pcost + base + _transition_cost(pstate, state, n.onset - free_from, params)
                    if state not in layer or total < layer[state][0]:
                        layer[state] = (total, pstate)
        layers.append(layer)
        prev_note = n

    state = min(layers[-1], key=lambda s: layers[-1][s][0])
    for k in range(len(layers) - 1, -1, -1):
        n = notes[order[k]]
        n.string, n.fret = state[0], state[1]
        state = layers[k][state][1]
    return notes
