"""調號：用 music21 的調性分析（Krumhansl-Schmuckler 類的方法），以音長加權。"""

from __future__ import annotations

from scoreapp.model import NoteEvent


def detect_key(notes: list[NoteEvent]) -> tuple[int, str]:
    """回傳（升降記號數量, "major" / "minor"）。正數是升記號，負數是降記號。"""
    if len(notes) < 8:
        return 0, "major"
    from music21 import note, stream

    s = stream.Stream()
    for n in notes:
        m = note.Note(n.pitch)
        m.quarterLength = max(0.25, min(4.0, round(n.duration * 4) / 4))
        s.append(m)
    k = s.analyze("key")
    sharps = int(k.sharps)
    # 7 個升降記號的調用等音的少記號調表示（例如 C# 大調 → D♭ 大調）
    if sharps > 6:
        sharps -= 12
    elif sharps < -6:
        sharps += 12
    return sharps, "minor" if k.mode == "minor" else "major"
