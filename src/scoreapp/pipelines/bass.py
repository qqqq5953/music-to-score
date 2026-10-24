"""階段一流程：歌曲 → 貝斯 TAB / MIDI / 分軌。

每個步驟都是獨立模組，之後有更好的模型可以直接替換：
fetch → separate → transcribe.mono → rhythm.beats → rhythm.quantize → key → export.*
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from scoreapp import fetch as fetch_mod
from scoreapp.export.midi import score_to_midi
from scoreapp.export.musicxml import score_to_musicxml
from scoreapp.export.tab import TabParams, assign_tab
from scoreapp.key import detect_key
from scoreapp.model import Score
from scoreapp.rhythm.beats import analyze_rhythm
from scoreapp.rhythm.quantize import QuantizeParams, quantize_line
from scoreapp.transcribe.mono import MonoParams, transcribe_bass

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class BassOptions:
    mono: MonoParams = field(default_factory=MonoParams)
    quantize: QuantizeParams = field(default_factory=QuantizeParams)
    tab: TabParams = field(default_factory=TabParams)
    use_beat_this: bool = True
    demucs_model: str = "htdemucs_ft"
    device: str | None = None
    midi_quantized: bool = False  # False：MIDI 用原始時間（和音訊完全對齊）
    separate: bool = True  # False：輸入已經是貝斯單軌（測試用）


@dataclass
class BassResult:
    out_dir: Path
    files: dict[str, Path]
    score: Score
    info: dict


def _stage(progress: ProgressFn | None, start: float, end: float):
    def report(fraction: float, message: str) -> None:
        if progress:
            progress(start + (end - start) * min(max(fraction, 0.0), 1.0), message)

    return report


def run_bass_pipeline(
    source: str,
    out_root: Path,
    options: BassOptions | None = None,
    progress: ProgressFn | None = None,
) -> BassResult:
    options = options or BassOptions()
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / f".work-{uuid.uuid4().hex[:8]}"
    try:
        report = _stage(progress, 0.0, 0.08)
        report(0.0, "下載 / 讀取音訊…" if fetch_mod.is_url(source) else "讀取音訊…")
        fetched = fetch_mod.fetch(source, work)
        title = fetched.title
        out_dir = out_root / fetch_mod.safe_name(title)
        out_dir.mkdir(parents=True, exist_ok=True)
        mix = out_dir / "mix.wav"
        shutil.move(str(fetched.wav_path), mix)
        info = sf.info(str(mix))
        duration = info.frames / info.samplerate

        if options.separate:
            from scoreapp.separate import separate_bass

            report = _stage(progress, 0.08, 0.60)
            report(0.0, "分離貝斯音軌（Demucs）…")
            stems = separate_bass(mix, out_dir, options.demucs_model, options.device,
                                  progress=lambda f: report(f, "分離貝斯音軌（Demucs）…"))
            bass_wav, no_bass_wav = stems.bass, stems.no_bass
        else:
            bass_wav = out_dir / "bass.wav"
            shutil.copy(mix, bass_wav)
            no_bass_wav = None

        report = _stage(progress, 0.60, 0.82)
        report(0.0, "偵測貝斯音符…")
        line = transcribe_bass(bass_wav, options.mono)

        report = _stage(progress, 0.82, 0.92)
        report(0.0, "偵測節拍與小節線…")
        grid, beat_source = analyze_rhythm(mix, duration, use_beat_this=options.use_beat_this)

        report = _stage(progress, 0.92, 1.0)
        report(0.0, "量化、指派把位、輸出檔案…")
        triplet_beats = quantize_line(line, grid, options.quantize)
        assign_tab(line.notes, options.tab)
        fifths, mode = detect_key(line.notes)
        score = Score(lines=[line], grid=grid, key_fifths=fifths, key_mode=mode, title=title)

        files = {
            "bass.musicxml": score_to_musicxml(score, out_dir / "bass.musicxml"),
            "bass.mid": score_to_midi(score, out_dir / "bass.mid", quantized=options.midi_quantized),
            "bass.wav": bass_wav,
        }
        if no_bass_wav is not None:
            files["no_bass.wav"] = no_bass_wav
        files["mix.wav"] = mix

        result_info = {
            "title": title,
            "duration_sec": round(duration, 2),
            "tempo_bpm": round(grid.tempo_bpm(), 1),
            "time_signature": f"{grid.beats_per_bar}/4",
            "pickup_beats": grid.pickup_beats,
            "key": {"fifths": fifths, "mode": mode},
            "beat_source": beat_source,
            "notes": len(line.notes),
            "triplet_beats": len(triplet_beats),
        }
        (out_dir / "analysis.json").write_text(json.dumps(result_info, ensure_ascii=False, indent=2), encoding="utf-8")
        report(1.0, "完成")
        return BassResult(out_dir, files, score, result_info)
    finally:
        shutil.rmtree(work, ignore_errors=True)
