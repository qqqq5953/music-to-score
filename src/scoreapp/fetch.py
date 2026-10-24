"""取得音訊：YouTube 連結或本機音檔 → 44.1kHz 立體聲 wav。

ffmpeg 用 imageio-ffmpeg 內附的執行檔，不需要另外安裝。
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".webm", ".mp4"}


@dataclass
class FetchedAudio:
    wav_path: Path
    title: str


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip(), re.IGNORECASE))


def safe_name(name: str, max_len: int = 80) -> str:
    """把歌名變成可以當資料夾名稱的字串（保留中文）。"""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    name = re.sub(r"\s+", " ", name)
    return name[:max_len] or "untitled"


def to_wav(src: Path, dst: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """任何 ffmpeg 讀得懂的檔案 → 16-bit 立體聲 wav。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 轉檔失敗：{proc.stderr.strip()[-500:]}")
    return dst


def download_youtube(url: str, work_dir: Path) -> tuple[Path, str]:
    """下載 YouTube 的最佳音訊串流，回傳（原始檔路徑, 影片標題）。

    不使用 yt-dlp 的轉檔功能（那需要 ffprobe），下載後自己用 ffmpeg 轉 wav。
    """
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_exe(),
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        # 有些情況副檔名會和 prepare_filename 不同
        candidates = sorted(work_dir.glob("download.*"))
        if not candidates:
            raise RuntimeError("YouTube 下載失敗：找不到下載的檔案")
        path = candidates[0]
    return path, str(info.get("title") or "youtube")


def fetch(source: str, work_dir: Path) -> FetchedAudio:
    """source 是 YouTube 連結或音檔路徑；輸出 work_dir/mix.wav。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "mix.wav"
    if is_url(source):
        raw, title = download_youtube(source.strip(), work_dir / "_download")
        to_wav(raw, out)
        for f in raw.parent.glob("*"):
            f.unlink(missing_ok=True)
        raw.parent.rmdir()
        return FetchedAudio(out, title)

    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"找不到檔案：{src}")
    if src.suffix.lower() not in AUDIO_EXTENSIONS:
        log.warning("不認得的副檔名 %s，仍嘗試用 ffmpeg 讀取", src.suffix)
    to_wav(src, out)
    return FetchedAudio(out, src.stem)


def guess_title(source: str) -> str:
    """不下載就能知道的名稱（用來決定輸出資料夾）；URL 的話要等下載後才知道。"""
    return "youtube" if is_url(source) else Path(source).stem
