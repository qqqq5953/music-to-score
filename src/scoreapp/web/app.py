"""本機網頁介面：貼上 YouTube 連結或拖入音檔 → 進度條 → 下載檔案。

只聽 127.0.0.1（不會對外開放）。工作一次跑一個（分軌很吃記憶體），其他排隊。
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from scoreapp.fetch import AUDIO_EXTENSIONS, is_url, safe_name
from scoreapp.paths import OUTPUT_DIR

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
DOWNLOADS = ["bass.musicxml", "bass.mid", "no_bass.wav", "bass.wav", "mix.wav"]


@dataclass
class Job:
    id: str
    source: str
    label: str
    midi_quantized: bool = False
    status: str = "queued"  # queued / running / done / error
    progress: float = 0.0
    message: str = "排隊中…"
    title: str | None = None
    folder: str | None = None
    files: list[str] = field(default_factory=list)
    info: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    upload_dir: str | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("source")
        d.pop("upload_dir")
        return d


def run_job_pipeline(job: Job, out_root: Path, progress) -> tuple[Path, list[str], dict, str]:
    """實際執行轉譜（測試時可以換掉）。回傳（輸出資料夾, 檔名列表, 分析資訊, 歌名）。"""
    from scoreapp.pipelines.bass import BassOptions, run_bass_pipeline

    result = run_bass_pipeline(job.source, out_root, BassOptions(midi_quantized=job.midi_quantized), progress=progress)
    return result.out_dir, list(result.files), result.info, result.info["title"]


def friendly_error(exc: BaseException) -> str:
    text = str(exc)
    name = type(exc).__name__
    if "DownloadError" in name or "yt_dlp" in type(exc).__module__:
        return f"YouTube 下載失敗，請確認連結是否正確、影片是否公開。（{text[:200]}）"
    module = type(exc).__module__
    if module.startswith(("huggingface_hub", "urllib", "requests")) or "urlopen" in text or "Connection" in name:
        return ("模型或檔案下載失敗，請確認網路連線。第一次使用時需要下載分軌模型（約 300MB）"
                f"和節拍模型。（{text[:200]}）")
    if isinstance(exc, FileNotFoundError):
        return f"找不到檔案：{text}"
    if "ffmpeg" in text:
        return f"音檔讀取失敗，檔案可能損毀或格式不支援。（{text[:200]}）"
    return f"{name}: {text[:300]}"


class JobManager:
    def __init__(self, out_root: Path) -> None:
        self.out_root = out_root
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, name="scoreapp-worker", daemon=True)
        self.worker.start()

    def submit(self, job: Job) -> Job:
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "找不到這個工作")
        return job

    def _loop(self) -> None:
        while True:
            job = self.jobs[self.queue.get()]
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status, job.message = "running", "開始…"

        def progress(fraction: float, message: str) -> None:
            job.progress = round(float(fraction), 4)
            job.message = message

        try:
            out_dir, files, info, title = run_job_pipeline(job, self.out_root, progress)
            job.folder, job.files, job.info, job.title = out_dir.name, [f for f in DOWNLOADS if f in files], info, title
            job.status, job.progress, job.message = "done", 1.0, "完成"
        except Exception as exc:  # 把錯誤顯示在網頁上，不要讓背景工作掛掉
            log.error("工作失敗：%s", traceback.format_exc())
            job.status, job.error, job.message = "error", friendly_error(exc), "失敗"
        finally:
            if job.upload_dir:
                shutil.rmtree(job.upload_dir, ignore_errors=True)


def create_app(out_root: Path | None = None) -> FastAPI:
    out_root = Path(out_root or OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    manager = JobManager(out_root)
    app = FastAPI(title="music-to-score")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/jobs")
    async def create_job(
        url: str = Form(""),
        midi_quantized: bool = Form(False),
        file: UploadFile | None = File(None),
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        url = url.strip()
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in AUDIO_EXTENSIONS:
                raise HTTPException(400, f"不支援的檔案格式：{suffix or '（沒有副檔名）'}")
            upload_dir = out_root / ".uploads" / job_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / (safe_name(Path(file.filename).stem) + suffix)
            with dest.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            job = Job(job_id, str(dest), Path(file.filename).name, midi_quantized, upload_dir=str(upload_dir))
        elif url:
            if not is_url(url):
                raise HTTPException(400, "請貼上以 http:// 或 https:// 開頭的連結")
            job = Job(job_id, url, url, midi_quantized)
        else:
            raise HTTPException(400, "請貼上 YouTube 連結或選擇音檔")
        manager.submit(job)
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return manager.get(job_id).public()

    @app.get("/api/results")
    def results() -> list[dict]:
        out = []
        for d in out_root.iterdir():
            info_file = d / "analysis.json"
            if d.is_dir() and not d.name.startswith(".") and info_file.exists():
                try:
                    info = json.loads(info_file.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                files = [f for f in DOWNLOADS if (d / f).exists()]
                out.append({"folder": d.name, "info": info, "files": files, "mtime": info_file.stat().st_mtime})
        return sorted(out, key=lambda r: -r["mtime"])

    @app.get("/api/results/{folder}/{name}")
    def download(folder: str, name: str) -> FileResponse:
        if name not in DOWNLOADS:
            raise HTTPException(404, "沒有這個檔案")
        folders = {d.name: d for d in out_root.iterdir() if d.is_dir() and not d.name.startswith(".")}
        if folder not in folders or not (folders[folder] / name).exists():
            raise HTTPException(404, "沒有這個檔案")
        media = {"musicxml": "application/vnd.recordare.musicxml+xml", "mid": "audio/midi", "wav": "audio/wav"}
        return FileResponse(folders[folder] / name, media_type=media[name.rsplit(".", 1)[1]],
                            filename=f"{folder} - {name}")

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    import socket

    import uvicorn

    url = f"http://{host}:{port}"
    with socket.socket() as s:
        if s.connect_ex((host, port)) == 0:
            # 已經有一個在跑了（例如重複雙擊 start.command）→ 直接開瀏覽器
            print(f"已經在執行中：{url}")
            if open_browser:
                webbrowser.open(url)
            return

    if open_browser:
        def wait_and_open() -> None:
            for _ in range(100):
                with socket.socket() as s:
                    if s.connect_ex((host, port)) == 0:
                        webbrowser.open(url)
                        return
                time.sleep(0.2)

        threading.Thread(target=wait_and_open, daemon=True).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
