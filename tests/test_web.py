import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from scoreapp.web import app as web
from synth import SR, render_notes, riff_events


@pytest.fixture
def client(tmp_path, monkeypatch):
    from scoreapp.pipelines.bass import BassOptions, run_bass_pipeline

    def fake_pipeline(job, out_root, progress):
        # 雲端沒有 Demucs 模型：輸入本身就是貝斯，節拍用 librosa
        r = run_bass_pipeline(job.source, out_root, BassOptions(separate=False, use_beat_this=False,
                                                                 midi_quantized=job.midi_quantized), progress=progress)
        return r.out_dir, list(r.files), r.info, r.info["title"]

    monkeypatch.setattr(web, "run_job_pipeline", fake_pipeline)
    return TestClient(web.create_app(tmp_path / "output"))


def wait(client, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.3)
    raise TimeoutError


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "貝斯轉譜" in r.text


def test_upload_and_download(client, tmp_path):
    secs, _ = riff_events(110, start=0.3, repeats=2)
    path = tmp_path / "我的歌.wav"
    sf.write(path, render_notes(secs, 10.0), SR)
    with path.open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("我的歌.wav", fh, "audio/wav")}, data={"midi_quantized": "true"})
    assert r.status_code == 200, r.text
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert job["progress"] == 1.0
    assert job["folder"] == "我的歌"
    assert {"bass.musicxml", "bass.mid", "bass.wav", "mix.wav"} <= set(job["files"])
    assert job["info"]["notes"] > 10

    d = client.get(f"/api/results/{job['folder']}/bass.musicxml")
    assert d.status_code == 200
    assert b"<score-partwise" in d.content
    assert "attachment" in d.headers["content-disposition"]

    hist = client.get("/api/results").json()
    assert hist[0]["folder"] == "我的歌"
    # 上傳的暫存檔要清掉
    assert not any((tmp_path / "output" / ".uploads").glob("*/*"))


def test_rejects_bad_input(client):
    assert client.post("/api/jobs", data={"url": ""}).status_code == 400
    assert client.post("/api/jobs", data={"url": "not a url"}).status_code == 400
    r = client.post("/api/jobs", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404


def test_download_rejects_path_tricks(client):
    assert client.get("/api/results/..%2F..%2Fetc/bass.mid").status_code == 404
    assert client.get("/api/results/x/passwd").status_code == 404


def test_error_is_reported(client, monkeypatch):
    def boom(job, out_root, progress):
        raise FileNotFoundError("missing.mp3")

    monkeypatch.setattr(web, "run_job_pipeline", boom)
    r = client.post("/api/jobs", data={"url": "https://www.youtube.com/watch?v=x"})
    job = wait(client, r.json()["id"])
    assert job["status"] == "error"
    assert "找不到檔案" in job["error"]
