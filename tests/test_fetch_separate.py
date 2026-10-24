import numpy as np
import soundfile as sf

from scoreapp import fetch, separate


def test_is_url_and_safe_name():
    assert fetch.is_url("https://www.youtube.com/watch?v=abc")
    assert not fetch.is_url("/Users/me/song.mp3")
    assert fetch.safe_name('a/b:c*?"<>|  歌名 ') == "a_b_c______ 歌名"
    assert fetch.safe_name("...") == "untitled"


def test_fetch_local_file_converts_to_44k_stereo(tmp_path):
    sr = 22050
    t = np.arange(sr) / sr
    sf.write(tmp_path / "in.flac", 0.3 * np.sin(2 * np.pi * 110 * t), sr)
    got = fetch.fetch(str(tmp_path / "in.flac"), tmp_path / "work")
    assert got.title == "in"
    audio, out_sr = sf.read(got.wav_path, always_2d=True)
    assert out_sr == 44100
    assert audio.shape[1] == 2
    assert abs(len(audio) / out_sr - 1.0) < 0.05


def test_pick_device_falls_back_to_cpu_without_gpu():
    import torch

    dev = separate.pick_device()
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        assert dev == "cpu"
    assert separate.pick_device("cpu") == "cpu"


def test_bass_submodel_picks_specialised_model():
    from demucs.apply import BagOfModels
    from demucs.pretrained import demucs_unittest

    models = [demucs_unittest() for _ in range(4)]
    bag = BagOfModels(models, weights=np.eye(4).tolist())
    bass_idx = bag.sources.index("bass")
    assert separate._bass_submodel(bag) is models[bass_idx]
    uniform = BagOfModels(models[:2])
    assert separate._bass_submodel(uniform) is uniform


def test_separate_bass_writes_stems_that_sum_to_mix(tmp_path, monkeypatch):
    """用隨機權重的小模型跑完整流程（不需要下載），檢查檔案與 bass + no_bass = mix。"""
    from demucs.pretrained import demucs_unittest

    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    mix = np.stack([0.2 * np.sin(2 * np.pi * 55 * t), 0.2 * np.sin(2 * np.pi * 440 * t)], axis=1)
    sf.write(tmp_path / "mix.wav", mix, sr, subtype="PCM_16")
    monkeypatch.setattr(separate, "load_model", lambda name: demucs_unittest().eval())

    progress = []
    stems = separate.separate_bass(tmp_path / "mix.wav", tmp_path, device="cpu", progress=progress.append)
    bass, _ = sf.read(stems.bass, always_2d=True)
    no_bass, _ = sf.read(stems.no_bass, always_2d=True)
    orig, _ = sf.read(tmp_path / "mix.wav", always_2d=True)
    assert bass.shape == no_bass.shape == orig.shape
    assert np.max(np.abs(bass + no_bass - orig)) < 2e-3
    assert progress and all(0 <= p <= 1 for p in progress)


def test_youtube_download_options(tmp_path, monkeypatch):
    """不連網：檢查傳給 yt-dlp 的設定（deno 路徑、ffmpeg、只抓單一影片），以及下載後轉成 wav。"""
    import yt_dlp

    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            sr = 44100
            t = np.arange(sr // 2) / sr
            sf.write(tmp_path / "work" / "_download" / "download.wav", 0.2 * np.sin(2 * np.pi * 220 * t), sr)
            return {"title": "Song / Title", "ext": "wav"}

        def prepare_filename(self, info):
            return str(tmp_path / "work" / "_download" / "download.wav")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    got = fetch.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "work")
    assert got.title == "Song / Title"
    assert sf.info(str(got.wav_path)).samplerate == 44100
    assert captured["noplaylist"] is True
    assert captured["ffmpeg_location"] == fetch.ffmpeg_exe()
    assert captured["js_runtimes"]["deno"]["path"].endswith("deno")
    assert not (tmp_path / "work" / "_download").exists()
