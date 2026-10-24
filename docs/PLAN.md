# music-to-score 計畫

## 討論中確認的決定

1. **階段一的「貝斯譜」= TAB**。要用**帶節奏符桿的 TAB**（TAB 數字下方畫符桿和符尾，看得出節奏），不要純 TAB。要不要顯示五線譜，使用者在 MuseScore 裡自己切換。
2. **MIDI 是很重要的輸出**：要在 Logic 或 Cubase 裡逐音編輯，所以要帶 tempo map，讓 MIDI 拉進 DAW 後能和原曲音訊對齊。
3. **輸入**：YouTube 連結是主要方式，音檔（mp3、wav、m4a、flac）是次要方式。只供個人練習使用。
4. **階段二是管弦樂（電影配樂）**，目的是作曲時分析配器：
   - 依「旋律線條」判斷，**不分樂器**。齊奏、八度重疊只取一條。
   - **高、中、低是最少的分區，不是只有三條線**。同一個音域裡有幾條獨立線條就盡量抓幾條，例如中音域的和聲和副旋律要分成兩條。
   - 旋律跨越音域時跟著線條走，不用固定音高硬切。
   - 抓不完整可以接受，每個音域至少要有一條。
5. **流行歌主旋律**是之後的選用功能，不在這次範圍內，但架構要能接得上。
6. 架構要模組化，每個階段（分軌、轉譜、節拍、量化、聲部、輸出）都可以替換，之後有更好的模型就能直接換上。

使用者環境：4 弦貝斯（E-A-D-G 標準調音）、Logic Pro / Cubase、MuseScore 4；MacBook Pro 16 吋 M5 Pro 24GB；
系統沒有 Python、不裝 Homebrew → 用 `uv` 管理 Python 與套件，ffmpeg 用 `imageio-ffmpeg` 內建執行檔，雙擊 `start.command` 啟動本機網頁。

---

## 規格

### 階段一：流行歌貝斯線

| 項目 | 內容 |
|---|---|
| 輸入 | YouTube 連結 / 音檔（mp3、wav、m4a、flac） |
| `bass.musicxml` | 4 弦貝斯（E-A-D-G），**帶節奏符桿的 TAB**，含調號、拍號、速度，MuseScore 直接開啟 |
| `bass.mid` | 貝斯 MIDI，帶 tempo map，拉進 DAW 能和原曲對齊 |
| `no_bass.wav` | 去掉貝斯的伴奏，跟著練習用 |
| `bass.wav` | 分離出的貝斯音軌，聽不清楚時對照用 |
| 介面 | 本機網頁：貼上連結或拖入音檔 → 顯示進度 → 下載 4 個檔案 |

### 階段二：管弦樂多聲部（階段一完成後才開始）

| 項目 | 內容 |
|---|---|
| 聲部分區 | 高／中／低是最少的分區，每個音域盡量抓出所有獨立線條 |
| 線條判斷 | 依旋律連續性串接，不分樂器；齊奏和八度重疊只保留一條 |
| 線條類型 | 每條線標成「旋律線」或「和聲長音」，UI 可以選擇隱藏和聲長音 |
| `score.musicxml` | 分成 High／Mid／Low 三組，每條線一行譜，依音域自動選譜號，名稱用 `High 1`、`Mid 2` 這類通用名稱 |
| `score.mid` | 每條線一個 MIDI 軌，帶軌名和 tempo map |
| 可調參數 | 每個音域最多幾條線、線條最短長度、音域分界、是否保留和聲長音 |
| 已知限制 | 線條越多越不準，中音域最容易出錯；rubato 段落的小節線可能不準，但 MIDI 時間點要對齊原曲 |

## 技術選型

| 功能 | 工具 |
|---|---|
| 環境 | `uv` + Python 3.11 |
| YouTube 下載 | `yt-dlp` |
| ffmpeg | `imageio-ffmpeg` |
| 分軌 | Demucs `htdemucs_ft`（MPS，沒有就用 CPU） |
| 貝斯音符偵測 | Basic Pitch（Apache 2.0）+ 4 弦音域限制（E1～G4）+ pYIN 八度校正 |
| 節拍／小節線 | beat_this（MIT），librosa 當備援 |
| 調號 | music21 調性分析 |
| 多音轉譜（階段二） | YourMT3+ 和 Basic Pitch 比較後擇一 |
| MusicXML | 用 ElementTree 自己寫（music21 對 TAB 支援不完整） |
| MIDI | `pretty_midi` |
| 網頁介面 | FastAPI + 單一 HTML 頁面，背景工作處理並回報進度 |

注意：Melodia 演算法的授權不允許商業使用，不要用（Basic Pitch 的 `melodia_trick` 參數也一律關閉）。

## 架構

共用資料格式：`NoteEvent(pitch, onset, offset, velocity)` → `Line`（一條單聲部線）→ `Score`（多條線 + 節拍格 + 調號 + 拍號）。

```
music-to-score/
  pyproject.toml
  start.command                ← 雙擊啟動，自動開瀏覽器
  src/scoreapp/
    model.py                   NoteEvent / Line / Score
    fetch.py                   YouTube 或音檔 → 44.1kHz wav
    separate.py                Demucs 包裝 → bass / no_bass
    transcribe/
      mono.py                  單聲部：Basic Pitch + 音域限制 + 八度校正
      poly.py                  多音轉譜（階段二）
    rhythm/
      beats.py                 節拍、downbeat、拍號、tempo map
      quantize.py              對齊節拍格（16 分音符 + 三連音）、清除過短音符、合併重複音
    voices/                    （階段二）
      separate_lines.py        把多音的音符串成多條連續線條
      classify.py              分辨旋律線和和聲長音，依音域分組
    export/
      tab.py                   TAB 把位指派（動態規劃，讓手的移動最少）
      musicxml.py              TAB 譜表 / 多行總譜
      midi.py
    pipelines/
      bass.py
      orchestral.py
    web/
      app.py
      static/index.html
  tests/
  output/<歌名>/               （加進 .gitignore）
```

## 實作步驟

### 階段一

1. **M0 環境**：`pyproject.toml`（uv）、`.gitignore`、README（給不熟 Python 的人看的安裝與啟動說明）、`start.command`。
2. **M1 取得音訊與分軌**：`fetch.py` + `separate.py`，產生 `bass.wav` 和 `no_bass.wav`。
3. **M2 貝斯音符偵測**：`transcribe/mono.py`，限制音域、做八度校正、過濾大鼓漏進來的假音符。
4. **M3 節奏**：`beats.py` + `quantize.py`。
5. **M4 輸出**：
   - `tab.py`：指派把位。
   - `musicxml.py`：輸出 4 線 TAB 譜表。`<staff-details>` 設 EADG 調音，每個音帶 `<technical><string>` 和 `<fret>`，樂器設成 Electric Bass。要確保 MuseScore 匯入後顯示**有節奏符桿的 TAB**。
   - `midi.py`：輸出 MIDI。
6. **M5 網頁介面**：貼連結或上傳檔案 → 進度條 → 下載 4 個檔案。
7. **M6 調參**：之後由使用者在本機用熟悉的歌測試，再一起調整。

### 階段二（之後）

M7 多音轉譜比較 → M8 線條分離（依時間逐步分配音符，成本考慮音程跳躍、時間間隔、音量連續性，線條數量動態增減，偵測齊奏和八度重疊並合併）→ M9 分類與分組 → M10 多行 MusicXML、多軌 MIDI，網頁加上模式切換和參數設定 → M11 調參。

## 驗證

- `uv run pytest`：量化、TAB 把位（弦和格在範圍內、移動合理）、MusicXML 結構，以及用合成音檔跑的端對端測試。
- 本機（使用者執行）：用 YouTube 連結跑完整流程，在 MuseScore 4 開 `bass.musicxml`，確認是 4 線 TAB、有符桿、播放音高正確；`bass.mid` 拉進 Logic 和 `bass.wav` 對齊。

## 實作備註（階段一）

和上面計畫不同、或計畫沒寫到的細節：

- **Basic Pitch** 用套件內附的 ONNX 模型 + onnxruntime（Mac 和 Linux 一致），不裝 TensorFlow / CoreML。`melodia_trick` 預設關閉。
- **Demucs** `htdemucs_ft` 是 4 個子模型的組合，只跑負責貝斯的那一個，`no_bass = 原曲 − 貝斯`，速度快 4 倍。MPS 失敗時自動退回 CPU。
- **貝斯後處理**（`transcribe/mono.py`）：泛音假音符刪除、pYIN + 頻譜八度校正、短視窗頻譜通量修正 onset、碎片合併（用各泛音能量是否重新上升判斷是否重新撥弦）、補回漏掉的音、大鼓漏音過濾。
- **MIDI 寫檔**用 `mido`（pretty_midi 的底層），因為 pretty_midi 不能直接寫每拍一個 tempo 事件；測試用 pretty_midi 讀回驗證對齊。
  音符時間預設用原始演奏時間（和音訊完全對齊），網頁可勾選改用量化時間。
- **節拍格**：0 秒到第一拍之間補成弱起小節，所以 MIDI 第 0 tick = 音訊第 0 秒，DAW 裡的小節線也會落在強拍上。
- **調號**偵測放在 `key.py`（music21）。
- **TAB 把位**：DP 狀態是（弦, 格, 食指把位），一指一格範圍內不用換把；休止符或空弦期間換把比較便宜。
- **MusicXML**：音高寫實際發聲音高、不加移調；MuseScore 4 匯入後是 4 線 TAB（tab Common 樣式：符桿和符槓畫在 TAB 下方），樂器自動對應 Electric Bass。測試用 MusicXML 4.0 schema 驗證。

## 注意事項

- YouTube 下載僅供個人練習與研究使用。
- PyTorch 加模型大約要 2～3GB 空間。
