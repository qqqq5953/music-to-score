# music-to-score

把一首歌（YouTube 連結或音檔）變成 **4 弦貝斯 TAB 譜** 和 **MIDI**，並分出貝斯音軌和去貝斯伴奏，方便練習與編曲。

每首歌會產生：

| 檔案 | 用途 |
|---|---|
| `bass.musicxml` | 4 弦貝斯（E-A-D-G）**帶節奏符桿的 TAB**，用 MuseScore 4 直接開 |
| `bass.mid` | 貝斯 MIDI，帶 tempo map，拉進 Logic / Cubase 能和原曲對齊 |
| `no_bass.wav` | 去掉貝斯的伴奏，跟著練習用 |
| `bass.wav` | 分離出來的貝斯音軌，聽不清楚時對照用 |
| `mix.wav` | 原曲（轉好的 wav），放進 DAW 和 MIDI 對齊用 |

> YouTube 下載功能僅供個人練習與研究使用。

---

## 第一次使用（Mac）

你**不需要**自己裝 Python，也不需要 Homebrew。所有東西都會裝在這個資料夾和你的使用者資料夾裡，不會動到系統。

### 1. 下載這個專案

在 GitHub 頁面按綠色的 **Code → Download ZIP**，解壓縮後把資料夾放在你喜歡的位置（例如「文件」裡）。

### 2. 雙擊 `start.command`

- 第一次雙擊時，macOS 可能會說「無法打開，因為它來自未識別的開發者」。這時請：
  **在 `start.command` 上按右鍵（或 control + 點一下）→ 選「打開」→ 再按一次「打開」**。
  之後就可以直接雙擊了。
  （如果沒有「打開」可以按，就到「系統設定 → 隱私權與安全性」，往下找到 `start.command` 按「強制打開」。）
- 會跳出一個終端機視窗，**不用打任何字**，它會自動：
  1. 安裝 `uv`（一個管理 Python 的小工具，裝在 `~/.local/bin`）
  2. 下載 Python 3.11 和需要的套件（約 2～3GB，第一次要等幾分鐘）
  3. 打開瀏覽器，進入 <http://127.0.0.1:8765>
- 第一次轉一首歌時，還會下載分軌模型（Demucs，約 300MB），所以第一首會比較慢。

### 3. 使用

1. 在網頁上貼上 YouTube 連結，或把音檔（mp3、wav、m4a、flac）拖進去。
2. 按「開始轉譜」，等進度條跑完（一首 4 分鐘的歌大約 1～3 分鐘）。
3. 下載檔案。所有檔案也會存在專案資料夾的 `output/<歌名>/` 裡。

要結束時，直接關掉終端機視窗就可以了。

### 如果雙擊沒反應 / 顯示「沒有權限」

ZIP 下載有時會弄掉檔案的執行權限。打開「終端機」（在「應用程式 → 工具程式」裡），輸入 `chmod +x `（最後有一個空格），把 `start.command` 拖進終端機視窗，按 Enter。之後再雙擊就好。

### 如果自動安裝 uv 失敗

打開「終端機」，貼上這行後按 Enter：

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

裝完關掉終端機，再雙擊 `start.command`。

---

## 在 MuseScore 和 Logic 裡使用

**MuseScore 4**：直接打開 `bass.musicxml`。會看到 4 線 TAB，數字下方有符桿和符尾。
想同時看五線譜：在譜表上按右鍵 →「譜表/聲部屬性」→ 換成標準譜表，或在「樂器」面板加一個連結的五線譜。

**Logic Pro**：
1. 先把 `mix.wav`（或 `no_bass.wav`）拉到第 1 小節的最開頭。
2. 「檔案 → 輸入 → MIDI 檔案」選 `bass.mid`，出現詢問時選擇**匯入速度（tempo）**。
3. MIDI 的小節線會跟著原曲的速度變化，音符和音訊對齊。

**Cubase**：把 `bass.mid` 拖進專案，出現「匯入速度軌」時選「是」；音訊同樣放在專案起點。

---

## 移除

1. 刪掉整個專案資料夾（模型、套件、輸出都在裡面）。
2. 如果也想移除 uv 和它下載的 Python，打開終端機貼上：

```sh
rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv
```

---

## 給進階使用者

```sh
uv run scoreapp bass "https://www.youtube.com/watch?v=..."   # 命令列轉一首歌
uv run scoreapp bass song.mp3 --out ~/Desktop/scores
uv run scoreapp serve --open                                 # 啟動網頁
uv run pytest                                                # 跑測試
```

專案計畫和架構請看 [`docs/PLAN.md`](docs/PLAN.md)。
