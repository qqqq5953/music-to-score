#!/bin/bash
# 雙擊這個檔案就會啟動 music-to-score，並自動打開瀏覽器。
# 第一次啟動會下載 Python 和套件（約 2～3GB），需要幾分鐘，之後就很快。

cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "找不到 uv，現在自動安裝（只會裝在你的使用者資料夾，不需要 Homebrew）……"
  curl -LsSf https://astral.sh/uv/install.sh | sh || {
    echo "uv 安裝失敗，請看 README 的「安裝 uv」一節。"
    read -r -p "按 Enter 關閉視窗……"
    exit 1
  }
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "檢查並安裝需要的套件（第一次會比較久）……"
if ! uv sync --frozen; then
  echo "套件安裝失敗，請把上面的訊息截圖回報。"
  read -r -p "按 Enter 關閉視窗……"
  exit 1
fi

echo ""
echo "啟動中……瀏覽器會自動打開 http://127.0.0.1:8765"
echo "要結束時，關掉這個終端機視窗就可以了。"
echo ""
uv run --frozen scoreapp serve --open
read -r -p "伺服器已停止，按 Enter 關閉視窗……"
