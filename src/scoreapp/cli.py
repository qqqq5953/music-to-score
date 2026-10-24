"""命令列入口：`scoreapp serve`（網頁介面）或 `scoreapp bass <連結或音檔>`。"""

from __future__ import annotations

import argparse
import logging
import sys

from scoreapp.paths import OUTPUT_DIR, setup_cache_env


def main(argv: list[str] | None = None) -> int:
    setup_cache_env()
    parser = argparse.ArgumentParser(prog="scoreapp")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="啟動本機網頁介面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="啟動後自動開瀏覽器")

    bass = sub.add_parser("bass", help="把一首歌轉成貝斯 TAB / MIDI")
    bass.add_argument("source", help="YouTube 連結或音檔路徑")
    bass.add_argument("--out", default=str(OUTPUT_DIR), help="輸出資料夾（預設 output/）")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.command == "serve":
        from scoreapp.web.app import serve as run_server

        run_server(host=args.host, port=args.port, open_browser=args.open)
        return 0

    if args.command == "bass":
        from pathlib import Path

        from scoreapp.pipelines.bass import run_bass_pipeline

        def report(fraction: float, message: str) -> None:
            print(f"[{fraction * 100:5.1f}%] {message}", flush=True)

        result = run_bass_pipeline(args.source, Path(args.out), progress=report)
        print(f"\n完成！檔案在：{result.out_dir}")
        for name, path in result.files.items():
            print(f"  {name}: {path.name}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
