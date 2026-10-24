from __future__ import annotations

import argparse
from pathlib import Path

from src.ocr_text import save_ocr_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Vietnamese, Traditional Chinese, English, and French text."
    )
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("output"),
        help="Directory for UTF-8 .txt and .json files",
    )
    args = parser.parse_args()

    text_path = args.output_dir / f"{args.input.stem}.txt"
    json_path = args.output_dir / f"{args.input.stem}.json"
    lines = save_ocr_files(args.input, text_path, json_path)

    print(f"Text: {text_path}")
    print(f"JSON: {json_path}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
