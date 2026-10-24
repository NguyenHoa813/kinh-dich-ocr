from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.orc_marker2 import mark_text_regions
from src.ocr_text import save_ocr_files


def _add_input_argument(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("input", type=Path, help="Input image path")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run OCR marking, text extraction, or both."
	)
	commands = parser.add_subparsers(dest="command")

	marker = commands.add_parser("marker", help="Draw language boxes on the image")
	_add_input_argument(marker)
	marker.add_argument("output", type=Path, help="Marked image output path")

	text = commands.add_parser("text", help="Extract OCR text to TXT and JSON")
	_add_input_argument(text)
	text.add_argument(
		"output_dir",
		type=Path,
		nargs="?",
		default=Path("output/ocr"),
		help="Output directory (default: output/ocr)",
	)

	all_ocr = commands.add_parser("all", help="Run marker and text extraction")
	_add_input_argument(all_ocr)
	all_ocr.add_argument(
		"output_dir",
		type=Path,
		nargs="?",
		default=Path("output"),
		help="Output directory (default: output)",
	)
	return parser


def _run_marker(input_path: Path, output_path: Path) -> None:
	regions = mark_text_regions(input_path, output_path)
	print(f"Marked image: {output_path}")
	print(f"Detected regions: {len(regions)}")


def _run_text(input_path: Path, output_dir: Path) -> None:
	text_path = output_dir / f"{input_path.stem}.txt"
	json_path = output_dir / f"{input_path.stem}.json"
	lines = save_ocr_files(input_path, text_path, json_path)
	print(f"Text: {text_path}")
	print(f"JSON: {json_path}")
	print(f"Lines: {len(lines)}")


def main() -> None:
	parser = _build_parser()
	arguments = sys.argv[1:]

	# Keep the previous form: main.py input-image marked-image.
	commands = {"marker", "text", "all"}
	if arguments and arguments[0] not in commands and not arguments[0].startswith("-"):
		arguments = ["marker", *arguments]

	args = parser.parse_args(arguments)
	if args.command == "marker":
		_run_marker(args.input, args.output)
	elif args.command == "text":
		_run_text(args.input, args.output_dir)
	elif args.command == "all":
		_run_marker(args.input, args.output_dir / f"{args.input.stem}_marked.png")
		_run_text(args.input, args.output_dir / "ocr")
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
