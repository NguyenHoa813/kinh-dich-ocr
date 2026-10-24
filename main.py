from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.hex_image import save_hexagram_image
from src.ocr_marker2 import mark_text_regions
from src.ocr_text import save_ocr_files


def _add_input_argument(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("input", type=Path, help="Input image path")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run hexagram cropping, OCR marking, text extraction, or both."
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

	hex_split = commands.add_parser("hex", help="Crop the hexagram from an image")
	_add_input_argument(hex_split)
	hex_split.add_argument(
		"output",
		type=Path,
		nargs="?",
		help="Cropped image output path (default: output/hex/<input>_hex.png)",
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


def _run_hex_split(
	input_path: Path,
	output_path: Path,
	*,
	required: bool = False,
) -> None:
	gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
	if gray is None:
		raise SystemExit(f"Không đọc được ảnh: {input_path}")

	result = save_hexagram_image(gray, None, output_path)
	if result is None:
		message = f"Không tìm thấy quẻ trong ảnh: {input_path}"
		if required:
			raise SystemExit(message)
		print(f"{message}; tiếp tục xử lý ảnh.", file=sys.stderr)
		return

	print(f"Đã lưu ảnh quẻ: {result}")


def main() -> None:
	parser = _build_parser()
	arguments = sys.argv[1:]

	# Keep the previous form: main.py input-image marked-image.
	commands = {"marker", "text", "all", "hex"}
	if arguments and arguments[0] not in commands and not arguments[0].startswith("-"):
		arguments = ["marker", *arguments]

	args = parser.parse_args(arguments)
	if args.command == "marker":
		hex_output = args.output.parent / "hex" / f"{args.input.stem}_hex.png"
		_run_hex_split(args.input, hex_output)
		_run_marker(args.input, args.output)
	elif args.command == "text":
		hex_output = args.output_dir / "hex" / f"{args.input.stem}_hex.png"
		_run_hex_split(args.input, hex_output)
		_run_text(args.input, args.output_dir)
	elif args.command == "all":
		hex_output = args.output_dir / "hex" / f"{args.input.stem}_hex.png"
		_run_hex_split(args.input, hex_output)
		_run_marker(args.input, args.output_dir / f"{args.input.stem}_marked.png")
		_run_text(args.input, args.output_dir / "ocr")
	elif args.command == "hex":
		output = args.output or Path("output/hex") / f"{args.input.stem}_hex.png"
		_run_hex_split(args.input, output, required=True)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
