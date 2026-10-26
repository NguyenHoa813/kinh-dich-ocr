from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2

from src.hex_image import save_hexagram_image
from src.ocr_marker2 import mark_text_regions
from src.ocr_export import export_ocr_document
from src.ocr_text import save_ocr_files
from src.page_crop import save_page_crop


def _add_input_argument(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("input", type=Path, help="Input image path")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Crop book pages, hexagrams, mark OCR regions, or extract text."
	)
	commands = parser.add_subparsers(dest="command")

	marker = commands.add_parser("marker", help="Draw language boxes on the image")
	_add_input_argument(marker)
	marker.add_argument("output", type=Path, help="Marked image output path")

	text = commands.add_parser(
		"text",
		help="Extract OCR text and export TXT, HTML, and JSON",
	)
	_add_input_argument(text)
	text.add_argument(
		"output_dir",
		type=Path,
		nargs="?",
		default=Path("output/ocr"),
		help="Output directory (default: output/ocr)",
	)

	all_ocr = commands.add_parser(
		"all",
		help="Crop a page, mark OCR regions, extract text, and export documents",
	)
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

	page_crop = commands.add_parser("page", help="Detect, crop, and straighten a book page")
	_add_input_argument(page_crop)
	page_crop.add_argument(
		"output",
		type=Path,
		nargs="?",
		help="Output path (default: output/page/<input>_page.png)",
	)
	return parser


def _run_marker(input_path: Path, output_path: Path) -> None:
	regions = mark_text_regions(input_path, output_path)
	print(f"Marked image: {output_path}")
	print(f"Detected regions: {len(regions)}")


def _run_text(
	input_path: Path,
	output_dir: Path,
	hex_image_path: Path | None = None,
) -> None:
	text_path = output_dir / f"{input_path.stem}.txt"
	ocr_json_path = output_dir / f"{input_path.stem}_ocr.json"
	lines = save_ocr_files(input_path, text_path, ocr_json_path)
	html_path = output_dir / f"{input_path.stem}.html"
	json_path = output_dir / f"{input_path.stem}.json"
	stylesheet_source = Path(__file__).parent / "output" / "style.css"
	stylesheet_path = output_dir.parent / "style.css"
	stylesheet_path.parent.mkdir(parents=True, exist_ok=True)
	if stylesheet_source.resolve() != stylesheet_path.resolve():
		shutil.copyfile(stylesheet_source, stylesheet_path)
	stylesheet = os.path.relpath(stylesheet_path, start=output_dir).replace("\\", "/")
	export_ocr_document(
		lines,
		html_path,
		json_path,
		document_id=input_path.stem,
		hexagram_image=(
			hex_image_path
			if hex_image_path is not None and hex_image_path.is_file()
			else None
		),
		stylesheet=stylesheet,
	)
	print(f"Text: {text_path}")
	print(f"OCR JSON: {ocr_json_path}")
	print(f"Lines: {len(lines)}")
	print(f"HTML: {html_path}")
	print(f"JSON: {json_path}")


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


def _run_page_crop(input_path: Path, output_path: Path) -> None:
	result = save_page_crop(input_path, output_path)
	if result is None:
		raise SystemExit(f"Không tìm thấy trang sách trong ảnh: {input_path}")
	print(f"Đã lưu trang sách đã cắt và chỉnh thẳng: {result}")


def _run_all(input_path: Path, output_dir: Path) -> None:
	processed_path = output_dir / f"{input_path.stem}_page.png"
	_run_page_crop(input_path, processed_path)

	hex_output = output_dir / "hex" / f"{input_path.stem}_hex.png"
	_run_hex_split(processed_path, hex_output)
	_run_marker(processed_path, output_dir / f"{input_path.stem}_marked.png")
	_run_text(processed_path, output_dir / "ocr", hex_output)


def main() -> None:
	for stream in (sys.stdout, sys.stderr):
		reconfigure = getattr(stream, "reconfigure", None)
		if reconfigure is not None:
			reconfigure(encoding="utf-8")

	parser = _build_parser()
	arguments = sys.argv[1:]

	# Keep the previous form: main.py input-image marked-image.
	commands = {"marker", "text", "all", "hex", "page"}
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
		_run_text(args.input, args.output_dir, hex_output)
	elif args.command == "all":
		_run_all(args.input, args.output_dir)
	elif args.command == "hex":
		output = args.output or Path("output/hex") / f"{args.input.stem}_hex.png"
		_run_hex_split(args.input, output, required=True)
	elif args.command == "page":
		output = args.output or Path("output/page") / f"{args.input.stem}_page.png"
		_run_page_crop(args.input, output)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
