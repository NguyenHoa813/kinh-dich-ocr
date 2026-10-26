from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from src.image_loader import load_image
from src.ocr_marker import configure_tesseract

OCR_LANGUAGES = ("vie", "chi_tra")
MULTILINGUAL_LANGUAGES = "+".join(OCR_LANGUAGES)
VIETNAMESE_CONFIG = "--oem 1 --psm 4"
HAN_CONFIG = "--oem 1 --psm 6"
HAN_FALLBACK_CONFIG = "--oem 1 --psm 11"
MIN_WORD_CONFIDENCE = 50.0


@dataclass(frozen=True)
class OcrLine:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _normalize_heading(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^A-Z]", "", without_marks.upper())


def _read_language_lines(
    image: Image.Image,
    language: str,
    *,
    config: str,
    top_offset: int = 0,
) -> list[OcrLine]:
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DICT,
    )
    grouped: dict[tuple[int, int, int], list[tuple[int, str, int, int, int, int, float]]] = {}

    for index, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        if not text:
            continue
        if language == "chi_tra" and not _has_cjk(text):
            continue
        if language == "vie" and _has_cjk(text):
            continue

        try:
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            confidence = float(data["conf"][index])
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if (
            width <= 0
            or height <= 0
            or (
                language == "vie"
                and confidence < MIN_WORD_CONFIDENCE
            )
        ):
            continue

        grouped.setdefault(key, []).append(
            (left, text, top, width, height, index, confidence)
        )

    lines: list[OcrLine] = []
    for words in grouped.values():
        words.sort(key=lambda item: item[5])
        left = min(word[0] for word in words)
        top = min(word[2] for word in words) + top_offset
        right = max(word[0] + word[3] for word in words)
        bottom = max(word[2] + word[4] for word in words) + top_offset
        confidence = sum(word[6] for word in words) / len(words)
        lines.append(
            OcrLine(
                text=" ".join(word[1] for word in words),
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
                confidence=confidence,
            )
        )

    return lines


def _han_section_bounds(
    lines: list[OcrLine],
    image_height: int,
) -> tuple[int, int] | None:
    start_heading: OcrLine | None = None
    sound_heading: OcrLine | None = None
    explanation_heading: OcrLine | None = None
    for line in lines:
        heading = _normalize_heading(line.text)
        if heading == "KINHVAN" and start_heading is None:
            start_heading = line
        elif (
            sound_heading is None
            and re.match(
                r"^\s*am\s*[:：]",
                unicodedata.normalize("NFD", line.text)
                .encode("ascii", "ignore")
                .decode("ascii"),
                re.IGNORECASE,
            )
        ):
            sound_heading = line
        elif heading == "GIAITHICH" and explanation_heading is None:
            explanation_heading = line

    end_heading = sound_heading or explanation_heading
    if start_heading is None or end_heading is None:
        return None

    top = start_heading.bottom + 2
    bottom = min(end_heading.top - 2, image_height)
    if top >= bottom:
        return None
    return top, bottom


def _read_lines(image: Image.Image) -> list[OcrLine]:
    vietnamese_lines = _read_language_lines(
        image,
        "vie",
        config=VIETNAMESE_CONFIG,
    )

    bounds = _han_section_bounds(vietnamese_lines, image.height)
    if bounds is not None:
        top, bottom = bounds
        han_crop = image.crop((0, top, image.width, bottom))
        han_lines = _read_language_lines(
            han_crop,
            "chi_tra",
            config=HAN_CONFIG,
            top_offset=top,
        )
    else:
        han_lines = _read_language_lines(
            image,
            "chi_tra",
            config=HAN_FALLBACK_CONFIG,
        )

    return sorted(
        [*vietnamese_lines, *han_lines],
        key=lambda line: (line.top, line.left),
    )


def extract_text(input_path: str | Path) -> list[OcrLine]:
    configure_tesseract()
    image = load_image(input_path)
    return _read_lines(image)


def save_ocr_files(
    input_path: str | Path,
    text_output: str | Path,
    json_output: str | Path | None = None,
) -> list[OcrLine]:
    lines = extract_text(input_path)
    text_path = Path(text_output)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        "\n".join(line.text for line in lines) + "\n",
        encoding="utf-8",
    )

    if json_output is not None:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "input": str(input_path),
                    "languages": MULTILINGUAL_LANGUAGES,
                    "lines": [asdict(line) for line in lines],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return lines
