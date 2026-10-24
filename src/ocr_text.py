from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from src.image_loader import load_image
from src.ocr_marker import configure_tesseract

MULTILINGUAL_LANGUAGES = "vie+chi_tra+eng+fra"


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


def _read_lines(image: Image.Image) -> list[OcrLine]:
    data = pytesseract.image_to_data(
        image,
        lang=MULTILINGUAL_LANGUAGES,
        config="--psm 11",
        output_type=Output.DICT,
    )
    grouped: dict[tuple[int, int, int], list[tuple[int, str, int, int, int, int, float]]] = {}

    for index, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        if not text:
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

        if width <= 0 or height <= 0:
            continue

        grouped.setdefault(key, []).append(
            (left, text, top, width, height, index, confidence)
        )

    lines = []
    for words in grouped.values():
        words.sort(key=lambda item: item[5])
        left = min(word[0] for word in words)
        top = min(word[2] for word in words)
        right = max(word[0] + word[3] for word in words)
        bottom = max(word[2] + word[4] for word in words)
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

    return sorted(lines, key=lambda line: (line.top, line.left))


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
