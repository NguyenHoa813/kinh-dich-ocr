from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

from src.image_loader import load_image


LANGUAGE_CONFIG = {
    "vie": ("Vietnamese", (220, 40, 40)),
    "chi_tra": ("Traditional Chinese", (40, 90, 210)),
    "eng": ("English", (30, 150, 70)),
    "fra": ("French", (220, 130, 20)),
}

VIETNAMESE_MARKERS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạấầẩẫậắằẳẵặ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶ"
    "ÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰ"
    "ÝỲỶỸỴ"
)
VIETNAMESE_UNIQUE_MARKERS = set("ăâđơưĂÂĐƠƯ")
FRENCH_MARKERS = set(
    "éèëê"
    "àâ"
    "ç"
    "îï"
    "ô"
    "ùûü"
    "ÿ"
    "œæ"
    "ÉÈËÊ"
    "ÀÂ"
    "Ç"
    "ÎÏ"
    "Ô"
    "ÙÛÜ"
    "Ÿ"
    "ŒÆ"
)

@dataclass(frozen=True)
class TextRegion:
    left: int
    top: int
    width: int
    height: int
    text: str
    language: str
    confidence: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def configure_tesseract() -> None:
    configured = os.environ.get("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return

    if shutil.which("tesseract"):
        return

    for candidate in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ):
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return


def _script_bonus(text: str, language: str) -> float:
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in text)
    has_vietnamese = any(char in VIETNAMESE_MARKERS for char in text)
    has_french = any(char in FRENCH_MARKERS for char in text)

    if language == "chi_tra" and has_cjk:
        return 25.0
    if language == "vie" and has_vietnamese:
        return 20.0
    if language == "fra" and has_french and not has_vietnamese:
        return 8.0
    if language == "eng" and not has_cjk and not has_vietnamese:
        return 2.0
    return 0.0


def _has_language_evidence(text: str, language: str) -> bool:
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in text)
    has_vietnamese = any(char in VIETNAMESE_MARKERS for char in text)
    has_french = any(char in FRENCH_MARKERS for char in text)
    if language == "chi_tra":
        return has_cjk
    if language == "vie":
        return has_vietnamese
    if language == "fra":
        return has_french and not has_vietnamese and not has_cjk
    return any(char.isalpha() for char in text) and not has_cjk and not has_vietnamese and not has_french


def _overlap(first: TextRegion, second: TextRegion) -> float:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0

    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / min(first_area, second_area)


def _extract_regions(image: Image.Image, language: str) -> list[TextRegion]:
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config="--psm 11",
        output_type=Output.DICT,
    )
    regions = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if not text or confidence < 25 or not _has_language_evidence(text, language):
            continue

        try:
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        regions.append(
            TextRegion(left, top, width, height, text, language, confidence)
        )
    return regions


def detect_text_regions(image: Image.Image) -> list[TextRegion]:
    configure_tesseract()
    candidates = []
    for language in LANGUAGE_CONFIG:
        for region in _extract_regions(image, language):
            candidates.append(
                (region.confidence + _script_bonus(region.text, language), region)
            )

    has_distinct_non_english = any(
        region.language == "chi_tra"
        or any(char in VIETNAMESE_UNIQUE_MARKERS for char in region.text)
        for _, region in candidates
    )
    if not has_distinct_non_english:
        candidates = [
            (score, region) for score, region in candidates if region.language == "eng"
        ]

    selected = []
    for _, region in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_overlap(region, existing) >= 0.4 for existing in selected):
            continue
        selected.append(region)

    return sorted(selected, key=lambda region: (region.top, region.left))


def _font() -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), 16)
    return ImageFont.load_default()


def mark_text_regions(input_path: str | Path, output_path: str | Path) -> list[TextRegion]:
    image = load_image(input_path)
    regions = detect_text_regions(image)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = _font()

    for region in regions:
        _, color = LANGUAGE_CONFIG[region.language]
        draw.rectangle(
            (region.left, region.top, region.right, region.bottom),
            outline=color,
            width=3,
        )
        label = LANGUAGE_CONFIG[region.language][0]
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 4
        label_top = max(0, region.top - label_height)
        draw.rectangle(
            (region.left, label_top, region.left + label_width, region.top),
            fill=color,
        )
        draw.text(
            (region.left + 4, label_top + 2),
            label,
            fill="white",
            font=font,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    marked.save(output)
    return regions
