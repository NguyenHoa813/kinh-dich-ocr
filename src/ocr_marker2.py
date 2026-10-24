from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

try:
    from src.image_loader import load_image
except ModuleNotFoundError:
    from image_loader import load_image


LANGUAGE_CONFIG = {
    "vie": ("Vietnamese", (220, 40, 40)),
    "fra": ("French", (220, 130, 20)),
    "chi_tra": ("Traditional Chinese", (40, 90, 210)),
    "eng": ("English", (30, 150, 70)),
}

SCAN_ORDER = ("vie", "fra", "chi_tra", "eng")

VIETNAMESE_MARKERS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
FRENCH_MARKERS = set("éèëêàâçîïôùûüÿœæÉÈËÊÀÂÇÎÏÔÙÛÜŸŒÆ")
VIETNAMESE_WORDS = {
    "bát", "biến", "cho", "có", "cùng", "đạo", "đều", "đức", "giải",
    "hào", "hạ", "hanh", "hạnh", "khai", "kiên", "kiền", "kinh", "là",
    "lợi", "mạnh", "mông", "nghĩa", "nguyên", "sáu", "sinh", "thượng",
    "trinh", "trung", "tự", "tu", "vua", "vận", "vậy", "về", "với",
    "am", "che", "tha",
}
FRENCH_WORDS = {
    "avec", "avance", "bien", "ciel", "dans", "français", "homme", "le",
    "les", "pour", "force", "vigueur", "noble", "jamais", "cesser",
}


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
    normalized = re.sub(r"[^\wÀ-ỹ]", "", text.lower())

    if language == "vie" and (has_vietnamese or normalized in VIETNAMESE_WORDS):
        return 20.0
    if language == "fra" and (
        has_french or normalized in FRENCH_WORDS
    ) and not has_vietnamese:
        return 8.0
    if language == "chi_tra" and has_cjk:
        return 25.0
    if language == "eng" and not has_cjk and not has_vietnamese:
        return 2.0
    return 0.0


def _has_language_evidence(text: str, language: str) -> bool:
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in text)
    has_vietnamese = any(char in VIETNAMESE_MARKERS for char in text)
    has_french = any(char in FRENCH_MARKERS for char in text)
    normalized = re.sub(r"[^\wÀ-ỹ]", "", text.lower())

    if language == "vie":
        return has_vietnamese or normalized in VIETNAMESE_WORDS
    if language == "fra":
        return (
            (has_french or normalized in FRENCH_WORDS)
            and not has_vietnamese
            and not has_cjk
        )
    if language == "chi_tra":
        return has_cjk
    return (
        any(char.isalpha() for char in text)
        and not has_cjk
        and not has_vietnamese
        and normalized not in VIETNAMESE_WORDS
        and normalized not in FRENCH_WORDS
    )


def _box_is_claimed(
    painted_mask: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    height, width = painted_mask.shape
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(width, right)
    y1 = min(height, bottom)
    if x0 >= x1 or y0 >= y1:
        return True
    return bool(painted_mask[y0:y1, x0:x1].any())


def _paint_box(
    painted_mask: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    height, width = painted_mask.shape
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(width, right)
    y1 = min(height, bottom)
    if x0 < x1 and y0 < y1:
        painted_mask[y0:y1, x0:x1] = True


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
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if (
            not text
            or confidence < 25
            or width <= 0
            or height <= 0
            or not _has_language_evidence(text, language)
        ):
            continue

        regions.append(TextRegion(left, top, width, height, text, language, confidence))
    return regions


def detect_text_regions(image: Image.Image) -> list[TextRegion]:
    configure_tesseract()
    painted_regions: list[TextRegion] = []
    painted_mask = np.zeros((image.height, image.width), dtype=bool)

    for language in SCAN_ORDER:
        candidates = [
            (
                region.confidence + _script_bonus(region.text, language),
                region,
            )
            for region in _extract_regions(image, language)
        ]

        for _, region in sorted(candidates, key=lambda item: item[0], reverse=True):
            if _box_is_claimed(
                painted_mask,
                region.left,
                region.top,
                region.right,
                region.bottom,
            ):
                continue

            painted_regions.append(region)
            _paint_box(
                painted_mask,
                region.left,
                region.top,
                region.right,
                region.bottom,
            )

    return sorted(painted_regions, key=lambda region: (region.top, region.left))


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
    painted_regions: list[TextRegion] = []
    painted_mask = np.zeros((image.height, image.width), dtype=bool)

    for region in regions:
        if _box_is_claimed(
            painted_mask,
            region.left,
            region.top,
            region.right,
            region.bottom,
        ):
            continue

        _, color = LANGUAGE_CONFIG[region.language]
        draw.rectangle(
            (region.left, region.top, region.right, region.bottom),
            outline=color,
            width=3,
        )

        _paint_box(
            painted_mask,
            region.left,
            region.top,
            region.right,
            region.bottom,
        )
        painted_regions.append(region)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    marked.save(output)
    return painted_regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark OCR regions in order: Vietnamese, French, Chinese, English."
    )
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Marked output image path")
    args = parser.parse_args()

    regions = mark_text_regions(args.input, args.output)
    print(f"Marked image: {args.output}")
    print(f"Painted regions: {len(regions)}")


if __name__ == "__main__":
    main()
