
from __future__ import annotations

import os
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

VIETNAMESE_STRONG_MARKERS = set(
    "ăâđơưĂÂĐƠƯ"
    "ấầẩẫậắằẳẵặ"
    "ếềểễệ"
    "ốồổỗộớờởỡợ"
    "ứừửữự"
    "ẤẦẨẪẬẮẰẲẴẶ"
    "ẾỀỂỄỆ"
    "ỐỒỔỖỘỚỜỞỠỢ"
    "ỨỪỬỮỰ"
)

FRENCH_MARKERS = set(
    "éèëêàâçîïôùûüÿœæ"
    "ÉÈËÊÀÂÇÎÏÔÙÛÜŸŒÆ"
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

    candidates = (
        Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        ),
        Path(
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ),
    )

    for candidate in candidates:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return


def _has_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u9fff"
        for char in text
    )


def _has_vietnamese(text: str) -> bool:
    return any(
        char in VIETNAMESE_MARKERS
        for char in text
    )


def _has_strong_vietnamese(text: str) -> bool:
    return any(
        char in VIETNAMESE_STRONG_MARKERS
        for char in text
    )


def _has_language_evidence(
    text: str,
    language: str,
) -> bool:
    has_cjk = _has_cjk(text)
    has_vietnamese = _has_vietnamese(text)
    has_strong_vietnamese = _has_strong_vietnamese(text)
    has_french = any(
        char in FRENCH_MARKERS
        for char in text
    )

    if language == "vie":
        return has_strong_vietnamese or (
            has_vietnamese
            and not has_french
            and not has_cjk
        )

    if language == "fra":
        return (
            has_french
            and not has_strong_vietnamese
            and not has_cjk
        )

    if language == "chi_tra":
        return has_cjk

    if language == "eng":
        return (
            any(char.isalpha() for char in text)
            and not has_cjk
            and not has_vietnamese
            and not has_french
        )

    return False


def _extract_regions(
    image: Image.Image,
    language: str,
) -> list[TextRegion]:
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config="--psm 11",
        output_type=Output.DICT,
    )

    regions: list[TextRegion] = []

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

        if not text or confidence < 25:
            continue

        if width <= 0 or height <= 0:
            continue

        if not _has_language_evidence(text, language):
            continue

        regions.append(
            TextRegion(
                left=left,
                top=top,
                width=width,
                height=height,
                text=text,
                language=language,
                confidence=confidence,
            )
        )

    return regions


def _region_is_claimed(
    region: TextRegion,
    claimed_mask: np.ndarray,
) -> bool:

    height, width = claimed_mask.shape

    x0 = max(0, region.left)
    y0 = max(0, region.top)
    x1 = min(width, region.right)
    y1 = min(height, region.bottom)

    if x0 >= x1 or y0 >= y1:
        return True

    return bool(claimed_mask[y0:y1, x0:x1].any())


def _claim_region(
    region: TextRegion,
    claimed_mask: np.ndarray,
) -> None:

    height, width = claimed_mask.shape

    x0 = max(0, region.left)
    y0 = max(0, region.top)
    x1 = min(width, region.right)
    y1 = min(height, region.bottom)

    if x0 < x1 and y0 < y1:
        claimed_mask[y0:y1, x0:x1] = True


def detect_text_regions(
    image: Image.Image,
) -> list[TextRegion]:
    configure_tesseract()

    working = np.array(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    height, width = working.shape[:2]

    claimed_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    selected_regions: list[TextRegion] = []

    for language in SCAN_ORDER:
        scan_array = working.copy()
        scan_array[claimed_mask] = 255

        scan_image = Image.fromarray(scan_array)

        candidates = _extract_regions(
            scan_image,
            language,
        )

        candidates.sort(
            key=lambda region: region.confidence,
            reverse=True,
        )

        for region in candidates:
            
            if _region_is_claimed(region, claimed_mask):
                continue

            selected_regions.append(region)
            _claim_region(region, claimed_mask)

    selected_regions.sort(
        key=lambda region: (
            region.top,
            region.left,
        )
    )

    return selected_regions


def _font() -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )

    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), 16)

    return ImageFont.load_default()


def mark_text_regions(
    input_path: str | Path,
    output_path: str | Path,
) -> list[TextRegion]:
    image = load_image(input_path).convert("RGB")
    regions = detect_text_regions(image)

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = _font()

    for region in regions:
        label, color = LANGUAGE_CONFIG[region.language]

        draw.rectangle(
            (
                region.left,
                region.top,
                region.right,
                region.bottom,
            ),
            outline=color,
            width=3,
        )

        label_box = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 4

        label_top = max(0, region.top - label_height)

        draw.rectangle(
            (
                region.left,
                label_top,
                region.left + label_width,
                region.top,
            ),
            fill=color,
        )

        draw.text(
            (
                region.left + 4,
                label_top + 2,
            ),
            label,
            fill="white",
            font=font,
        )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    marked.save(output)

    return regions
