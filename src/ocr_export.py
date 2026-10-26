from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Sequence

from src.ocr_text import OcrLine


def _normalized(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _is_heading(text: str, heading: str) -> bool:
    normalized = " ".join(_normalized(text).upper().split())
    return normalized == heading or normalized.startswith(f"{heading} ")


def _has_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _split_trigrams(text: str) -> tuple[str, str]:
    parts = text.split("+", maxsplit=1)
    if len(parts) != 2:
        return "", ""

    upper, lower = (part.strip().rstrip("=").strip() for part in parts)
    if _normalized(upper).lower().endswith(" thuong"):
        upper = upper.rsplit(maxsplit=1)[0]
    if _normalized(lower).lower().endswith(" ha"):
        lower = lower.rsplit(maxsplit=1)[0]
    return upper, lower


def _footer(lines: list[str]) -> tuple[str, int | None, set[int]]:
    volume = ""
    page_number = None
    consumed: set[int] = set()

    for index, text in enumerate(lines):
        match = re.match(
            r"^\s*quyen\s+\d+\s*$",
            _normalized(text),
            re.IGNORECASE,
        )
        if not match:
            continue
        volume = text.strip()
        consumed.add(index)
        if index + 1 < len(lines) and re.fullmatch(r"\s*\d+\s*", lines[index + 1]):
            page_number = int(lines[index + 1].strip())
            consumed.add(index + 1)
        break

    return volume, page_number, consumed


def _build_document(
    lines: Sequence[OcrLine],
    document_id: str,
    page_number: int | None,
    hexagram_image: str,
) -> tuple[dict[str, object], str]:
    texts = [line.text.strip() for line in lines if line.text.strip()]
    volume, detected_page, footer_indexes = _footer(texts)
    if page_number is None:
        page_number = detected_page

    que: dict[str, str] = {
        "type": "Que",
        "title": "",
        "hexagram_image": hexagram_image,
        "upper_trigram": "",
        "lower_trigram": "",
        "hexagram_name": "",
    }
    kinh_van: dict[str, str] = {
        "type": "KinhVan",
        "han_text": "",
        "sound": "",
        "meaning": "",
    }
    explanation: dict[str, str] = {"type": "GiaiThich", "text": ""}
    notes: dict[str, object] = {"type": "GhiChuNguoiDich", "notes": []}

    first_text = texts[0] if texts else ""
    state = "explanation"
    if _is_heading(first_text, "QUE"):
        state = "que"
    elif _is_heading(first_text, "KINH VAN"):
        state = "kinh_van"
    elif _is_heading(first_text, "GHI CHU"):
        state = "notes"
    elif any("+" in text for text in texts[:4]) and len(first_text) < 60:
        state = "que"
    section_order = [state]
    sound_lines: list[str] = []
    han_lines: list[str] = []
    explanation_lines: list[str] = []
    note_items: list[dict[str, object]] = []
    intro_seen = False

    def set_state(next_state: str) -> None:
        nonlocal state
        state = next_state
        if next_state not in section_order:
            section_order.append(next_state)

    for index, text in enumerate(texts):
        if index in footer_indexes:
            continue
        if _is_heading(text, "QUE"):
            set_state("que")
            que["title"] = text
            continue
        if _is_heading(text, "KINH VAN"):
            set_state("kinh_van")
            continue
        if _is_heading(text, "GIAI THICH"):
            set_state("explanation")
            continue
        if _is_heading(text, "GHI CHU"):
            set_state("notes")
            continue

        if state == "que":
            if not que["title"]:
                que["title"] = text
            elif "+" in text and not intro_seen:
                que["upper_trigram"], que["lower_trigram"] = _split_trigrams(text)
                intro_seen = True
            elif not que["hexagram_name"]:
                que["hexagram_name"] = text
            else:
                que["hexagram_name"] += f"\n{text}"
        elif state == "kinh_van":
            normalized_text = _normalized(text).strip().lower()
            if re.match(r"^am\s*[:：]", normalized_text):
                sound_lines.append(re.split(r"[:：]", text, maxsplit=1)[1].strip())
            else:
                if re.match(r"^nghia\s*[:：]", normalized_text):
                    kinh_van["meaning"] = "\n".join(
                        part
                        for part in (
                            kinh_van["meaning"],
                            re.split(r"[:：]", text, maxsplit=1)[1].strip(),
                        )
                        if part
                    )
                elif _has_cjk(text):
                    han_lines.append(text)
                else:
                    sound_lines.append(text)
        elif state == "explanation":
            explanation_lines.append(text)
        elif state == "notes":
            match = re.match(r"^\s*(\d+)[.)、]\s*(.*)$", text)
            if match:
                note_items.append(
                    {
                        "number": int(match.group(1)),
                        "text": match.group(2),
                        "han_text": "",
                    }
                )
            elif note_items:
                note_items[-1]["text"] = (
                    f"{note_items[-1]['text']}\n{text}".strip()
                )
            else:
                note_items.append(
                    {"number": 1, "text": text, "han_text": ""}
                )

    kinh_van["han_text"] = "\n".join(han_lines)
    kinh_van["sound"] = "\n".join(line for line in sound_lines if line)
    explanation["text"] = "\n".join(explanation_lines)
    notes["notes"] = note_items
    sections_by_state = {
        "que": que,
        "kinh_van": kinh_van,
        "explanation": explanation,
        "notes": notes,
    }
    has_content = {
        "que": bool(
            que["title"]
            or que["upper_trigram"]
            or que["lower_trigram"]
            or que["hexagram_name"]
            or que["hexagram_image"]
        ),
        "kinh_van": bool(
            kinh_van["han_text"] or kinh_van["sound"] or kinh_van["meaning"]
        ),
        "explanation": bool(explanation["text"]),
        "notes": bool(note_items),
    }
    ordered_sections = [
        sections_by_state[name]
        for name in section_order
        if has_content[name]
    ]
    ordered_sections.extend(
        sections_by_state[name]
        for name in ("que", "kinh_van", "explanation", "notes")
        if name not in section_order and has_content[name]
    )
    document = {
        "document_id": document_id,
        "pages": [
            {
                "page_number": page_number,
                "sections": ordered_sections,
            }
        ],
    }
    return document, volume


def _html_text(text: str, *, preserve_lines: bool = False) -> str:
    escaped = html.escape(text)
    return escaped.replace("\n", "<br>\n") if preserve_lines else " ".join(escaped.splitlines())


def _render_html(
    document: dict[str, object],
    volume: str,
    stylesheet: str,
) -> str:
    page = document["pages"][0]
    page_number = page["page_number"]
    section_markup: list[str] = []
    for section in page["sections"]:
        section_type = section["type"]
        if section_type == "Que":
            image_markup = ""
            if section["hexagram_image"]:
                image_markup = (
                    '                <figure class="hexagram-figure">\n'
                    f'                    <img src="{html.escape(section["hexagram_image"], quote=True)}" '
                    f'alt="Hình {html.escape(section["title"], quote=True)}">\n'
                    "                </figure>\n"
                )
            trigram_markup = ""
            if section["upper_trigram"] or section["lower_trigram"]:
                trigram_markup = (
                    '                <div class="trigram-line" aria-label="Thượng quái và hạ quái">\n'
                    f'                    <span>{_html_text(section["upper_trigram"])} thượng</span>\n'
                    '                    <span class="trigram-separator">+</span>\n'
                    f'                    <span>{_html_text(section["lower_trigram"])} hạ</span>\n'
                    "                </div>\n"
                )
            section_markup.append(
                '            <header class="hexagram-header">\n'
                f'                <h1>{_html_text(section["title"])}</h1>\n'
                '                <div class="hexagram-intro">\n'
                f'{image_markup}                <div class="hexagram-name">\n'
                f'{trigram_markup}'
                f'                    <h2>{_html_text(section["hexagram_name"])}</h2>\n'
                "                </div>\n"
                "                </div>\n"
                "            </header>\n"
                '            <hr class="section-divider">'
            )
        elif section_type == "KinhVan":
            section_markup.append(
                '            <section class="kinhvan content-card">\n'
                '                <h2 class="section-heading">KINH VĂN</h2>\n'
                f'                <div class="han-text" lang="zh-Hant">{_html_text(section["han_text"], preserve_lines=True)}</div>\n'
                '                <div class="translation-grid">\n'
                '                    <section class="subsection am">\n'
                f'                        <h3>Âm</h3><p>{_html_text(section["sound"])}</p>\n'
                "                    </section>\n"
                '                    <section class="subsection meaning">\n'
                f'                        <h3>Nghĩa</h3><p>{_html_text(section["meaning"])}</p>\n'
                "                    </section>\n"
                "                </div>\n"
                "            </section>"
            )
        elif section_type == "GiaiThich":
            section_markup.append(
                '            <section class="explanation content-card">\n'
                '                <h2 class="section-heading">GIẢI THÍCH</h2>\n'
                f'                <p>{_html_text(section["text"])}</p>\n'
                "            </section>"
            )
        elif section_type == "GhiChuNguoiDich":
            note_items = "\n".join(
                f'                    <li data-note-number="{note["number"]}">'
                f'<span class="note-number">{note["number"]}.</span>'
                f'<span>{_html_text(note["text"])}</span></li>'
                for note in section["notes"]
            )
            section_markup.append(
                '            <section class="translator-notes content-card">\n'
                '                <h2 class="section-heading">GHI CHÚ NGƯỜI DỊCH</h2>\n'
                "                <ol>\n"
                f"{note_items}\n"
                "                </ol>\n"
                "            </section>"
            )
    body_markup = "\n".join(section_markup)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Kinh Dịch — Tài liệu OCR</title>
    <link rel="stylesheet" href="{html.escape(stylesheet, quote=True)}">
</head>
<body>
    <main class="document" id="document-{html.escape(str(document["document_id"]), quote=True)}">
        <article class="book-page" data-page-number="{html.escape(str(page_number or ""), quote=True)}">
{body_markup}
            <footer class="page-footer">
                <span class="book-volume">{_html_text(volume)}</span>
                <span class="page-number">{_html_text(str(page_number) if page_number is not None else "")}</span>
            </footer>
        </article>
    </main>
</body>
</html>
"""


def export_ocr_document(
    lines: Sequence[OcrLine],
    html_output: str | Path,
    json_output: str | Path,
    *,
    document_id: str | None = None,
    page_number: int | None = None,
    hexagram_image: str | Path | None = None,
    stylesheet: str = "style.css",
) -> dict[str, object]:
    html_path = Path(html_output)
    json_path = Path(json_output)
    image_reference = ""
    if hexagram_image is not None:
        image_reference = os.path.relpath(
            Path(hexagram_image),
            start=html_path.parent,
        ).replace("\\", "/")
    document, volume = _build_document(
        lines,
        document_id or html_path.stem,
        page_number,
        image_reference,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        _render_html(document, volume, stylesheet),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document
