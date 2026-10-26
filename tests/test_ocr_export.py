import json

from src.ocr_export import export_ocr_document
from src.ocr_text import OcrLine


def make_lines():
    texts = [
        "QUẺ KIỀN",
        "Kiền thượng + Khảm hạ =",
        "Bát Thuần Kiền",
        "KINH VĂN",
        "乾",
        "Am: Kiền, Nguyên, Hanh",
        "Nghia: Bốn đức",
        "GIẢI THÍCH",
        "Cả sáu hào đều dương.",
        "GHI CHÚ NGƯỜI DỊCH",
        "1) Chú thích",
        "nối dòng",
        "QUYỂN 1",
        "33",
    ]
    return [
        OcrLine(text, 0, index * 10, 100, 8, 90.0)
        for index, text in enumerate(texts)
    ]


def test_export_builds_sample_json_and_html(tmp_path):
    image_path = tmp_path / "images" / "hex.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")
    html_path = tmp_path / "page.html"
    json_path = tmp_path / "page.json"

    document = export_ocr_document(
        make_lines(),
        html_path,
        json_path,
        document_id="001",
        hexagram_image=image_path,
        stylesheet="style.css",
    )

    page = document["pages"][0]
    que, kinh_van, explanation, notes = page["sections"]
    assert document["document_id"] == "001"
    assert page["page_number"] == 33
    assert que == {
        "type": "Que",
        "title": "QUẺ KIỀN",
        "hexagram_image": "images/hex.png",
        "upper_trigram": "Kiền",
        "lower_trigram": "Khảm",
        "hexagram_name": "Bát Thuần Kiền",
    }
    assert kinh_van == {
        "type": "KinhVan",
        "han_text": "乾",
        "sound": "Kiền, Nguyên, Hanh",
        "meaning": "Bốn đức",
    }
    assert explanation["text"] == "Cả sáu hào đều dương."
    assert notes["notes"] == [
        {"number": 1, "text": "Chú thích\nnối dòng", "han_text": ""},
    ]

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved == document
    html_content = html_path.read_text(encoding="utf-8")
    assert '<img src="images/hex.png"' in html_content
    assert "<div class=\"han-text\" lang=\"zh-Hant\">乾</div>" in html_content
    assert "<span class=\"page-number\">33</span>" in html_content
    assert '<figure class="page-scan">' not in html_content
    assert "<details class=\"ocr-transcript\">" not in html_content


def test_export_escapes_ocr_html_content(tmp_path):
    lines = [OcrLine("<script>alert(1)</script>", 0, 0, 10, 10, 90.0)]

    export_ocr_document(lines, tmp_path / "page.html", tmp_path / "page.json")

    html_content = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_content
    assert "<script>alert(1)</script>" not in html_content


def test_export_reflows_ocr_line_breaks_for_readable_html(tmp_path):
    lines = [
        OcrLine("GIẢI THÍCH", 0, 0, 100, 10, 90.0),
        OcrLine("Đây là dòng thứ nhất", 0, 10, 100, 10, 90.0),
        OcrLine("và đây là dòng tiếp theo.", 0, 20, 100, 10, 90.0),
    ]

    export_ocr_document(lines, tmp_path / "page.html", tmp_path / "page.json")

    html_content = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "Đây là dòng thứ nhất và đây là dòng tiếp theo." in html_content


def test_export_keeps_continuation_page_order_without_que_header(tmp_path):
    lines = [
        OcrLine("Đoạn giải thích tiếp tục từ trang trước.", 0, 0, 100, 10, 90.0),
        OcrLine("Ghi chú của người dịch", 0, 20, 100, 10, 90.0),
        OcrLine("(1) Nội dung chú thích.", 0, 30, 100, 10, 90.0),
        OcrLine("KINH VĂN", 0, 40, 100, 10, 90.0),
        OcrLine("初 九 潛 龍", 0, 50, 100, 10, 90.0),
    ]
    html_path = tmp_path / "page.html"
    json_path = tmp_path / "page.json"

    document = export_ocr_document(lines, html_path, json_path)

    sections = document["pages"][0]["sections"]
    assert [section["type"] for section in sections] == [
        "GiaiThich",
        "GhiChuNguoiDich",
        "KinhVan",
    ]
    html_content = html_path.read_text(encoding="utf-8")
    assert "<h1>Đoạn giải thích tiếp tục từ trang trước.</h1>" not in html_content
    assert html_content.index("GIẢI THÍCH") < html_content.index(
        "GHI CHÚ NGƯỜI DỊCH"
    ) < html_content.index("KINH VĂN")
