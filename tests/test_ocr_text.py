from PIL import Image

from src import ocr_text


def _ocr_data(texts, *, left=20, top=40):
    return {
        "text": texts,
        "conf": ["90"] * len(texts),
        "left": [left + index * 20 for index in range(len(texts))],
        "top": [top] * len(texts),
        "width": [18] * len(texts),
        "height": [16] * len(texts),
        "block_num": [1] * len(texts),
        "par_num": [1] * len(texts),
        "line_num": [1] * len(texts),
    }


def test_read_lines_runs_dedicated_vietnamese_and_chinese_passes(monkeypatch):
    calls = []

    def image_to_data(image, *, lang, config, output_type):
        assert output_type is ocr_text.Output.DICT
        calls.append((image.size, lang, config, output_type))
        if lang == "vie":
            return {
                "text": ["KINH", "VĂN", "Âm:", "Kiền", "GIẢI", "THÍCH"],
                "conf": ["90"] * 6,
                "left": [10, 60, 10, 50, 10, 60],
                "top": [100, 100, 200, 200, 300, 300],
                "width": [40, 40, 35, 40, 40, 50],
                "height": [18] * 6,
                "block_num": [1, 1, 2, 2, 3, 3],
                "par_num": [1, 1, 1, 1, 1, 1],
                "line_num": [1, 1, 1, 1, 1, 1],
            }
        return _ocr_data(["乾", "元"], top=10)

    monkeypatch.setattr(ocr_text.pytesseract, "image_to_data", image_to_data)

    lines = ocr_text._read_lines(Image.new("RGB", (100, 400), "white"))

    assert calls == [
        ((100, 400), "vie", ocr_text.VIETNAMESE_CONFIG, ocr_text.Output.DICT),
        ((100, 78), "chi_tra", ocr_text.HAN_CONFIG, ocr_text.Output.DICT),
    ]
    assert [(line.text, line.left, line.top) for line in lines] == [
        ("KINH VĂN", 10, 100),
        ("乾 元", 20, 130),
        ("Âm: Kiền", 10, 200),
        ("GIẢI THÍCH", 10, 300),
    ]
    assert all(line.width > 0 and line.height > 0 for line in lines)


def test_read_language_lines_ignores_empty_and_low_confidence_words(monkeypatch):
    data = _ocr_data(["", "Kiền", "rác"], left=10, top=10)
    data["conf"] = ["-1", "92", "35"]

    def image_to_data(image, **kwargs):
        assert image.size == (100, 80)
        assert kwargs["lang"] == "vie"
        return data

    monkeypatch.setattr(ocr_text.pytesseract, "image_to_data", image_to_data)

    lines = ocr_text._read_language_lines(
        Image.new("RGB", (100, 80), "white"),
        "vie",
        config=ocr_text.VIETNAMESE_CONFIG,
    )

    assert len(lines) == 1
    assert lines[0].text == "Kiền"


def test_read_lines_falls_back_to_sparse_han_scan_when_headings_missing(monkeypatch):
    calls = []

    def image_to_data(image, *, lang, config, output_type):
        assert output_type is ocr_text.Output.DICT
        calls.append((lang, config, image.size))
        return _ocr_data(["Kiền"] if lang == "vie" else ["乾"])

    monkeypatch.setattr(ocr_text.pytesseract, "image_to_data", image_to_data)

    lines = ocr_text._read_lines(Image.new("RGB", (100, 80), "white"))

    assert calls == [
        ("vie", ocr_text.VIETNAMESE_CONFIG, (100, 80)),
        ("chi_tra", ocr_text.HAN_FALLBACK_CONFIG, (100, 80)),
    ]
    assert [line.text for line in lines] == ["Kiền", "乾"]
