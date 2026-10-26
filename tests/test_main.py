import sys
from pathlib import Path

import pytest

import main
from src.ocr_text import OcrLine


@pytest.mark.parametrize(
    ("arguments", "expected_hex_output"),
    [
        (
            ["marker", "page.png", "output/marked.png"],
            Path("output/hex/page_hex.png"),
        ),
        (
            ["text", "page.png", "output/ocr"],
            Path("output/ocr/hex/page_hex.png"),
        ),
    ],
)
def test_ocr_commands_automatically_split_hexagram(
    monkeypatch,
    arguments,
    expected_hex_output,
):
    split_calls = []
    monkeypatch.setattr(
        main,
        "_run_hex_split",
        lambda input_path, output_path, **kwargs: split_calls.append(
            (input_path, output_path, kwargs)
        ),
    )
    monkeypatch.setattr(main, "_run_marker", lambda *_args: None)
    monkeypatch.setattr(main, "_run_text", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])

    main.main()

    assert split_calls == [
        (Path("page.png"), expected_hex_output, {}),
    ]


def test_hex_command_uses_default_output_and_requires_crop(monkeypatch):
    split_calls = []
    monkeypatch.setattr(
        main,
        "_run_hex_split",
        lambda input_path, output_path, **kwargs: split_calls.append(
            (input_path, output_path, kwargs)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "hex", "page.png"])

    main.main()

    assert split_calls == [
        (
            Path("page.png"),
            Path("output/hex/page_hex.png"),
            {"required": True},
        ),
    ]


def test_page_command_uses_default_output(monkeypatch):
    crop_calls = []
    monkeypatch.setattr(
        main,
        "_run_page_crop",
        lambda input_path, output_path: crop_calls.append((input_path, output_path)),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "page", "page.png"])

    main.main()

    assert crop_calls == [
        (Path("page.png"), Path("output/page/page_page.png")),
    ]


def test_all_command_runs_page_processing_before_ocr_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "_run_page_crop",
        lambda input_path, output_path: calls.append(
            ("page", input_path, output_path)
        ),
    )
    monkeypatch.setattr(
        main,
        "_run_hex_split",
        lambda input_path, output_path, **kwargs: calls.append(
            ("hex", input_path, output_path)
        ),
    )
    monkeypatch.setattr(
        main,
        "_run_marker",
        lambda input_path, output_path: calls.append(
            ("marker", input_path, output_path)
        ),
    )
    monkeypatch.setattr(
        main,
        "_run_text",
        lambda input_path, output_dir, hex_path: calls.append(
            ("text-export", input_path, output_dir, hex_path)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "all", "page3.png", "output/page3"])

    main.main()

    processed = Path("output/page3/page3_page.png")
    hex_output = Path("output/page3/hex/page3_hex.png")
    assert calls == [
        ("page", Path("page3.png"), processed),
        ("hex", processed, hex_output),
        ("marker", processed, Path("output/page3/page3_marked.png")),
        ("text-export", processed, Path("output/page3/ocr"), hex_output),
    ]


def test_run_text_saves_raw_ocr_and_exports_html_json(monkeypatch, tmp_path):
    input_path = tmp_path / "page.png"
    output_dir = tmp_path / "output" / "ocr"
    hex_path = output_dir / "hex" / "page_hex.png"
    hex_path.parent.mkdir(parents=True)
    hex_path.write_bytes(b"image")
    lines = [OcrLine("KINH VĂN", 0, 0, 10, 10, 90.0)]
    saved_paths = []
    export_args = []

    def save_ocr_files(input_file, text_path, json_path):
        saved_paths.extend((text_path, json_path))
        return lines

    def export_ocr_document(ocr_lines, html_path, json_path, **kwargs):
        export_args.append((ocr_lines, html_path, json_path, kwargs))

    monkeypatch.setattr(main, "save_ocr_files", save_ocr_files)
    monkeypatch.setattr(main, "export_ocr_document", export_ocr_document)

    main._run_text(input_path, output_dir, hex_path)

    assert saved_paths == [
        output_dir / "page.txt",
        output_dir / "page_ocr.json",
    ]
    assert export_args == [
        (
            lines,
            output_dir / "page.html",
            output_dir / "page.json",
            {
                "document_id": "page",
                "hexagram_image": hex_path,
                "stylesheet": "../style.css",
            },
        )
    ]
    assert (output_dir.parent / "style.css").read_text(encoding="utf-8") == (
        Path(main.__file__).parent / "output" / "style.css"
    ).read_text(encoding="utf-8")
