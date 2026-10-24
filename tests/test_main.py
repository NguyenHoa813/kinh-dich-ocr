import sys
from pathlib import Path

import pytest

import main


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
        (
            ["all", "page.png", "output"],
            Path("output/hex/page_hex.png"),
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
