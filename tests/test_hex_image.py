import cv2
import numpy as np

from src.hex_image import crop_hexagram_image, save_hexagram_image


def make_page():
    img = np.full((1600, 1000), 255, np.uint8)
    for i in range(6):
        y = 300 + i * 25
        if i in (0, 3, 5):
            cv2.rectangle(img, (100, y), (220, y + 12), 0, -1)
        else:
            cv2.rectangle(img, (100, y), (150, y + 12), 0, -1)
            cv2.rectangle(img, (170, y), (220, y + 12), 0, -1)
    cv2.putText(img, "Can thuong + Kham ha =", (450, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.line(img, (80, 520), (920, 520), 0, 3)
    return img


def test_crops_only_the_hexagram_block():
    h, w = crop_hexagram_image(make_page(), rule_y=520).shape[:2]
    assert 140 <= h <= 200
    assert 100 <= w <= 160


def test_crop_contains_dark_pixels():
    assert (crop_hexagram_image(make_page(), rule_y=520) < 128).sum() > 500


def test_returns_none_when_no_block_found():
    assert crop_hexagram_image(np.full((1600, 1000), 255, np.uint8), rule_y=520) is None


def test_ignores_content_below_the_rule():
    img = make_page()
    cv2.rectangle(img, (100, 700), (300, 760), 0, -1)
    assert crop_hexagram_image(img, rule_y=520).shape[0] < 250


def test_prefers_block_near_rule_over_title_above():
    img = make_page()
    cv2.putText(img, "QUE MONG", (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 6)  
    h, _ = crop_hexagram_image(img, rule_y=520).shape[:2]
    assert h < 200


def test_save_writes_png(tmp_path):
    p = save_hexagram_image(make_page(), 520, tmp_path / "images" / "que_1.png")
    assert p.exists() and cv2.imread(str(p)) is not None


def test_save_returns_none_when_not_found(tmp_path):
    assert save_hexagram_image(np.full((800, 600), 255, np.uint8), 400, tmp_path / "x.png") is None
