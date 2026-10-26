import cv2
import numpy as np

from src.page_crop import crop_page, detect_page_corners, save_page_crop


def make_skewed_page():
    image = np.zeros((700, 600, 3), dtype=np.uint8)
    page = np.array([[120, 60], [470, 100], [430, 620], [80, 580]], dtype=np.int32)
    cv2.fillConvexPoly(image, page, (245, 245, 245))
    cv2.putText(image, "PAGE", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return image


def make_open_book():
    image = np.full((700, 600, 3), 100, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (470, 620), (245, 245, 245), -1)
    cv2.rectangle(image, (476, 95), (590, 610), (240, 240, 240), -1)
    cv2.line(image, (473, 100), (473, 600), (125, 125, 125), 5)
    cv2.putText(image, "PAGE", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return image


def test_detects_page_corners():
    corners = detect_page_corners(make_skewed_page())

    assert corners is not None
    assert corners.shape == (4, 2)


def test_crop_page_straightens_skewed_page():
    result = crop_page(make_skewed_page())

    assert result is not None
    assert result.shape[1] > result.shape[0] * 0.6
    assert result.shape[0] > result.shape[1]


def test_crop_page_returns_none_when_no_page_is_visible():
    assert crop_page(np.zeros((300, 400, 3), dtype=np.uint8)) is None


def test_detects_a_page_that_fills_the_image():
    image = np.full((500, 400, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEXT", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    result = crop_page(image)

    assert result is not None
    assert result.shape[0] >= image.shape[0] - 2
    assert result.shape[1] >= image.shape[1] - 2


def test_extracts_one_page_from_an_open_book():
    corners = detect_page_corners(make_open_book())
    result = crop_page(make_open_book())

    assert corners is not None
    assert 350 <= corners[1, 0] - corners[0, 0] <= 380
    assert result is not None
    assert result.shape[1] < 400


def test_save_page_crop_writes_image(tmp_path):
    input_path = tmp_path / "skewed.jpg"
    output_path = tmp_path / "cropped" / "page.png"
    assert cv2.imwrite(str(input_path), make_skewed_page())

    result = save_page_crop(input_path, output_path)

    assert result == output_path
    assert cv2.imread(str(output_path)) is not None
