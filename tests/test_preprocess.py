from PIL import Image

from src.preprocess import preprocess_image


def test_preprocess_returns_valid_image():
    image = Image.new("RGB", (800, 1200), "white")

    result = preprocess_image(image)

    assert result is not None
    assert result.ndim == 2             #mảng 2 chiều (ảnh xám)
    assert result.shape == (1200, 800)


def test_preprocess_preserves_page_dimensions():
    image = Image.new("RGB", (1000, 1500), "white")

    result = preprocess_image(image)

    assert result.shape[1] == 1000      #đảm bảo không bị cắt ảnh
    assert result.shape[0] == 1500