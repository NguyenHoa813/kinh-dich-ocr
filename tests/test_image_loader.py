from pathlib import Path

import pytest
from PIL import Image

from src.image_loader import load_image, load_images


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "page.png"  #tạo ảnh trắng để test
    Image.new("RGB", (800, 1200), "white").save(path)
    return path


def test_load_valid_image(valid_image):
    image = load_image(valid_image)

    assert image is not None
    assert image.size == (800, 1200)


def test_reject_unsupported_extension(tmp_path):
    path = tmp_path / "document.txt" #file text sai địng dạng để test
    path.write_text("not an image")

    with pytest.raises(ValueError):
        load_image(path)


def test_reject_corrupted_image(tmp_path):
    path = tmp_path / "broken.png" #file ảnh có nội dung không phải ảnh thật
    path.write_bytes(b"not a valid image")

    with pytest.raises(ValueError):
        load_image(path)


def test_load_multiple_images(valid_image, tmp_path):
    second = tmp_path / "page2.png"  #thêm ảnh thứ 2 để test load 2 ảnh được
    Image.new("RGB", (600, 900), "white").save(second)

    images = load_images([valid_image, second])

    assert len(images) == 2