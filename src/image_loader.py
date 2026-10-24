from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
    "TIFF",
    "BMP",
    "PNG",
}

def load_image(path: str | Path) -> Image.Image:
    path = Path(path)

    if not path.is_file():
        raise ValueError(f"Không tìm thấy file: {path}")

    try:
        with Image.open(path) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError("Định dạng ảnh không hỗ trợ")

            image.load()
            return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Không hợp lệ: {path}") from exc


def load_images(paths: list[str | Path]) -> list[Image.Image]:
    return [load_image(path) for path in paths]