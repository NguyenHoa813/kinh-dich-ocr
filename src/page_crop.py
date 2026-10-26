from pathlib import Path

import cv2
import numpy as np


def _order_corners(points: np.ndarray) -> np.ndarray:
    corners = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    corners[0] = points[np.argmin(sums)]
    corners[2] = points[np.argmax(sums)]
    corners[1] = points[np.argmin(differences)]
    corners[3] = points[np.argmax(differences)]
    return corners


def _full_frame_page_corners(gray: np.ndarray) -> np.ndarray | None:
    height, width = gray.shape
    border = np.concatenate(
        (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1])
    )
    light_pixels = np.count_nonzero(gray >= 170) / gray.size
    dark_pixels = np.count_nonzero(gray < 100) / gray.size
    light_border = np.count_nonzero(border >= 170) / border.size
    if light_pixels < 0.65 or not 0.003 <= dark_pixels <= 0.35:
        return None
    if light_border < 0.75:
        return None
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def _largest_paper_region(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    paper = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(paper)
    image_area = gray.shape[0] * gray.shape[1]
    regions = [
        stats[index]
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= image_area * 0.2
    ]
    if not regions:
        return None

    region = max(regions, key=lambda item: item[cv2.CC_STAT_AREA])
    x = int(region[cv2.CC_STAT_LEFT])
    y = int(region[cv2.CC_STAT_TOP])
    width = int(region[cv2.CC_STAT_WIDTH])
    height = int(region[cv2.CC_STAT_HEIGHT])
    if height < gray.shape[0] * 0.35:
        return None
    return x, y, width, height


def _vertical_gutter_candidates(
    gray: np.ndarray,
    region: tuple[int, int, int, int],
) -> list[tuple[int, float]]:
    x, y, width, height = region
    band = gray[y + height // 10 : y + height * 9 // 10, x : x + width]
    profile = cv2.GaussianBlur(
        band.astype(np.float32).mean(axis=0).reshape(1, -1),
        (0, 0),
        sigmaX=4,
    ).ravel()
    baseline = cv2.blur(profile.reshape(1, -1), (51, 1)).ravel()
    prominence = baseline - profile

    candidates = []
    margin = max(8, round(width * 0.08))
    for position in range(margin + 1, width - margin - 1):
        if (
            prominence[position] >= 4
            and prominence[position] >= prominence[position - 1]
            and prominence[position] >= prominence[position + 1]
        ):
            candidates.append((x + position, float(prominence[position])))
    return candidates


def _detect_open_book_page(gray: np.ndarray) -> np.ndarray | None:
    region = _largest_paper_region(gray)
    if region is None:
        return None

    x, y, width, height = region
    if width / height < 0.8:
        return None

    page_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    row_coverage = (page_mask[:, x : x + width] > 0).mean(axis=1)
    row_coverage = cv2.blur(
        row_coverage.astype(np.float32).reshape(-1, 1),
        (1, 31),
    ).ravel()
    rows = np.flatnonzero(row_coverage > 0.5)
    row_groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    page_rows = [
        group
        for group in row_groups
        if len(group) >= gray.shape[0] * 0.3
        and group[0] >= gray.shape[0] * 0.05
        and group[-1] <= gray.shape[0] * 0.95
    ]
    if not page_rows:
        return None

    page_rows_group = max(page_rows, key=len)
    top = int(page_rows_group[0])
    bottom = int(page_rows_group[-1])
    page_height = bottom - top
    if page_height < gray.shape[0] * 0.35:
        return None

    candidates = [
        (position, prominence)
        for position, prominence in _vertical_gutter_candidates(gray, region)
        if x + width * 0.12 <= position <= x + width * 0.88
    ]
    valid_pairs = []
    for gutter, _ in candidates:
        for left, right in ((x, gutter), (gutter, x + width)):
            ratio = (right - left) / page_height
            if not 0.5 <= ratio <= 0.8:
                continue
            valid_pairs.append((abs(ratio - 0.68), left, right))

    if not valid_pairs:
        return None
    _, left, right = min(valid_pairs)
    return np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def detect_page_corners(image: np.ndarray) -> np.ndarray | None:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
        )
    else:
        raise ValueError("Ảnh phải có dạng grayscale, BGR hoặc BGRA")

    height, width = gray.shape
    if height < 2 or width < 2:
        raise ValueError("Ảnh quá nhỏ để phát hiện trang sách")

    scale = min(1.0, 1600 / max(height, width))
    if scale < 1:
        working = cv2.resize(
            gray,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = gray

    open_book_corners = _detect_open_book_page(working)
    if open_book_corners is not None:
        if scale < 1:
            open_book_corners /= scale
        return open_book_corners

    blurred = cv2.GaussianBlur(working, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    minimum_area = working.shape[0] * working.shape[1] * 0.2
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue

        corners = _order_corners(polygon.reshape(4, 2).astype(np.float32))
        if scale < 1:
            corners /= scale
        return corners
    return _full_frame_page_corners(gray)


def crop_page(image: np.ndarray) -> np.ndarray | None:
    corners = detect_page_corners(image)
    if corners is None:
        return None

    top_left, top_right, bottom_right, bottom_left = corners
    output_width = round(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    output_height = round(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if output_width < 2 or output_height < 2:
        return None

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (output_width, output_height))


def save_page_crop(input_path: str | Path, output_path: str | Path) -> Path | None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {input_path}")

    result = crop_page(image)
    if result is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Không thể lưu ảnh đã cắt: {output_path}")
    return output_path
