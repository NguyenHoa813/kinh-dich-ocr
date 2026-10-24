import argparse
from pathlib import Path

import cv2


def _group_horizontal_strokes(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = gray.shape[:2]
    kernel_width = max(20, width // 35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < max(40, width // 40) or box_height > max(40, height // 40):
            continue
        if x <= 2 or y <= 2 or x + box_width >= width - 2:
            continue
        candidates.append((x, y, box_width, box_height))

    for anchor in candidates:
        anchor_x = anchor[0] + anchor[2] / 2
        nearby = [
            candidate
            for candidate in candidates
            if abs(candidate[0] + candidate[2] / 2 - anchor_x) <= 220
        ]
        rows = []
        for candidate in sorted(nearby, key=lambda item: item[1]):
            center_y = candidate[1] + candidate[3] / 2
            row = next(
                (row for row in rows if abs(center_y - row["center_y"]) <= 12),
                None,
            )
            if row is None:
                rows.append({"center_y": center_y, "boxes": [candidate]})
            else:
                row["boxes"].append(candidate)

        rows.sort(key=lambda row: row["center_y"])
        for start in range(max(0, len(rows) - 5)):
            group = rows[start : start + 6]
            if len(group) < 6:
                continue
            y_values = [row["center_y"] for row in group]
            gaps = [second - first for first, second in zip(y_values, y_values[1:])]
            average_gap = sum(gaps) / len(gaps)
            if not 10 <= average_gap <= 60 or max(gaps) - min(gaps) > 25:
                continue

            boxes = [box for row in group for box in row["boxes"]]
            x0 = min(box[0] for box in boxes)
            y0 = min(box[1] for box in boxes)
            x1 = max(box[0] + box[2] for box in boxes)
            y1 = max(box[1] + box[3] for box in boxes)
            if x1 - x0 <= max(250, width // 10):
                return x0, y0, x1 - x0, y1 - y0
    return None


def crop_hexagram_image(gray, rule_y=None, pad=8):
    if rule_y is None:
        box = _group_horizontal_strokes(gray)
        if box is None:
            return None
        x, y, box_width, box_height = box
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(gray.shape[1], x + box_width + pad)
        y1 = min(gray.shape[0], y + box_height + pad)
        return gray[y0:y1, x0:x1]

    h, w = gray.shape[:2]
    region = gray[: max(0, int(rule_y) - 5), : w // 2]
    if region.size == 0:
        return None
    _, bw = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(3, int(w * 0.025))  # nối 6 vạch thành 1 khối
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) + (cv2.contourArea(c),) for c in cnts]
    boxes = [b for b in boxes if b[2] * b[3] >= 2000]  # bỏ nhiễu nhỏ
    if not boxes:
        return None
    interior_boxes = [
        b
        for b in boxes
        if b[0] > 2
        and b[1] > 2
        and b[0] + b[2] < region.shape[1] - 2
        and b[1] + b[3] < region.shape[0] - 2
    ]
    if interior_boxes:
        boxes = interior_boxes
    max_area = max(b[4] for b in boxes)
    # ứng viên đủ lớn; chọn khối nằm sát đường kẻ nhất (tránh nhầm với chữ tiêu đề phía trên)
    cands = [b for b in boxes if b[4] >= 0.5 * max_area]
    x, y, cw, ch, _ = max(cands, key=lambda b: b[1] + b[3])

    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(region.shape[1], x + cw + pad), min(region.shape[0], y + ch + pad)
    return gray[y0:y1, x0:x1]


def save_hexagram_image(gray, rule_y, out_path):
    crop = crop_hexagram_image(gray, rule_y)
    if crop is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Crop the hexagram area above a horizontal rule."
    )
    parser.add_argument("input", type=Path, help="Input page image")
    parser.add_argument(
        "second",
        help="Output image path, or rule_y when using the legacy form",
    )
    parser.add_argument(
        "third",
        nargs="?",
        help="Output image path when rule_y is supplied",
    )
    args = parser.parse_args()

    if args.third is None:
        rule_y = None
        output = Path(args.second)
    else:
        rule_y = int(args.second)
        output = Path(args.third)

    gray = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Không đọc được ảnh: {args.input}")

    result = save_hexagram_image(gray, rule_y, output)
    if result is None:
        raise SystemExit("Không tìm thấy quẻ trong vùng ảnh đã chọn")
    print(f"Đã lưu ảnh quẻ: {result}")


if __name__ == "__main__":
    main()
