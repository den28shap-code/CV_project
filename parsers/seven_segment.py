from typing import List, Tuple
import cv2
import numpy as np


SEGMENT_MAP = {
    (1, 1, 1, 0, 1, 1, 1): "0",
    (0, 0, 1, 0, 0, 1, 0): "1",
    (1, 0, 1, 1, 1, 0, 1): "2",
    (1, 0, 1, 1, 0, 1, 1): "3",
    (0, 1, 1, 1, 0, 1, 0): "4",
    (1, 1, 0, 1, 0, 1, 1): "5",
    (1, 1, 0, 1, 1, 1, 1): "6",
    (1, 0, 1, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}


def preprocess_segment_display(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    mask = cv2.inRange(v, 140, 255)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def extract_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = binary.shape[:2]

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (w * h) * 0.005:
            continue
        if ch < h * 0.25:
            continue
        if 0.12 < cw / max(ch, 1) < 1.25:
            boxes.append((x, y, cw, ch))

    boxes = sorted(boxes, key=lambda b: b[0])

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w1, h1 = box
        px, py, pw, ph = merged[-1]

        if x <= px + pw + 5:
            nx = min(px, x)
            ny = min(py, y)
            nw = max(px + pw, x + w1) - nx
            nh = max(py + ph, y + h1) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    return merged


def decode_single_7seg(digit_img: np.ndarray):
    h, w = digit_img.shape[:2]

    segments = [
        ((int(w * 0.2), 0), (int(w * 0.8), int(h * 0.15))),             # a
        ((0, int(h * 0.1)), (int(w * 0.2), int(h * 0.5))),              # f
        ((int(w * 0.8), int(h * 0.1)), (w, int(h * 0.5))),              # b
        ((int(w * 0.2), int(h * 0.42)), (int(w * 0.8), int(h * 0.58))), # g
        ((0, int(h * 0.5)), (int(w * 0.2), int(h * 0.9))),              # e
        ((int(w * 0.8), int(h * 0.5)), (w, int(h * 0.9))),              # c
        ((int(w * 0.2), int(h * 0.85)), (int(w * 0.8), h)),             # d
    ]

    on = []
    fill_scores = []

    for (x1, y1), (x2, y2) in segments:
        seg = digit_img[y1:y2, x1:x2]
        if seg.size == 0:
            on.append(0)
            fill_scores.append(0.0)
            continue

        ratio = float(np.mean(seg > 0))
        fill_scores.append(ratio)
        on.append(1 if ratio > 0.35 else 0)

    dp_area = digit_img[int(h * 0.75):h, int(w * 0.8):w]
    dp_ratio = float(np.mean(dp_area > 0)) if dp_area.size else 0.0
    decimal_point = dp_ratio > 0.18

    char = SEGMENT_MAP.get(tuple(on), "?")
    confidence = min(1.0, float(np.mean(fill_scores)) * 1.8)
    return char, decimal_point, confidence


def decode_7segment_display(roi: np.ndarray):
    binary = preprocess_segment_display(roi)
    boxes = extract_digit_boxes(binary)

    if not boxes:
        return "", False, 0.0

    chars = []
    any_dp = False
    confs = []

    for x, y, w, h in boxes:
        digit = binary[y:y + h, x:x + w]
        char, dp, conf = decode_single_7seg(digit)
        if char != "?":
            chars.append(char)
            confs.append(conf)
            if dp:
                any_dp = True

    text = "".join(chars)
    confidence = float(np.mean(confs)) if confs else 0.0

    if any_dp and len(text) >= 2:
        text = text[:-1] + "." + text[-1]

    return text, any_dp, confidence
