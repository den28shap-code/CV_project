from typing import Dict, Any, Optional
import re
import cv2
import numpy as np

from models import DeviceResult, ChannelResult
from utils import four_point_transform, safe_float, ocr_text
from parsers.seven_segment import decode_7segment_display


def find_display_roi(image: np.ndarray) -> np.ndarray:
    """
    Пытается найти крупный прямоугольный дисплей.
    Если не получается — возвращает центральную часть кадра.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    h, w = image.shape[:2]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(c)
        if area <= best_area:
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        ratio = cw / max(ch, 1)

        if area > (w * h) * 0.05 and 1.15 < ratio < 10.0:
            best = approx.reshape(4, 2)
            best_area = area

    if best is not None:
        return four_point_transform(image, best)

    x1 = int(w * 0.15)
    y1 = int(h * 0.18)
    x2 = int(w * 0.85)
    y2 = int(h * 0.82)
    return image[y1:y2, x1:x2]


def classify_display_type(roi: np.ndarray) -> str:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bright_ratio = float(np.mean(v > 180))
    saturated_ratio = float(np.mean(s > 100))
    mean_gray = float(np.mean(gray))

    if bright_ratio > 0.08 and saturated_ratio > 0.08:
        return "7_segment_led"
    if mean_gray < 90 and bright_ratio > 0.04:
        return "led_or_vfd"
    if 90 <= mean_gray <= 200:
        return "lcd_or_mechanical"
    return "text_display"


def detect_unit_from_image_text(full_img: np.ndarray) -> Optional[str]:
    gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    txt, _ = ocr_text(gray, psm=6, lang="eng+rus")
    low = txt.lower()

    if "mpa" in low:
        return "MPa"
    if "kv" in low:
        return "kV"
    if "kwh" in low or "kw·h" in low or "kw h" in low:
        return "kWh"
    if re.search(r"(^|[^a-z])v([^a-z]|$)", low):
        return "V"
    if re.search(r"(^|[^a-z])a([^a-z]|$)", low):
        return "A"
    return None


def parse_with_ocr_fallback(roi: np.ndarray):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    whitelist = (
        "0123456789.-_"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    )

    txt1, conf1 = ocr_text(th, psm=7, whitelist=whitelist, lang="eng+rus")
    txt2, conf2 = ocr_text(gray, psm=7, whitelist=whitelist, lang="eng+rus")

    if conf1 >= conf2:
        return txt1, conf1
    return txt2, conf2


def try_parse_dual_channel(roi: np.ndarray):
    """
    Пытается прочитать верх/низ как два независимых канала.
    Полезно для V/A индикаторов.
    """
    h, w = roi.shape[:2]
    if h < 50 or w < 80:
        return None

    top = roi[: h // 2, :]
    bottom = roi[h // 2 :, :]

    top_text, _, top_conf = decode_7segment_display(top)
    bottom_text, _, bottom_conf = decode_7segment_display(bottom)

    non_empty = sum(bool(x) for x in [top_text, bottom_text])
    if non_empty == 0:
        return None

    channels = [
        ChannelResult(
            label="channel_1",
            value=safe_float(top_text),
            unit=None,
            raw_reading=top_text or None,
            confidence=top_conf,
        ).to_dict(),
        ChannelResult(
            label="channel_2",
            value=safe_float(bottom_text),
            unit=None,
            raw_reading=bottom_text or None,
            confidence=bottom_conf,
        ).to_dict(),
    ]

    return DeviceResult(
        image_path="",
        device_type="multi_channel_display",
        display_type="7_segment_led_dual",
        detected_text=None,
        value=None,
        unit=None,
        raw_reading=None,
        decimal_point=None,
        confidence=max(top_conf, bottom_conf),
        status="ok",
        channels=channels,
        extra=None,
    )


def analyze_instrument(image_path: str) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        return {
            "image_path": image_path,
            "status": "error",
            "error": "Cannot read image",
        }

    roi = find_display_roi(image)
    display_type = classify_display_type(roi)
    unit_hint = detect_unit_from_image_text(image)

    # 1) dual-channel
    dual_result = try_parse_dual_channel(roi)
    if dual_result is not None and dual_result.confidence > 0.45:
        dual_result.image_path = image_path
        dual_result.extra = {"detected_unit_hint": unit_hint}
        return dual_result.to_dict()

    # 2) seven-segment decode
    seg_text, has_dp, seg_conf = decode_7segment_display(roi)
    if seg_text and seg_conf > 0.45:
        result = DeviceResult(
            image_path=image_path,
            device_type="digital_instrument",
            display_type=display_type,
            detected_text=seg_text,
            value=safe_float(seg_text),
            unit=unit_hint,
            raw_reading=seg_text,
            decimal_point=has_dp,
            confidence=seg_conf,
            status="ok",
            channels=None,
            extra=None,
        )
        return result.to_dict()

    # 3) OCR fallback
    txt, conf = parse_with_ocr_fallback(roi)
    value = safe_float(txt)
    device_type = "text_display" if value is None else "meter_or_indicator"

    result = DeviceResult(
        image_path=image_path,
        device_type=device_type,
        display_type=display_type,
        detected_text=txt or None,
        value=value,
        unit=unit_hint,
        raw_reading=txt or None,
        decimal_point=("." in txt) if txt else None,
        confidence=conf,
        status="ok" if txt else "uncertain",
        channels=None,
        extra=None,
    )
    return result.to_dict()
