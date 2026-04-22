import re
from typing import Optional, Tuple
import cv2
import numpy as np
import pytesseract


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def normalize_text(text: str) -> str:
    text = text.strip().replace(",", ".")
    text = re.sub(r"\s+", " ", text)
    return text


def safe_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", ".")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if cleaned.count(".") > 1:
        first = cleaned.find(".")
        cleaned = cleaned[:first + 1] + cleaned[first + 1:].replace(".", "")
    try:
        return float(cleaned)
    except Exception:
        return None


def ocr_text(img, psm: int = 7, whitelist: Optional[str] = None, lang: str = "eng") -> Tuple[str, float]:
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=pytesseract.Output.DICT,
        lang=lang
    )

    texts = []
    confs = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt = (txt or "").strip()
        try:
            conf = float(conf)
        except Exception:
            conf = -1
        if txt and conf >= 0:
            texts.append(txt)
            confs.append(conf)

    full_text = normalize_text(" ".join(texts))
    mean_conf = float(np.mean(confs) / 100.0) if confs else 0.0
    return full_text, mean_conf
