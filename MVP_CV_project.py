import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np


DB_NAME = "recognition.db"
ALLOWED_MODES = {"auto", "all_text", "display_value"}


DISPLAY_VALUE_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")
UNIT_RE = re.compile(r"^(A|V|W|KW|MW|HZ|OHM|Ω|KWH|MWH|KV|MA|MV|PA|BAR|M3|L|KG|G|C|°C|F)$", re.IGNORECASE)
LABEL_RE = re.compile(r"^[A-ZА-Я0-9\-_/]{3,}$")


@dataclass
class OCRItem:
    text: str
    confidence: float
    bbox: List[List[int]]


@dataclass
class Candidate:
    text: str
    normalized_text: str
    confidence: float
    bbox: List[List[int]]
    category: str
    score: float
    features: Dict[str, Any]


@dataclass
class OCRResult:
    source_path: str
    mode: str
    raw_text: str
    normalized_text: str
    confidence: float
    items: List[OCRItem]
    candidates: List[Candidate]
    saved_at: str
    meta: Dict[str, Any]


class OCRPipeline:
    def __init__(self, languages: List[str], use_gpu: bool = False):
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def run(
        self,
        image_path: str,
        mode: str = "auto",
        crop: Optional[Tuple[int, int, int, int]] = None,
        min_confidence: float = 0.15,
        save_debug: bool = False,
        debug_dir: Optional[str] = None,
    ) -> OCRResult:
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Allowed: {sorted(ALLOWED_MODES)}")

        src = cv2.imread(image_path)
        if src is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = self._apply_crop(src, crop)

        if mode == "auto":
            mode = "display_value"

        text_variants = self._build_variants(image)
        all_candidates: List[Candidate] = []
        all_items: List[OCRItem] = []

        for variant_name, variant_img in text_variants:
            items = self._read_text(variant_img, min_confidence=min_confidence)
            if save_debug and debug_dir:
                self._save_debug_image(debug_dir, image_path, f"variant_{variant_name}", variant_img)

            # перерасчёт bbox из variant в координаты исходного image
            mapped_items = self._map_items_to_original(items, image.shape[:2], variant_img.shape[:2])
            all_items.extend(mapped_items)

            variant_candidates = self._build_candidates(image, mapped_items, variant_name)
            all_candidates.extend(variant_candidates)

        merged_candidates = self._deduplicate_candidates(all_candidates)
        merged_candidates.sort(key=lambda c: c.score, reverse=True)

        best_display = self._select_best_display_candidate(merged_candidates)
        best_any = merged_candidates[0] if merged_candidates else None

        if mode == "display_value":
            chosen = best_display or best_any
        else:
            chosen = best_any

        raw_text = chosen.text if chosen else ""
        normalized_text = chosen.normalized_text if chosen else ""
        confidence = float(chosen.confidence) if chosen else 0.0

        meta = {
            "image_shape": list(image.shape),
            "items_count": len(all_items),
            "candidates_count": len(merged_candidates),
            "best_display_found": best_display is not None,
        }

        return OCRResult(
            source_path=str(Path(image_path).resolve()),
            mode=mode,
            raw_text=raw_text,
            normalized_text=normalized_text,
            confidence=confidence,
            items=all_items,
            candidates=merged_candidates,
            saved_at=datetime.utcnow().isoformat(),
            meta=meta,
        )

    def _apply_crop(self, image: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if not crop:
            return image
        x, y, w, h = crop
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        return image[y:y + h, x:x + w].copy()

    def _build_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        denoised = cv2.bilateralFilter(enlarged, 9, 75, 75)

        otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv_otsu = cv2.bitwise_not(otsu)

        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        image_big = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(image_big, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 30, 40]), np.array([95, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([179, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        return [
            ("gray", gray),
            ("enlarged", enlarged),
            ("otsu", otsu),
            ("inv_otsu", inv_otsu),
            ("adaptive", adaptive),
            ("green_mask", green_mask),
            ("red_mask", red_mask),
        ]

    def _read_text(self, image: np.ndarray, min_confidence: float) -> List[OCRItem]:
        result = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="0123456789.,-ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯabcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя/%Ω°",
        )

        items: List[OCRItem] = []
        for bbox, text, conf in result:
            conf = float(conf)
            text = text.strip()
            if conf < min_confidence or not text:
                continue
            clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
            items.append(OCRItem(text=text, confidence=conf, bbox=clean_bbox))

        items.sort(key=self._bbox_sort_key)
        return items

    def _map_items_to_original(
        self,
        items: List[OCRItem],
        original_shape: Tuple[int, int],
        variant_shape: Tuple[int, int],
    ) -> List[OCRItem]:
        orig_h, orig_w = original_shape
        var_h, var_w = variant_shape

        scale_x = orig_w / max(var_w, 1)
        scale_y = orig_h / max(var_h, 1)

        mapped: List[OCRItem] = []
        for item in items:
            bbox = [
                [int(p[0] * scale_x), int(p[1] * scale_y)]
                for p in item.bbox
            ]
            mapped.append(OCRItem(text=item.text, confidence=item.confidence, bbox=bbox))
        return mapped

    def _build_candidates(
        self,
        image: np.ndarray,
        items: List[OCRItem],
        variant_name: str,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for item in items:
            bbox = item.bbox
            x1, y1, x2, y2 = self._bbox_rect(bbox)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh
            rel_area = area / float(w * h)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            roi = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            roi_mean = float(np.mean(roi)) if roi.size else 255.0
            roi_std = float(np.std(roi)) if roi.size else 0.0

            normalized = self._normalize_text(item.text)

            is_numeric_like = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", normalized))
            has_digit = bool(re.search(r"\d", normalized))
            only_symbols = bool(re.fullmatch(r"[-.]+", normalized))
            is_unit = bool(UNIT_RE.fullmatch(normalized.upper()))
            is_label = bool(LABEL_RE.fullmatch(normalized.upper()))

            aspect = bw / float(max(bh, 1))
            dark_background_score = max(0.0, (160.0 - roi_mean) / 160.0)
            contrast_score = min(roi_std / 80.0, 1.0)
            size_score = min(rel_area / 0.03, 1.0)

            seven_segment_hint = self._seven_segment_hint(image, x1, y1, x2, y2)

            display_score = 0.0
            if is_numeric_like:
                display_score += 1.2
            if has_digit:
                display_score += 0.35
            if only_symbols:
                display_score -= 0.8
            if is_unit:
                display_score -= 0.7
            if is_label:
                display_score -= 0.4

            display_score += dark_background_score * 0.7
            display_score += contrast_score * 0.4
            display_score += size_score * 0.5
            display_score += seven_segment_hint * 0.8
            display_score += item.confidence * 0.6

            # предпочтение более центральным крупным числам
            center_bonus = 1.0 - abs(cx - (w / 2.0)) / max(w / 2.0, 1.0)
            display_score += center_bonus * 0.2

            if is_numeric_like and len(normalized) >= 2:
                category = "display_value"
            elif is_unit:
                category = "unit"
            elif is_label:
                category = "label"
            else:
                category = "unknown"

            if category == "unit":
                final_score = item.confidence + 0.2
            elif category == "label":
                final_score = item.confidence + 0.1
            else:
                final_score = display_score

            candidates.append(
                Candidate(
                    text=item.text,
                    normalized_text=normalized,
                    confidence=item.confidence,
                    bbox=bbox,
                    category=category,
                    score=float(final_score),
                    features={
                        "variant": variant_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": bw,
                        "height": bh,
                        "area": area,
                        "rel_area": rel_area,
                        "aspect": aspect,
                        "roi_mean": roi_mean,
                        "roi_std": roi_std,
                        "dark_background_score": dark_background_score,
                        "contrast_score": contrast_score,
                        "size_score": size_score,
                        "seven_segment_hint": seven_segment_hint,
                        "is_numeric_like": is_numeric_like,
                        "is_unit": is_unit,
                        "is_label": is_label,
                    },
                )
            )

        return candidates

    def _seven_segment_hint(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        roi = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th == 255) > 0.5:
            th = cv2.bitwise_not(th)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        rectish = 0
        total = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 6:
                continue
            total += 1
            ratio = w / float(max(h, 1))
            if 0.15 <= ratio <= 6.0:
                rectish += 1

        if total == 0:
            return 0.0

        hint = rectish / total
        return float(min(hint, 1.0))

    def _deduplicate_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        result: List[Candidate] = []

        for cand in candidates:
            duplicate = False
            for kept in result:
                if self._same_candidate(cand, kept):
                    duplicate = True
                    break
            if not duplicate:
                result.append(cand)

        return result

    def _same_candidate(self, a: Candidate, b: Candidate) -> bool:
        if a.normalized_text != b.normalized_text:
            return False

        ax1, ay1, ax2, ay2 = self._bbox_rect(a.bbox)
        bx1, by1, bx2, by2 = self._bbox_rect(b.bbox)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        b_area = max((bx2 - bx1) * (by2 - by1), 1)

        iou_like = inter_area / float(min(a_area, b_area))
        return iou_like > 0.5

    def _select_best_display_candidate(self, candidates: List[Candidate]) -> Optional[Candidate]:
        display_candidates = [c for c in candidates if c.category == "display_value"]
        if not display_candidates:
            return None

        display_candidates.sort(key=lambda c: c.score, reverse=True)
        return display_candidates[0]

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace(",", ".")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^0-9A-Za-zА-Яа-я.\-/%Ω°]", "", text)

        if text.count(".") > 1:
            parts = text.split(".")
            text = parts[0] + "." + "".join(parts[1:])

        if text.count("-") > 1:
            text = "-" + text.replace("-", "")
        elif "-" in text and not text.startswith("-"):
            text = "-" + text.replace("-", "")

        return text

    def _bbox_sort_key(self, item: OCRItem) -> Tuple[int, float]:
        xs = [p[0] for p in item.bbox]
        ys = [p[1] for p in item.bbox]
        cx = float(sum(xs) / len(xs))
        cy = float(sum(ys) / len(ys))
        return int(cy // 30), cx

    def _bbox_rect(self, bbox: List[List[int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _save_debug_image(self, debug_dir: str, image_path: str, variant_name: str, image: np.ndarray) -> None:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        out_path = out_dir / f"{stem}_{variant_name}.png"
        cv2.imwrite(str(out_path), image)


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                mode TEXT NOT NULL,
                raw_text TEXT,
                normalized_text TEXT,
                confidence REAL NOT NULL,
                items_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                saved_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def save(self, result: OCRResult) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ocr_results (
                source_path, mode, raw_text, normalized_text,
                confidence, items_json, candidates_json, meta_json, saved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.source_path,
                result.mode,
                result.raw_text,
                result.normalized_text,
                result.confidence,
                json.dumps([asdict(item) for item in result.items], ensure_ascii=False),
                json.dumps([asdict(c) for c in result.candidates], ensure_ascii=False),
                json.dumps(result.meta, ensure_ascii=False),
                result.saved_at,
            ),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return int(row_id)


def parse_crop(crop_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not crop_str:
        return None
    parts = [p.strip() for p in crop_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop must be in format x,y,w,h")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="General OCR for instrument displays, meters and digital panels."
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--mode", choices=sorted(ALLOWED_MODES), default="display_value")
    parser.add_argument("--db", default=DB_NAME, help="SQLite database path")
    parser.add_argument("--lang", nargs="+", default=["en", "ru"], help="EasyOCR languages")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--crop", help="Crop region x,y,w,h before OCR")
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    crop = parse_crop(args.crop)

    pipeline = OCRPipeline(languages=args.lang, use_gpu=args.gpu)
    result = pipeline.run(
        image_path=args.image,
        mode=args.mode,
        crop=crop,
        min_confidence=args.min_confidence,
        save_debug=args.save_debug,
        debug_dir=args.debug_dir,
    )

    store = SQLiteStore(args.db)
    row_id = store.save(result)

    payload = {
        "db_row_id": row_id,
        "source_path": result.source_path,
        "mode": result.mode,
        "raw_text": result.raw_text,
        "normalized_text": result.normalized_text,
        "confidence": round(result.confidence, 4),
        "saved_at": result.saved_at,
        "meta": result.meta,
        "items": [asdict(item) for item in result.items],
        "candidates": [asdict(c) for c in result.candidates],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"[OK] saved to DB row id={row_id}")
    print(f"source_path     : {payload['source_path']}")
    print(f"mode            : {payload['mode']}")
    print(f"raw_text        : {payload['raw_text']}")
    print(f"normalized_text : {payload['normalized_text']}")
    print(f"confidence      : {payload['confidence']}")
    print(f"saved_at        : {payload['saved_at']}")
    print(f"meta            : {json.dumps(payload['meta'], ensure_ascii=False)}")

    print("\nTop candidates:")
    for idx, cand in enumerate(result.candidates[:10], start=1):
        print(
            f"{idx}. text={cand.text!r} | norm={cand.normalized_text!r} | "
            f"cat={cand.category} | conf={cand.confidence:.4f} | score={cand.score:.4f}"
        )


if __name__ == "__main__":
    main()