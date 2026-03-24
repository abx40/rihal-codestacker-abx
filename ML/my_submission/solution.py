"""
DocFusion challenge solution.

Key capabilities:
- Interface-compliant DocFusionSolution for judge harness.
- OCR-based extraction for vendor/date/total with multi-pass tesseract.
- Extraction-focused local `analyze_image()` helper for the UI and debugging.
"""

from __future__ import annotations

import json
import io
import math
import os
import pickle
import random
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier  # type: ignore

    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False


DATE_CANDIDATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
DATE_TEXTUAL_RE = re.compile(
    r"\b(?:"
    r"(?:\d{1,2}\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s*,?\s*\d{2,4})|"
    r"(?:(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2},?\s*\d{2,4})"
    r")\b",
    re.I,
)
AMOUNT_RE = re.compile(
    r"(?<!\d)(-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|-?\d+(?:[.,]\d{2})|-?\d{2,})(?!\d)"
)
TOTAL_HINT_RE = re.compile(r"(total|amount\s*due|grand\s*total|net\s*total)", re.I)
TOTAL_LINE_HINT_RE = re.compile(
    r"(?:\btotal(?:\s+sales|\s+price)?\b|\bamount\s*due\b|\bgrand\s*total\b|\bnet\s*total\b|\bamt\b|\bjumlah\b)",
    re.I,
)
TOTAL_EXCLUSION_RE = re.compile(
    r"(subtotal|sub[- ]?total|tax|vat|discount|cash|change|paid|balance|rounding|qty|quantity|item)",
    re.I,
)
PAYMENT_HINT_RE = re.compile(
    r"(cash|change|kembali|tunai|paid|payment|credit|debit|visa|master|card|charge|balance)",
    re.I,
)
TOTAL_COMPONENT_RE = re.compile(
    r"(parking\s*fee|fee\b|add\s*gst|gst\b|sst\b|tax\b|vat\b|rounding|adjustment|discount|service|rate\s*incl)",
    re.I,
)
TOTAL_NOISE_RE = re.compile(
    r"(receipt\s*number|invoice\s*(?:no|number)?|gst\s*(?:id|no)?|roc\s*no|approval\s*code|"
    r"cashier|entry\s*time|exit\s*time|time\b|table\b|tel\b|fax\b|phone\b|park[- ]?our|"
    r"company\s*no|bill\s*to|email\b)",
    re.I,
)
VENDOR_LABEL_RE = re.compile(
    r"(?:vendor|merchant|store|seller|supplier|from)\s*[:\-]\s*(.+)$",
    re.I,
)
VENDOR_COMPANY_RE = re.compile(
    r"\b(?:sdn|bhd|s/?b|enterprise|trading|marketing|corporation|corp|company|co\b|network|"
    r"stationery|books|hardware|restaurant|parking|mart|market|pharmacy|services?|shop|"
    r"group|holdings?|kopitiam|kopitam|coffee|electrical|global)\b",
    re.I,
)
VENDOR_NOISE_RE = re.compile(
    r"(invoice|receipt|total|tax|gst|tel\b|fax\b|cashier|date\b|bill\s*to|description|qty\b|"
    r"price\b|amount\b|thank|approval|entry|exit|payment|change\b|email\b|copy\b)",
    re.I,
)
VENDOR_ADDRESS_RE = re.compile(
    r"(jalan|jln\b|lot\b|taman\b|selangor|kuala|klang|petaling|ground\s+floor|floor\b|"
    r"metro|perdana|barat|timur|kampung|no\.\b|gst|tel\b|fax\b|email\b|receipt|invoice)",
    re.I,
)
VENDOR_ID_RE = re.compile(r"\(\s*[A-Za-z0-9/-]{4,}\s*\)|\b[A-Z]?\d{5,}[A-Z/-]*\b", re.I)
SUBTOTAL_HINT_RE = re.compile(r"\bsub[- ]?total\b", re.I)
TAX_HINT_RE = re.compile(r"\b(?:tax|vat|gst|sst)\b", re.I)
ITEM_COUNT_RE = re.compile(r"\b(?:items?|qty|quantity)\s*[:#-]?\s*(\d{1,4})\b", re.I)
ROUNDING_HINT_RE = re.compile(r"\bround(?:ing|ed)?(?:\s*adjust(?:ment)?)?\b", re.I)
DISCOUNT_HINT_RE = re.compile(r"\b(?:discount|disc\b|rebate|coupon|voucher|promo)\b", re.I)
SERVICE_HINT_RE = re.compile(r"\b(?:service(?:\s*charge)?|svc(?:\s*chg)?)\b", re.I)
INCLUSIVE_HINT_RE = re.compile(r"\bincl\w*\b", re.I)
SUBTOTAL_EX_TAX_HINT_RE = re.compile(r"\b(?:before|ex\w{2,10}|without)\b", re.I)
SIGNED_AMOUNT_RE = AMOUNT_RE


@dataclass
class OCRLine:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    conf: float


@dataclass
class ReceiptObservation:
    vendor: str | None
    date_raw: str | None
    date: str | None
    total_raw: str | None
    total: str | None
    total_amount: float | None
    text: str
    ocr_lines: list[OCRLine]
    image_path: Path


OCR_QUALITY_FEATURE_NAMES = (
    "ocr_alpha_ratio",
    "ocr_non_alnum_ratio",
    "ocr_noisy_token_ratio",
    "ocr_long_token_ratio",
    "ocr_repeat_sequence_ratio",
    "ocr_avg_token_len_norm",
)

IMAGE_FORENSICS_FEATURE_NAMES = (
    "img_gray_mean",
    "img_gray_std",
    "img_sharpness",
    "img_edge_density",
    "img_blockiness",
    "img_color_std_mean",
    "img_color_std_gap",
    "img_entropy_norm",
)

ELA_FEATURE_NAMES = (
    "ela_mean",
    "ela_max",
    "ela_std",
    "ela_high_ratio",
)

PATCH_FEATURE_NAMES = (
    "patch_gray_mean",
    "patch_gray_std",
    "patch_sharpness",
    "patch_edge_density",
    "patch_blockiness",
    "patch_color_std_mean",
    "patch_color_std_gap",
    "patch_entropy_norm",
    "patch_area_ratio",
    "patch_aspect_ratio",
)

CURRENT_ANOMALY_PIPELINE = "feature_ensemble_v2_math_profile"
CURRENT_ANOMALY_VARIANT = "lgbm_base_scale_x2"
DEFAULT_ML_THRESHOLD = 0.30
MATH_PROFILE_SAMPLE_PER_SOURCE = 140


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _resolve_split_dataset_root(data_root: Path, split_name: str) -> tuple[Path, Path]:
    direct_path = data_root / f"{split_name}.jsonl"
    if direct_path.exists():
        return data_root, direct_path

    nested_root = data_root / split_name
    nested_path = nested_root / f"{split_name}.jsonl"
    if nested_path.exists():
        return nested_root, nested_path

    return data_root, direct_path


def _normalize_vendor(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", value).strip().upper()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or None


def _safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = AMOUNT_RE.search(str(value))
    if not match:
        return None
    token = re.sub(r"[^0-9,.\-]", "", match.group(0))
    if not token:
        return None

    if "," in token and "." in token:
        if token.rfind(".") > token.rfind(","):
            token = token.replace(",", "")
        else:
            token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        head, tail = token.rsplit(",", 1)
        if len(tail) == 2:
            token = head.replace(",", "") + "." + tail
        else:
            token = head.replace(",", "") + tail
    elif "." in token:
        head, tail = token.rsplit(".", 1)
        if len(tail) == 2:
            token = head.replace(".", "") + "." + tail if "." in head else token
        elif len(tail) == 3 and "." in head:
            token = head.replace(".", "") + tail
    try:
        return float(token)
    except ValueError:
        return None


def _normalize_total(value: str | float | int | None) -> str | None:
    amount = _safe_float(value)
    if amount is None:
        return None
    return f"{amount:.2f}"


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None

    text = re.sub(r"\s+", " ", value.strip())
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y %m %d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d %m %Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m.%d.%Y",
        "%m %d %Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%d %m %y",
        "%m-%d-%y",
        "%m/%d/%y",
        "%m %d %y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d %b %y",
        "%d %B %y",
        "%b %d %y",
        "%B %d %y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_date_parts(value: str | None) -> tuple[int, int, int]:
    normalized = _normalize_date(value) if value else None
    if not normalized:
        return 0, 0, 0
    try:
        year, month, day = normalized.split("-", 2)
        return int(year), int(month), int(day)
    except Exception:
        return 0, 0, 0


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(variance)
    if std < 1e-6:
        std = 1.0
    return mean, std


def _classification_counts(
    scores: list[float],
    targets: list[int],
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for score, target in zip(scores, targets):
        prediction = int(score >= threshold)
        if prediction == 1 and target == 1:
            tp += 1
        elif prediction == 1 and target == 0:
            fp += 1
        elif prediction == 0 and target == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _balanced_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return 0.5 * (tpr + tnr)


def _f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _extract_from_fields(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    fields = record.get("fields")
    if not isinstance(fields, dict):
        return None, None, None
    vendor = fields.get("vendor")
    date = fields.get("date")
    total = fields.get("total")
    if vendor is not None:
        vendor = str(vendor).strip() or None
    if date is not None:
        date = str(date).strip() or None
    if total is not None:
        total = str(total).strip() or None
    return vendor, date, total


def _is_forgery_supervision_record(record: dict[str, Any]) -> bool:
    source = str(record.get("source", "")).strip().lower()
    if source.startswith("find_it_again"):
        return True
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    regions = label.get("forgery_regions")
    return isinstance(regions, list) and len(regions) > 0


def _anomaly_training_records(train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [
        record
        for record in train_records
        if isinstance(record.get("label"), dict) and "is_forged" in record["label"]
    ]
    if not labeled:
        return []

    forgery_focused = [
        record
        for record in labeled
        if _is_forgery_supervision_record(record)
    ]
    forgery_focused_classes = {
        int(bool(record.get("label", {}).get("is_forged", 0)))
        for record in forgery_focused
    }

    if len(forgery_focused) >= 40 and len(forgery_focused_classes) >= 2:
        return forgery_focused
    return labeled


def _run_tesseract(image_path: Path, psm: int = 6, tsv: bool = False, preprocess: bool = False) -> str:
    if not image_path.exists():
        return ""

    run_path = image_path
    temp_path: Path | None = None
    if preprocess and PIL_AVAILABLE and ImageOps is not None and ImageEnhance is not None and ImageFilter is not None:
        try:
            with Image.open(image_path) as pil_img:
                gray = pil_img.convert("L")
                boosted = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.8)
                prepared = boosted.filter(ImageFilter.SHARPEN)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                prepared.save(temp_path)
                run_path = temp_path
        except Exception:
            run_path = image_path
            temp_path = None

    cmd = [
        "tesseract",
        str(run_path),
        "stdout",
        "--psm",
        str(psm),
        "--oem",
        "1",
        "-l",
        "eng",
    ]
    if tsv:
        cmd.append("tsv")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout or ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _contains_date(text: str) -> bool:
    return bool(DATE_CANDIDATE_RE.search(text) or DATE_TEXTUAL_RE.search(text))


def _contains_amount(text: str) -> bool:
    return bool(AMOUNT_RE.search(text))


def _merge_texts(texts: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for text in texts:
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line.strip())
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return "\n".join(merged)


def _ocr_text(image_path: Path) -> str:
    primary = _run_tesseract(image_path, psm=6)
    if primary and _contains_amount(primary) and _contains_date(primary):
        return primary

    secondary = _run_tesseract(image_path, psm=11)
    if secondary and (not _contains_amount(primary) or not _contains_date(primary)):
        tertiary = _run_tesseract(image_path, psm=4)
        return _merge_texts([primary, secondary, tertiary])
    merged = _merge_texts([primary, secondary])
    if merged.strip():
        return merged

    # Fallback for washed-out/low-contrast images.
    primary_pre = _run_tesseract(image_path, psm=6, preprocess=True)
    if primary_pre and _contains_amount(primary_pre) and _contains_date(primary_pre):
        return primary_pre
    secondary_pre = _run_tesseract(image_path, psm=11, preprocess=True)
    if secondary_pre and (not _contains_amount(primary_pre) or not _contains_date(primary_pre)):
        tertiary_pre = _run_tesseract(image_path, psm=4, preprocess=True)
        return _merge_texts([primary_pre, secondary_pre, tertiary_pre])
    return _merge_texts([primary_pre, secondary_pre])


def _parse_ocr_lines_tsv(tsv: str) -> list[OCRLine]:
    if not tsv.strip():
        return []

    lines_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows = [row for row in tsv.splitlines() if row.strip()]
    if not rows:
        return []

    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        level, _page, block, par, line_no, _word_no, left, top, width, height, conf, text = cols[:12]
        if level != "5":
            continue
        text = text.strip()
        if not text:
            continue
        try:
            conf_f = float(conf)
            left_i = int(left)
            top_i = int(top)
            width_i = int(width)
            height_i = int(height)
        except ValueError:
            continue

        key = (block, par, line_no)
        item = lines_map.get(key)
        if item is None:
            lines_map[key] = {
                "tokens": [text],
                "left": left_i,
                "top": top_i,
                "right": left_i + width_i,
                "bottom": top_i + height_i,
                "conf": conf_f,
                "count": 1,
            }
            continue

        item["tokens"].append(text)
        item["left"] = min(item["left"], left_i)
        item["top"] = min(item["top"], top_i)
        item["right"] = max(item["right"], left_i + width_i)
        item["bottom"] = max(item["bottom"], top_i + height_i)
        item["conf"] += conf_f
        item["count"] += 1

    out: list[OCRLine] = []
    for item in lines_map.values():
        text = " ".join(item["tokens"]).strip()
        if not text:
            continue
        conf = item["conf"] / max(1, item["count"])
        out.append(
            OCRLine(
                text=text,
                left=int(item["left"]),
                top=int(item["top"]),
                right=int(item["right"]),
                bottom=int(item["bottom"]),
                conf=float(conf),
            )
        )
    return out


def _ocr_lines(image_path: Path) -> list[OCRLine]:
    for psm in (6, 11, 4):
        lines = _parse_ocr_lines_tsv(_run_tesseract(image_path, psm=psm, tsv=True))
        if lines:
            return lines

    # Fallback for washed-out/low-contrast images.
    for psm in (6, 11, 4):
        lines = _parse_ocr_lines_tsv(_run_tesseract(image_path, psm=psm, tsv=True, preprocess=True))
        if lines:
            return lines

    return []


def _extract_date_from_text(text: str) -> str | None:
    for match in DATE_CANDIDATE_RE.findall(text):
        normalized = _normalize_date(match)
        if normalized:
            return normalized

    for match in DATE_TEXTUAL_RE.findall(text):
        normalized = _normalize_date(match)
        if normalized:
            return normalized

    compact_matches = re.findall(r"\b\d{4}\s+\d{1,2}\s+\d{1,2}\b", text)
    for match in compact_matches:
        normalized = _normalize_date(match)
        if normalized:
            return normalized
    return None


def _extract_total_from_text(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    scored_candidates: list[tuple[float, float]] = []
    for line in lines:
        tokens = AMOUNT_RE.findall(line)
        if not tokens:
            continue

        line_score = 0.0
        if TOTAL_HINT_RE.search(line):
            line_score += 3.0
        if TOTAL_EXCLUSION_RE.search(line):
            line_score -= 1.5

        for token in tokens:
            amount = _safe_float(token)
            if amount is None:
                continue
            score = line_score + min(max(amount, 0.0) / 1000.0, 0.75)
            if amount > 100000.0 and not TOTAL_HINT_RE.search(line):
                score -= 5.0
            if 1900.0 <= amount <= 2100.0 and _contains_date(line):
                score -= 4.0
            if TOTAL_NOISE_RE.search(line) and amount > 1000.0 and not TOTAL_HINT_RE.search(line):
                score -= 2.5
            if score < -3.5:
                continue
            scored_candidates.append((score, amount))

    if scored_candidates:
        scored_candidates.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return f"{scored_candidates[0][1]:.2f}"

    all_candidates = [_safe_float(token) for token in AMOUNT_RE.findall(text)]
    all_candidates = [value for value in all_candidates if value is not None]
    if all_candidates:
        return f"{max(all_candidates):.2f}"
    return None


def _ordered_ocr_lines(lines: list[OCRLine]) -> list[OCRLine]:
    return sorted(lines, key=lambda line: (line.top, line.left, line.right))


def _line_canvas_size(lines: list[OCRLine]) -> tuple[float, float]:
    if not lines:
        return 1.0, 1.0
    max_right = max(float(line.right) for line in lines)
    max_bottom = max(float(line.bottom) for line in lines)
    return max(1.0, max_right), max(1.0, max_bottom)


def _line_amount_score(
    line: OCRLine,
    context_text: str,
    amount: float,
    line_index: int,
    image_width: float,
    image_height: float,
    amount_count: int,
    token_index: int,
) -> float:
    text = line.text
    context = context_text or text
    alpha_count = len(re.findall(r"[A-Za-z]", text))
    current_has_hint = bool(TOTAL_LINE_HINT_RE.search(text))
    inherited_hint = bool(TOTAL_LINE_HINT_RE.search(context)) and not current_has_hint and alpha_count <= 3
    top_norm = float(line.top) / image_height
    left_norm = float(line.left) / image_width
    right_norm = float(line.right) / image_width

    score = 0.0
    if current_has_hint:
        score += 4.5
    elif inherited_hint:
        score += 3.6
    if TOTAL_EXCLUSION_RE.search(text):
        score -= 1.8
    if PAYMENT_HINT_RE.search(text):
        score -= 4.5
    if TOTAL_NOISE_RE.search(text):
        score -= 2.8
    if TOTAL_COMPONENT_RE.search(text) and not current_has_hint:
        score -= 4.8

    if top_norm >= 0.58:
        score += 1.35
    elif top_norm >= 0.42:
        score += 0.55
    elif not TOTAL_LINE_HINT_RE.search(context):
        score -= 1.3

    if right_norm >= 0.66:
        score += 0.7
    elif left_norm >= 0.45:
        score += 0.35

    if line.conf >= 70.0:
        score += 0.15
    elif line.conf < 25.0:
        score -= 0.35

    if amount_count > 1 and token_index == amount_count - 1:
        score += 0.2
    score += min(max(amount, 0.0) / 1500.0, 0.6)

    if amount > 100000.0 and not TOTAL_LINE_HINT_RE.search(context):
        score -= 5.0
    if amount < 0.0:
        score -= 2.0
    if 1900.0 <= amount <= 2100.0 and _contains_date(context):
        score -= 4.0
    if top_norm < 0.35 and amount > 1000.0 and not TOTAL_LINE_HINT_RE.search(context):
        score -= 1.1
    if line_index >= 18 and top_norm < 0.55:
        score -= 0.25
    return score


def _cash_change_relation_boost(
    current_index: int,
    candidates: list[tuple[float, int, OCRLine, float]],
) -> float:
    current_amount, _idx, current_line, _base_score = candidates[current_index]
    if len(re.findall(r"[A-Za-z]", current_line.text)) > 3:
        return 0.0
    current_top = current_line.top
    best_boost = 0.0
    for pay_amount, _pay_idx, pay_line, _pay_score in candidates:
        if pay_amount <= current_amount:
            continue
        if len(re.findall(r"[A-Za-z]", pay_line.text)) > 3:
            continue
        if pay_line.top + 8 < current_top:
            continue
        tolerance = max(0.15, 0.0025 * pay_amount)
        change_amount = pay_amount - current_amount
        for other_amount, _other_idx, other_line, _other_score in candidates:
            if len(re.findall(r"[A-Za-z]", other_line.text)) > 3:
                continue
            if other_line.top + 8 < current_top:
                continue
            if current_amount <= other_amount:
                continue
            if abs(other_amount - change_amount) <= tolerance:
                boost = 3.2
                if other_line.top >= current_line.top:
                    boost += 0.3
                best_boost = max(best_boost, boost)
    return best_boost


def _extract_total_from_lines_with_score(lines: list[OCRLine]) -> tuple[str | None, float]:
    ordered = _ordered_ocr_lines(lines)
    if not ordered:
        return None, -1.0

    image_width, image_height = _line_canvas_size(ordered)
    candidates: list[tuple[float, int, OCRLine, float]] = []
    for line_index, line in enumerate(ordered):
        tokens = AMOUNT_RE.findall(line.text)
        if not tokens:
            continue
        context_parts = [line.text]
        if line_index > 0:
            prev_line = ordered[line_index - 1]
            if line.top - prev_line.bottom <= max(18, (line.bottom - line.top) * 2):
                context_parts.insert(0, prev_line.text)
        if line_index + 1 < len(ordered):
            next_line = ordered[line_index + 1]
            if next_line.top - line.bottom <= max(18, (line.bottom - line.top) * 2):
                context_parts.append(next_line.text)
        context_text = " ".join(part for part in context_parts if part).strip()
        for token_index, token in enumerate(tokens):
            amount = _safe_float(token)
            if amount is None:
                continue
            base_score = _line_amount_score(
                line=line,
                context_text=context_text,
                amount=amount,
                line_index=line_index,
                image_width=image_width,
                image_height=image_height,
                amount_count=len(tokens),
                token_index=token_index,
            )
            if base_score < -4.0:
                continue
            candidates.append((amount, line_index, line, base_score))

    if not candidates:
        return None, -1.0

    scored: list[tuple[float, float]] = []
    for idx, candidate in enumerate(candidates):
        amount, _line_index, line, base_score = candidate
        score = base_score + _cash_change_relation_boost(idx, candidates)
        if TOTAL_LINE_HINT_RE.search(line.text) and PAYMENT_HINT_RE.search(line.text):
            score -= 1.2
        scored.append((score, amount))

    scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    return f"{scored[0][1]:.2f}", float(scored[0][0])


def _extract_total_from_lines(lines: list[OCRLine]) -> str | None:
    total, _score = _extract_total_from_lines_with_score(lines)
    return total


def _best_vendor_match(text_line: str, known_vendors: list[str]) -> str | None:
    normalized_line = _normalize_vendor(text_line)
    if normalized_line is None:
        return None

    best_fuzzy: tuple[float, str] = (0.0, "")
    for vendor in known_vendors:
        normalized_vendor = _normalize_vendor(vendor)
        if normalized_vendor is None:
            continue
        if normalized_vendor == normalized_line:
            return vendor
        if normalized_vendor in normalized_line or normalized_line in normalized_vendor:
            return vendor
        ratio = SequenceMatcher(a=normalized_vendor, b=normalized_line).ratio()
        if ratio > best_fuzzy[0]:
            best_fuzzy = (ratio, vendor)
    if best_fuzzy[0] >= 0.74:
        return best_fuzzy[1]
    return None


def _vendor_candidate_score(
    candidate: str,
    known_vendors: list[str],
    line_index: int,
) -> tuple[float, str]:
    candidate = VENDOR_ID_RE.sub(" ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -:")
    normalized_candidate = _normalize_vendor(candidate)
    if normalized_candidate is None:
        return -1.0, ""
    if normalized_candidate.isdigit() or len(normalized_candidate) < 3:
        return -1.0, ""
    if VENDOR_NOISE_RE.search(candidate):
        return -1.0, ""
    if VENDOR_ADDRESS_RE.search(candidate) and not VENDOR_COMPANY_RE.search(candidate):
        return -1.0, ""

    alpha_count = len(re.findall(r"[A-Za-z]", candidate))
    if alpha_count < 4:
        return -1.0, ""

    score = max(0.0, 1.8 - 0.12 * line_index)
    if VENDOR_COMPANY_RE.search(candidate):
        score += 1.6
    if any(token in candidate.lower() for token in ("&", "sdn", "bhd", "s/b")):
        score += 0.35

    mapped = _best_vendor_match(candidate, known_vendors)
    if mapped:
        mapped_norm = _normalize_vendor(mapped) or ""
        ratio = SequenceMatcher(a=mapped_norm, b=normalized_candidate).ratio()
        score += 1.0 + ratio
        return score, mapped

    cleaned = re.sub(r"[^A-Za-z0-9/&(). -]+", " ", candidate)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:")
    if len(cleaned) < 3:
        return -1.0, ""
    return score, cleaned


def _extract_vendor_from_lines(lines: list[OCRLine], known_vendors: list[str]) -> str | None:
    ordered = _ordered_ocr_lines(lines)
    if not ordered:
        return None

    image_width, image_height = _line_canvas_size(ordered)
    top_lines = [line for line in ordered if float(line.top) / image_height <= 0.34]
    if not top_lines:
        top_lines = ordered[:12]
    top_lines = top_lines[:12]
    if not top_lines:
        return None

    line_heights = [max(1, line.bottom - line.top) for line in top_lines]
    median_height = float(sorted(line_heights)[len(line_heights) // 2]) if line_heights else 1.0
    scored_candidates: list[tuple[float, str]] = []

    for idx, line in enumerate(top_lines):
        score, resolved = _vendor_candidate_score(line.text, known_vendors, idx)
        if score <= 0.0 or not resolved:
            continue
        top_norm = float(line.top) / image_height
        width_norm = float(line.right - line.left) / image_width
        height_norm = float(line.bottom - line.top) / max(1.0, median_height)
        score += max(0.0, 1.25 - 3.0 * top_norm)
        score += min(0.5, width_norm)
        score += min(0.45, max(0.0, height_norm - 1.0) * 0.35)
        if line.conf >= 55.0:
            score += 0.15
        scored_candidates.append((score, resolved))

    for idx in range(len(top_lines) - 1):
        first = top_lines[idx]
        second = top_lines[idx + 1]
        gap = second.top - first.bottom
        if gap > max(16.0, median_height * 1.8):
            continue
        combined = f"{first.text} {second.text}"
        score, resolved = _vendor_candidate_score(combined, known_vendors, idx)
        if score <= 0.0 or not resolved:
            continue
        top_norm = float(first.top) / image_height
        width_norm = float(max(first.right, second.right) - min(first.left, second.left)) / image_width
        if VENDOR_COMPANY_RE.search(first.text) or VENDOR_COMPANY_RE.search(second.text):
            score += 0.6
        score += max(0.0, 1.2 - 3.0 * top_norm)
        score += min(0.45, width_norm)
        scored_candidates.append((score, resolved))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return scored_candidates[0][1]


def _extract_vendor_from_text(text: str, known_vendors: list[str]) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    scored_candidates: list[tuple[float, str]] = []

    for line in lines[:16]:
        label_match = VENDOR_LABEL_RE.search(line)
        if not label_match:
            continue
        candidate = label_match.group(1).strip()
        score, resolved = _vendor_candidate_score(candidate, known_vendors, 0)
        if score > 0.0 and resolved:
            scored_candidates.append((score + 0.8, resolved))

    top_lines = lines[:12]
    for idx, line in enumerate(top_lines):
        score, resolved = _vendor_candidate_score(line, known_vendors, idx)
        if score > 0.0 and resolved:
            scored_candidates.append((score, resolved))

    for idx, line in enumerate(top_lines[:-1]):
        next_line = top_lines[idx + 1]
        if VENDOR_NOISE_RE.search(line) or VENDOR_NOISE_RE.search(next_line):
            continue
        combined = f"{line} {next_line}"
        score, resolved = _vendor_candidate_score(combined, known_vendors, idx)
        if score > 0.0 and resolved:
            if VENDOR_COMPANY_RE.search(line) or VENDOR_COMPANY_RE.search(next_line):
                score += 0.6
            scored_candidates.append((score, resolved))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return scored_candidates[0][1]


def _extract_with_ocr(
    image_path: Path,
    known_vendors: list[str],
    include_lines: bool = False,
) -> tuple[str | None, str | None, str | None, str, list[OCRLine]]:
    text = _ocr_text(image_path)
    lines = _ocr_lines(image_path)
    vendor = _extract_vendor_from_lines(lines, known_vendors) or _extract_vendor_from_text(text, known_vendors)
    date = _extract_date_from_text(text)
    total_from_lines, total_line_score = _extract_total_from_lines_with_score(lines)
    total_from_text = _extract_total_from_text(text)
    if total_from_lines and (total_line_score >= 5.5 or total_from_text is None):
        total = total_from_lines
    else:
        total = total_from_text or total_from_lines
    return vendor, date, total, text, lines if include_lines or lines else []


def _bbox_from_line(line: OCRLine) -> dict[str, int]:
    return {
        "left": line.left,
        "top": line.top,
        "right": line.right,
        "bottom": line.bottom,
    }


def _find_field_boxes(
    lines: list[OCRLine],
    vendor: str | None,
    date: str | None,
    total: str | None,
) -> dict[str, dict[str, int]]:
    boxes: dict[str, dict[str, int]] = {}

    if vendor:
        target = _normalize_vendor(vendor) or ""
        best_ratio = 0.0
        best_line: OCRLine | None = None
        for line in lines:
            ratio = SequenceMatcher(a=target, b=_normalize_vendor(line.text) or "").ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_line = line
        if best_line and best_ratio >= 0.55:
            boxes["vendor"] = _bbox_from_line(best_line)

    if date:
        for line in lines:
            date_from_line = _extract_date_from_text(line.text)
            if date_from_line and date_from_line == date:
                boxes["date"] = _bbox_from_line(line)
                break

    if total:
        for line in lines:
            total_from_line = _extract_total_from_text(line.text)
            if total_from_line and total_from_line == total:
                boxes["total"] = _bbox_from_line(line)
                break

    return boxes


def _text_features(text: str) -> tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = float(len(lines))
    char_count = float(len(text))
    digit_count = float(sum(ch.isdigit() for ch in text))
    digit_ratio = digit_count / max(1.0, char_count)
    return line_count, char_count, digit_ratio


def _ocr_quality_features(text: str) -> tuple[float, float, float, float, float, float]:
    if not text:
        return 0.0, 1.0, 1.0, 0.0, 0.0, 0.0

    compact = "".join(ch for ch in text if not ch.isspace())
    chars = float(len(compact))
    if chars <= 0:
        return 0.0, 1.0, 1.0, 0.0, 0.0, 0.0

    alpha_count = float(sum(ch.isalpha() for ch in compact))
    alnum_count = float(sum(ch.isalnum() for ch in compact))
    alpha_ratio = alpha_count / chars
    non_alnum_ratio = 1.0 - (alnum_count / chars)

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return alpha_ratio, non_alnum_ratio, 1.0, 0.0, 0.0, 0.0

    noisy_tokens = 0
    long_tokens = 0
    for token in tokens:
        token_len = max(1, len(token))
        alnum = sum(ch.isalnum() for ch in token)
        if (alnum / token_len) < 0.6:
            noisy_tokens += 1
        if token_len >= 18:
            long_tokens += 1

    repeat_sequences = len(re.findall(r"(.)\1{2,}", compact))
    avg_token_len = sum(len(token) for token in tokens) / max(1, len(tokens))

    return (
        alpha_ratio,
        non_alnum_ratio,
        noisy_tokens / max(1.0, float(len(tokens))),
        long_tokens / max(1.0, float(len(tokens))),
        repeat_sequences / max(1.0, float(len(tokens))),
        min(1.0, avg_token_len / 16.0),
    )


def _image_forensics_features(image_path: Path) -> tuple[float, float, float, float, float, float, float, float]:
    if not image_path.exists() or not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        return (0.0,) * len(IMAGE_FORENSICS_FEATURE_NAMES)

    try:
        with Image.open(image_path) as pil_img:
            rgb = pil_img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception:
        return (0.0,) * len(IMAGE_FORENSICS_FEATURE_NAMES)

    if arr.ndim != 3 or arr.shape[2] < 3:
        return (0.0,) * len(IMAGE_FORENSICS_FEATURE_NAMES)

    rgb_arr = arr[:, :, :3]
    gray = np.mean(rgb_arr, axis=2)
    if gray.size == 0:
        return (0.0,) * len(IMAGE_FORENSICS_FEATURE_NAMES)

    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    sharpness = min(1.0, math.log1p(float(np.var(grad_mag))) * 3.5)
    edge_density = float(np.mean(grad_mag > 0.12))

    height, width = gray.shape
    boundary_diffs: list[float] = []
    smooth_diffs: list[float] = []
    if width > 8:
        right_cols = gray[:, 8::8]
        left_cols = gray[:, 7::8]
        pair_count = min(right_cols.shape[1], left_cols.shape[1])
        if pair_count > 0:
            boundary_diffs.append(float(np.mean(np.abs(right_cols[:, :pair_count] - left_cols[:, :pair_count]))))
        smooth_diffs.append(float(np.mean(np.abs(np.diff(gray, axis=1)))))
    if height > 8:
        lower_rows = gray[8::8, :]
        upper_rows = gray[7::8, :]
        pair_count = min(lower_rows.shape[0], upper_rows.shape[0])
        if pair_count > 0:
            boundary_diffs.append(float(np.mean(np.abs(lower_rows[:pair_count, :] - upper_rows[:pair_count, :]))))
        smooth_diffs.append(float(np.mean(np.abs(np.diff(gray, axis=0)))))
    if boundary_diffs and smooth_diffs:
        blockiness = min(5.0, (sum(boundary_diffs) / len(boundary_diffs)) / max(1e-6, sum(smooth_diffs) / len(smooth_diffs)))
    else:
        blockiness = 0.0

    channel_std = np.std(rgb_arr, axis=(0, 1))
    color_std_mean = float(np.mean(channel_std))
    color_std_gap = float(np.max(channel_std) - np.min(channel_std))

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float32)
    hist = hist / max(1.0, float(np.sum(hist)))
    hist_nonzero = hist[hist > 0]
    entropy = float(-(hist_nonzero * np.log(hist_nonzero)).sum())
    entropy_norm = entropy / math.log(256.0)

    return (
        gray_mean,
        gray_std,
        sharpness,
        edge_density,
        blockiness,
        color_std_mean,
        color_std_gap,
        entropy_norm,
    )


def _image_rgb_array(image_path: Path) -> Any | None:
    if not image_path.exists() or not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        return None
    try:
        with Image.open(image_path) as pil_img:
            rgb = pil_img.convert("RGB")
            return np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception:
        return None


def _clip_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if width <= 1 or height <= 1:
        return None
    left = max(0, min(image_width - 1, x))
    top = max(0, min(image_height - 1, y))
    right = max(left + 1, min(image_width, x + width))
    bottom = max(top + 1, min(image_height, y + height))
    if right - left <= 1 or bottom - top <= 1:
        return None
    return left, top, right, bottom


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = float((right - left) * (bottom - top))
    first_area = float((first[2] - first[0]) * (first[3] - first[1]))
    second_area = float((second[2] - second[0]) * (second[3] - second[1]))
    union = max(1e-6, first_area + second_area - inter)
    return inter / union


def _patch_features_from_bbox(
    rgb_arr: Any,
    bbox: tuple[int, int, int, int],
) -> list[float] | None:
    left, top, right, bottom = bbox
    patch = rgb_arr[top:bottom, left:right, :3]
    if patch is None or getattr(patch, "size", 0) == 0:
        return None

    patch_height = patch.shape[0]
    patch_width = patch.shape[1]
    if patch_height < 2 or patch_width < 2:
        return None

    gray = np.mean(patch, axis=2)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    sharpness = min(1.0, math.log1p(float(np.var(grad_mag))) * 3.5)
    edge_density = float(np.mean(grad_mag > 0.12))

    boundary_diffs: list[float] = []
    smooth_diffs: list[float] = []
    if patch_width > 8:
        right_cols = gray[:, 8::8]
        left_cols = gray[:, 7::8]
        pair_count = min(right_cols.shape[1], left_cols.shape[1])
        if pair_count > 0:
            boundary_diffs.append(float(np.mean(np.abs(right_cols[:, :pair_count] - left_cols[:, :pair_count]))))
        smooth_diffs.append(float(np.mean(np.abs(np.diff(gray, axis=1)))))
    if patch_height > 8:
        lower_rows = gray[8::8, :]
        upper_rows = gray[7::8, :]
        pair_count = min(lower_rows.shape[0], upper_rows.shape[0])
        if pair_count > 0:
            boundary_diffs.append(float(np.mean(np.abs(lower_rows[:pair_count, :] - upper_rows[:pair_count, :]))))
        smooth_diffs.append(float(np.mean(np.abs(np.diff(gray, axis=0)))))
    if boundary_diffs and smooth_diffs:
        blockiness = min(5.0, (sum(boundary_diffs) / len(boundary_diffs)) / max(1e-6, sum(smooth_diffs) / len(smooth_diffs)))
    else:
        blockiness = 0.0

    channel_std = np.std(patch, axis=(0, 1))
    color_std_mean = float(np.mean(channel_std))
    color_std_gap = float(np.max(channel_std) - np.min(channel_std))

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float32)
    hist = hist / max(1.0, float(np.sum(hist)))
    hist_nonzero = hist[hist > 0]
    entropy = float(-(hist_nonzero * np.log(hist_nonzero)).sum())
    entropy_norm = entropy / math.log(256.0)

    image_area = float(max(1, rgb_arr.shape[0] * rgb_arr.shape[1]))
    patch_area = float((right - left) * (bottom - top))
    area_ratio = patch_area / image_area
    aspect_ratio = (right - left) / max(1.0, float(bottom - top))

    return [
        gray_mean,
        gray_std,
        sharpness,
        edge_density,
        blockiness,
        color_std_mean,
        color_std_gap,
        entropy_norm,
        area_ratio,
        aspect_ratio,
    ]


def _extract_forgery_region_bboxes(
    record: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    regions = label.get("forgery_regions")
    if not isinstance(regions, list):
        return []
    out: list[tuple[int, int, int, int]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        try:
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            width = int(region.get("width", 0))
            height = int(region.get("height", 0))
        except (TypeError, ValueError):
            continue
        bbox = _clip_bbox(x, y, width, height, image_width, image_height)
        if bbox is not None:
            out.append(bbox)
    return out


def _sample_negative_patch_bboxes(
    image_width: int,
    image_height: int,
    positive_bboxes: list[tuple[int, int, int, int]],
    sample_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    max_tries = sample_count * 50
    tries = 0
    while len(out) < sample_count and tries < max_tries:
        tries += 1
        if positive_bboxes:
            ref = rng.choice(positive_bboxes)
            width = max(6, ref[2] - ref[0])
            height = max(6, ref[3] - ref[1])
        else:
            width = max(14, int(image_width * rng.uniform(0.05, 0.20)))
            height = max(10, int(image_height * rng.uniform(0.01, 0.08)))

        width = min(width, image_width)
        height = min(height, image_height)
        if width <= 1 or height <= 1:
            continue

        x = rng.randint(0, max(0, image_width - width))
        y = rng.randint(0, max(0, image_height - height))
        candidate = (x, y, x + width, y + height)

        if any(_bbox_iou(candidate, box) >= 0.12 for box in positive_bboxes):
            continue
        if any(_bbox_iou(candidate, box) >= 0.30 for box in out):
            continue
        out.append(candidate)
    return out


def _candidate_patch_bboxes_from_ocr_lines(
    lines: list[OCRLine],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, float]] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        lower = text.lower()
        has_amount = bool(AMOUNT_RE.search(text))
        is_total_like = bool(
            TOTAL_HINT_RE.search(lower)
            or any(token in lower for token in ("total", "cash", "change", "gst", "tax", "tendered", "rounded"))
        )
        low_conf = line.conf < 82.0
        if not ((has_amount and not is_total_like) or (low_conf and has_amount)):
            continue

        margin_x = max(2, int((line.right - line.left) * 0.10))
        margin_y = max(2, int((line.bottom - line.top) * 0.30))
        bbox = _clip_bbox(
            x=line.left - margin_x,
            y=line.top - margin_y,
            width=(line.right - line.left) + 2 * margin_x,
            height=(line.bottom - line.top) + 2 * margin_y,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue

        priority = 0.0
        if has_amount and not is_total_like:
            priority += 1.5
        if low_conf:
            priority += (82.0 - line.conf) / 40.0
        candidates.append((bbox[0], bbox[1], bbox[2], bbox[3], priority))

    candidates.sort(key=lambda item: item[4], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom, _priority in candidates:
        candidate = (left, top, right, bottom)
        if any(_bbox_iou(candidate, existing) >= 0.55 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 8:
            break
    return selected


def _dense_patch_bboxes(
    image_width: int,
    image_height: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if image_width <= 2 or image_height <= 2:
        return []

    x_centers = (0.18, 0.38, 0.58, 0.78)
    y_centers = (0.10, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.92)
    window_specs = (
        (0.14, 0.028, 0.62),
        (0.22, 0.045, 0.50),
        (0.30, 0.065, 0.38),
    )

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for width_ratio, height_ratio, base_priority in window_specs:
        width = min(image_width, max(26, int(image_width * width_ratio)))
        height = min(image_height, max(14, int(image_height * height_ratio)))
        for x_center_ratio in x_centers:
            for y_center_ratio in y_centers:
                center_x = int(round(image_width * x_center_ratio))
                center_y = int(round(image_height * y_center_ratio))
                bbox = _clip_bbox(
                    x=center_x - (width // 2),
                    y=center_y - (height // 2),
                    width=width,
                    height=height,
                    image_width=image_width,
                    image_height=image_height,
                )
                if bbox is None:
                    continue
                x_focus = 1.0 - min(1.0, abs(x_center_ratio - 0.68) / 0.68)
                y_focus = 1.0 - min(1.0, abs(y_center_ratio - 0.55) / 0.55)
                priority = base_priority + (0.18 * x_focus) + (0.12 * y_focus)
                candidates.append((bbox, priority))
    return candidates


def _candidate_patch_bboxes(
    lines: list[OCRLine],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    ranked: list[tuple[tuple[int, int, int, int], float]] = []

    ocr_candidates = _candidate_patch_bboxes_from_ocr_lines(
        lines,
        image_width=image_width,
        image_height=image_height,
    )
    for idx, bbox in enumerate(ocr_candidates):
        ranked.append((bbox, 2.0 - (0.05 * idx)))

    ranked.extend(_dense_patch_bboxes(image_width=image_width, image_height=image_height))
    ranked.sort(key=lambda item: item[1], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    for bbox, _priority in ranked:
        if any(_bbox_iou(bbox, existing) >= 0.55 for existing in selected):
            continue
        selected.append(bbox)
        if len(selected) >= 36:
            break
    return selected


def _build_model(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly_records = _anomaly_training_records(train_records)
    stats_records = anomaly_records if len(anomaly_records) >= 80 else train_records

    vendor_counts: dict[str, int] = {}
    vendor_anomaly_counts: dict[str, int] = {}
    vendor_fraud_counts: dict[str, int] = {}
    vendor_name_counts: dict[str, dict[str, int]] = {}
    genuine_totals: list[float] = []
    all_totals: list[float] = []
    vendor_genuine_totals: dict[str, list[float]] = {}
    fraud_count = 0

    for record in train_records:
        vendor, _date, total = _extract_from_fields(record)
        vendor_key = _normalize_vendor(vendor)
        if vendor_key:
            vendor_counts[vendor_key] = vendor_counts.get(vendor_key, 0) + 1
            display_vendor = str(vendor).strip() if vendor else vendor_key.title()
            per_vendor_names = vendor_name_counts.setdefault(vendor_key, {})
            per_vendor_names[display_vendor] = per_vendor_names.get(display_vendor, 0) + 1

    for record in stats_records:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        is_forged = int(bool(label.get("is_forged", 0)))
        vendor, _date, total = _extract_from_fields(record)
        vendor_key = _normalize_vendor(vendor)

        amount = _safe_float(total)
        if amount is not None:
            all_totals.append(amount)
            if not is_forged:
                genuine_totals.append(amount)
                if vendor_key:
                    vendor_genuine_totals.setdefault(vendor_key, []).append(amount)

    for record in anomaly_records:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        is_forged = int(bool(label.get("is_forged", 0)))
        fraud_count += is_forged

        vendor, _date, _total = _extract_from_fields(record)
        vendor_key = _normalize_vendor(vendor)
        if vendor_key:
            vendor_anomaly_counts[vendor_key] = vendor_anomaly_counts.get(vendor_key, 0) + 1
            vendor_fraud_counts[vendor_key] = vendor_fraud_counts.get(vendor_key, 0) + is_forged

    known_vendor_keys = sorted(vendor_counts, key=lambda key: (-vendor_counts[key], key))
    overall_rate = fraud_count / max(1, len(anomaly_records) if anomaly_records else len(train_records))
    overall_mean, overall_std = _mean_std(genuine_totals if genuine_totals else all_totals)

    vendor_fraud_rate: dict[str, float] = {}
    for vendor_key, count in vendor_anomaly_counts.items():
        forged = vendor_fraud_counts.get(vendor_key, 0)
        vendor_fraud_rate[vendor_key] = (forged + 1.0) / (count + 2.0)

    vendor_total_stats: dict[str, dict[str, float]] = {}
    for vendor_key, values in vendor_genuine_totals.items():
        mean, std = _mean_std(values)
        vendor_total_stats[vendor_key] = {"mean": mean, "std": std}

    vendor_display_names: dict[str, str] = {}
    for vendor_key, names in vendor_name_counts.items():
        if not names:
            vendor_display_names[vendor_key] = vendor_key.title()
            continue
        vendor_display_names[vendor_key] = sorted(
            names,
            key=lambda name: (-names[name], name),
        )[0]

    model = {
        "known_vendor_keys": known_vendor_keys,
        "vendor_display_names": vendor_display_names,
        "global_fraud_rate": overall_rate,
        "global_total_mean": overall_mean,
        "global_total_std": overall_std,
        "vendor_fraud_rate": vendor_fraud_rate,
        "vendor_total_stats": vendor_total_stats,
        "threshold": 0.25,
        "ml_enabled": False,
        "ml_threshold": 0.5,
        "patch_enabled": False,
        "patch_threshold": 0.60,
    }

    candidate_thresholds = [round(v / 20.0, 2) for v in range(-20, 21)]
    best_threshold = model["threshold"]
    best_score = -1.0
    base_rate = fraud_count / max(1, len(anomaly_records) if anomaly_records else len(train_records))
    for threshold in candidate_thresholds:
        scores: list[float] = []
        targets: list[int] = []
        source_records = anomaly_records if anomaly_records else train_records
        for record in source_records:
            label = record.get("label") if isinstance(record.get("label"), dict) else {}
            target = int(bool(label.get("is_forged", 0)))
            vendor, date, total = _extract_from_fields(record)
            score = _fraud_score(vendor, date, total, model)
            scores.append(score)
            targets.append(target)

        tp, fp, tn, fn = _classification_counts(scores, targets, threshold)
        balanced = _balanced_accuracy(tp, fp, tn, fn)
        f1 = _f1_score(tp, fp, fn)
        predicted_rate = (tp + fp) / max(1, len(targets))
        prevalence_penalty = abs(predicted_rate - base_rate)
        score_value = balanced + 0.10 * f1 - 0.15 * prevalence_penalty

        if score_value > best_score:
            best_score = score_value
            best_threshold = threshold
    model["threshold"] = best_threshold
    return model


def _restore_vendor_display(vendor_key: str | None, model: dict[str, Any]) -> str | None:
    if vendor_key is None:
        return None
    display_names = model.get("vendor_display_names")
    if isinstance(display_names, dict):
        value = display_names.get(vendor_key)
        if isinstance(value, str) and value.strip():
            return value
    return vendor_key.title()


def _canonicalize_vendor(vendor: str | None, model: dict[str, Any]) -> str | None:
    if vendor is None:
        return None
    vendor_key = _normalize_vendor(vendor)
    if vendor_key is None:
        return None

    known_keys = model.get("known_vendor_keys")
    if not isinstance(known_keys, list):
        return _restore_vendor_display(vendor_key, model)
    known_keys = [key for key in known_keys if isinstance(key, str)]

    if vendor_key in known_keys:
        return _restore_vendor_display(vendor_key, model)

    best_ratio = 0.0
    best_key: str | None = None
    for key in known_keys:
        if key in vendor_key or vendor_key in key:
            return _restore_vendor_display(key, model)
        ratio = SequenceMatcher(a=key, b=vendor_key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key
    if best_key and best_ratio >= 0.74:
        return _restore_vendor_display(best_key, model)

    return _restore_vendor_display(vendor_key, model)


def _fraud_score(vendor: str | None, date: str | None, total: str | None, model: dict[str, Any]) -> float:
    global_rate = float(model.get("global_fraud_rate", 0.5))
    score = _logit(global_rate)

    vendor_key = _normalize_vendor(vendor)
    vendor_rates = model.get("vendor_fraud_rate")
    if vendor_key and isinstance(vendor_rates, dict) and vendor_key in vendor_rates:
        score += 0.55 * _logit(float(vendor_rates[vendor_key]))
    elif vendor_key is None:
        score += 0.30

    amount = _safe_float(total)
    if amount is None:
        score += 0.65
    else:
        mean = float(model.get("global_total_mean", 0.0))
        std = float(model.get("global_total_std", 1.0))
        z = abs((amount - mean) / max(std, 1e-6))
        if z > 3.0:
            score += 0.75
        elif z > 2.2:
            score += 0.40

        vendor_stats = model.get("vendor_total_stats")
        if vendor_key and isinstance(vendor_stats, dict) and vendor_key in vendor_stats:
            vendor_stat = vendor_stats[vendor_key]
            if isinstance(vendor_stat, dict):
                v_mean = float(vendor_stat.get("mean", mean))
                v_std = float(vendor_stat.get("std", std))
                v_z = abs((amount - v_mean) / max(v_std, 1e-6))
                if v_z > 3.0:
                    score += 0.50

    normalized_date = _normalize_date(date) if date else None
    if normalized_date is None:
        score += 0.25
    else:
        try:
            year = int(normalized_date.split("-", 1)[0])
            if year < 2000 or year > 2035:
                score += 0.35
        except (ValueError, IndexError):
            score += 0.25
    return score


def _feature_vector(
    vendor: str | None,
    date: str | None,
    total: str | None,
    text: str,
    model: dict[str, Any],
    ocr_quality: tuple[float, float, float, float, float, float] | None = None,
    image_forensics: tuple[float, float, float, float, float, float, float, float] | None = None,
) -> list[float]:
    vendor_key = _normalize_vendor(vendor)
    vendor_rates = model.get("vendor_fraud_rate") if isinstance(model.get("vendor_fraud_rate"), dict) else {}
    vendor_rate = float(vendor_rates.get(vendor_key, model.get("global_fraud_rate", 0.5)))

    amount = _safe_float(total)
    mean = float(model.get("global_total_mean", 0.0))
    std = float(model.get("global_total_std", 1.0))
    amount_value = amount if amount is not None else -1.0
    amount_z = abs((amount_value - mean) / max(std, 1e-6)) if amount is not None else 9.0

    year, month, day = _extract_date_parts(date)
    has_vendor = 1.0 if vendor else 0.0
    has_date = 1.0 if date else 0.0
    has_total = 1.0 if total else 0.0

    text_lines, text_chars, digit_ratio = _text_features(text)
    quality = ocr_quality if ocr_quality is not None else _ocr_quality_features(text)
    forensics = image_forensics if image_forensics is not None else (0.0,) * len(IMAGE_FORENSICS_FEATURE_NAMES)

    return [
        has_vendor,
        has_date,
        has_total,
        vendor_rate,
        amount_value,
        amount_z,
        float(year),
        float(month),
        float(day),
        text_lines,
        text_chars,
        digit_ratio,
        *quality,
        *forensics,
    ]


def _train_ml_classifier(
    train_records: list[dict[str, Any]],
    train_dir: Path,
    model: dict[str, Any],
) -> tuple[Any, float] | None:
    if not SKLEARN_AVAILABLE or RandomForestClassifier is None:
        return None

    records_for_training = _anomaly_training_records(train_records)
    if not records_for_training:
        return None

    known_vendors = [
        _restore_vendor_display(key, model)
        for key in model.get("known_vendor_keys", [])
        if isinstance(key, str)
    ]
    known_vendors = [value for value in known_vendors if isinstance(value, str)]

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for record in records_for_training:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        target = int(bool(label.get("is_forged", 0)))
        if target == 1:
            positives.append(record)
        else:
            negatives.append(record)

    selected_records = records_for_training
    if positives and len(negatives) > len(positives) * 4:
        max_negatives = len(positives) * 4
        step = len(negatives) / max(1, max_negatives)
        selected_negatives = [
            negatives[min(int(i * step), len(negatives) - 1)]
            for i in range(max_negatives)
        ]
        selected_records = positives + selected_negatives

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for record in selected_records:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        target = int(bool(label.get("is_forged", 0)))

        vendor, date, total = _extract_from_fields(record)
        text = ""

        image_ref = record.get("image_path")
        image_path = _resolve_image_path(train_dir, image_ref)
        image_forensics = _image_forensics_features(image_path)
        if image_path.exists():
            ocr_vendor, ocr_date, ocr_total, text, _ = _extract_with_ocr(
                image_path=image_path,
                known_vendors=known_vendors,
                include_lines=False,
            )
            if ocr_vendor:
                vendor = ocr_vendor
            if ocr_date:
                date = ocr_date
            if ocr_total:
                total = ocr_total

        vendor = _canonicalize_vendor(vendor, model)
        date = _normalize_date(date) if date else None
        total = _normalize_total(total)
        ocr_quality = _ocr_quality_features(text)

        x_rows.append(
            _feature_vector(
                vendor,
                date,
                total,
                text,
                model,
                ocr_quality=ocr_quality,
                image_forensics=image_forensics,
            )
        )
        y_rows.append(target)

    if len(set(y_rows)) < 2 or len(y_rows) < 20:
        return None

    clf = RandomForestClassifier(
        n_estimators=240,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        oob_score=True,
    )
    clf.fit(x_rows, y_rows)

    probabilities: list[float] = []
    oob_values = getattr(clf, "oob_decision_function_", None)
    if oob_values is not None:
        try:
            for row in oob_values:
                if not hasattr(row, "__len__") or len(row) < 2:
                    raise ValueError("invalid oob row")
                prob = float(row[1])
                if not math.isfinite(prob):
                    raise ValueError("invalid oob probability")
                probabilities.append(min(max(prob, 0.0), 1.0))
        except Exception:
            probabilities = []

    if len(probabilities) != len(y_rows):
        probabilities = [float(p[1]) for p in clf.predict_proba(x_rows)]

    best_threshold = 0.5
    best_score = -1.0
    base_rate = sum(y_rows) / max(1, len(y_rows))
    for raw in range(5, 96):
        threshold = raw / 100.0
        tp, fp, tn, fn = _classification_counts(probabilities, y_rows, threshold)
        balanced = _balanced_accuracy(tp, fp, tn, fn)
        f1 = _f1_score(tp, fp, fn)
        predicted_rate = (tp + fp) / max(1, len(y_rows))
        prevalence_penalty = abs(predicted_rate - base_rate)
        score_value = balanced + 0.10 * f1 - 0.15 * prevalence_penalty
        if score_value > best_score:
            best_score = score_value
            best_threshold = threshold
    return clf, best_threshold


def _train_patch_classifier(
    train_records: list[dict[str, Any]],
    train_dir: Path,
) -> tuple[Any, float] | None:
    if not SKLEARN_AVAILABLE or RandomForestClassifier is None:
        return None
    if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        return None

    records_for_training = _anomaly_training_records(train_records)
    if not records_for_training:
        return None

    rng = random.Random(42)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []

    for record in records_for_training:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        target = int(bool(label.get("is_forged", 0)))
        image_ref = record.get("image_path")
        image_path = _resolve_image_path(train_dir, image_ref)
        rgb_arr = _image_rgb_array(image_path)
        if rgb_arr is None:
            continue

        image_height = int(rgb_arr.shape[0])
        image_width = int(rgb_arr.shape[1])
        if image_height <= 2 or image_width <= 2:
            continue

        positive_bboxes = _extract_forgery_region_bboxes(record, image_width, image_height)
        if positive_bboxes:
            for bbox in positive_bboxes:
                feats = _patch_features_from_bbox(rgb_arr, bbox)
                if feats is None:
                    continue
                x_rows.append(feats)
                y_rows.append(1)

            negative_count = max(1, len(positive_bboxes) * 2)
            negative_bboxes = _sample_negative_patch_bboxes(
                image_width=image_width,
                image_height=image_height,
                positive_bboxes=positive_bboxes,
                sample_count=negative_count,
                rng=rng,
            )
            for bbox in negative_bboxes:
                feats = _patch_features_from_bbox(rgb_arr, bbox)
                if feats is None:
                    continue
                x_rows.append(feats)
                y_rows.append(0)
            continue

        if target == 0:
            negative_count = 1 if rng.random() < 0.7 else 2
            negative_bboxes = _sample_negative_patch_bboxes(
                image_width=image_width,
                image_height=image_height,
                positive_bboxes=[],
                sample_count=negative_count,
                rng=rng,
            )
            for bbox in negative_bboxes:
                feats = _patch_features_from_bbox(rgb_arr, bbox)
                if feats is None:
                    continue
                x_rows.append(feats)
                y_rows.append(0)

    if len(set(y_rows)) < 2 or len(y_rows) < 40:
        return None

    positive_indices = [idx for idx, target in enumerate(y_rows) if target == 1]
    negative_indices = [idx for idx, target in enumerate(y_rows) if target == 0]
    if positive_indices and len(negative_indices) > len(positive_indices) * 5:
        max_negatives = len(positive_indices) * 5
        step = len(negative_indices) / max(1, max_negatives)
        selected_negatives = [
            negative_indices[min(int(i * step), len(negative_indices) - 1)]
            for i in range(max_negatives)
        ]
        selected_indices = sorted(positive_indices + selected_negatives)
        x_rows = [x_rows[idx] for idx in selected_indices]
        y_rows = [y_rows[idx] for idx in selected_indices]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=9,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        oob_score=True,
    )
    clf.fit(x_rows, y_rows)

    probabilities: list[float] = []
    oob_values = getattr(clf, "oob_decision_function_", None)
    if oob_values is not None:
        try:
            for row in oob_values:
                if not hasattr(row, "__len__") or len(row) < 2:
                    raise ValueError("invalid oob row")
                prob = float(row[1])
                if not math.isfinite(prob):
                    raise ValueError("invalid oob probability")
                probabilities.append(min(max(prob, 0.0), 1.0))
        except Exception:
            probabilities = []

    if len(probabilities) != len(y_rows):
        probabilities = [float(p[1]) for p in clf.predict_proba(x_rows)]

    best_threshold = 0.60
    best_score = -1.0
    for raw in range(30, 91):
        threshold = raw / 100.0
        tp, fp, tn, fn = _classification_counts(probabilities, y_rows, threshold)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        beta2 = 4.0
        f2 = 0.0 if (precision + recall) <= 1e-12 else (1.0 + beta2) * precision * recall / max(1e-12, beta2 * precision + recall)
        fp_rate = fp / max(1, fp + tn)
        score_value = f2 - 0.08 * fp_rate
        if score_value > best_score:
            best_score = score_value
            best_threshold = threshold
    return clf, best_threshold


def _load_anomaly_classifier(model_dir: Path) -> Any | None:
    model_path = model_dir / "anomaly_model.pkl"
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_anomaly_classifier(model_dir: Path, classifier: Any) -> None:
    model_path = model_dir / "anomaly_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(classifier, handle)


def _load_outlier_model(model_dir: Path) -> Any | None:
    model_path = model_dir / "outlier_model.pkl"
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_outlier_model(model_dir: Path, classifier: Any) -> None:
    model_path = model_dir / "outlier_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(classifier, handle)


def _load_patch_classifier(model_dir: Path) -> Any | None:
    model_path = model_dir / "patch_model.pkl"
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_patch_classifier(model_dir: Path, classifier: Any) -> None:
    model_path = model_dir / "patch_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(classifier, handle)


def _resolve_image_path(data_root: Path, image_ref: Any) -> Path:
    if image_ref is None:
        return Path("")
    image_path = Path(str(image_ref))
    if image_path.is_absolute():
        return image_path
    return data_root / image_path


def _extract_last_amount(text: str) -> float | None:
    candidates = [_safe_float(token) for token in AMOUNT_RE.findall(text)]
    values = [value for value in candidates if value is not None]
    if not values:
        return None
    return float(values[-1])


def _extract_last_signed_amount(text: str) -> float | None:
    candidates = [_safe_float(token) for token in SIGNED_AMOUNT_RE.findall(text)]
    values = [value for value in candidates if value is not None]
    if not values:
        return _extract_last_amount(text)
    return float(values[-1])


def _looks_like_item_line(text: str) -> bool:
    if not text.strip() or not AMOUNT_RE.search(text):
        return False
    lower = text.lower()
    if TOTAL_HINT_RE.search(text) or SUBTOTAL_HINT_RE.search(text) or TAX_HINT_RE.search(text):
        return False
    if any(
        token in lower
        for token in ("cash", "change", "paid", "balance", "round", "discount", "tender", "refund")
    ):
        return False
    if sum(ch.isalpha() for ch in text) < 3:
        return False
    return True


def _looks_like_pretax_total_line(text: str) -> bool:
    lower = text.lower()
    if SUBTOTAL_HINT_RE.search(text):
        return True
    if not TAX_HINT_RE.search(text):
        return False
    if INCLUSIVE_HINT_RE.search(text):
        return False
    if "sales" in lower or "amount" in lower or "total" in lower:
        if SUBTOTAL_EX_TAX_HINT_RE.search(text):
            return True
    return False


def _looks_like_final_total_line(text: str) -> bool:
    lower = text.lower()
    if any(
        token in lower
        for token in ("subtotal", "sub total", "sub-total", "cash", "change", "paid", "tender", "refund", "qty", "item")
    ):
        return False
    if "total sales" in lower and INCLUSIVE_HINT_RE.search(text):
        return True
    if any(token in lower for token in ("grand total", "net total", "amount due", "total due", "total payable")):
        return True
    if lower.startswith("total") and not TAX_HINT_RE.search(text) and not DISCOUNT_HINT_RE.search(text):
        return True
    return False


def _receipt_structure_components(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    subtotal_candidates: list[float] = []
    tax_values: list[float] = []
    rounding_values: list[float] = []
    discount_values: list[float] = []
    service_values: list[float] = []
    total_candidates: list[float] = []
    explicit_item_counts: list[float] = []
    item_line_count = 0.0

    for line in lines:
        signed_amount = _extract_last_signed_amount(line)
        amount = abs(signed_amount) if signed_amount is not None else None
        if amount is not None and _looks_like_pretax_total_line(line):
            subtotal_candidates.append(amount)
        if amount is not None and TAX_HINT_RE.search(line) and not _looks_like_pretax_total_line(line):
            tax_values.append(amount)
        if amount is not None and ROUNDING_HINT_RE.search(line):
            rounding_values.append(float(signed_amount if signed_amount is not None else amount))
        if amount is not None and DISCOUNT_HINT_RE.search(line):
            discount_values.append(amount)
        if amount is not None and SERVICE_HINT_RE.search(line) and not TAX_HINT_RE.search(line):
            service_values.append(amount)
        if amount is not None and _looks_like_final_total_line(line):
            total_candidates.append(amount)

        match = ITEM_COUNT_RE.search(line)
        if match:
            try:
                explicit_item_counts.append(float(int(match.group(1))))
            except ValueError:
                pass

        if _looks_like_item_line(line):
            item_line_count += 1.0

    tax_amount = sum(tax_values) if tax_values else None
    explicit_count = max(explicit_item_counts) if explicit_item_counts else None
    num_items = explicit_count if explicit_count is not None else (item_line_count if item_line_count > 0 else None)

    return {
        "subtotal_candidates": subtotal_candidates,
        "tax_values": tax_values,
        "tax_amount": tax_amount,
        "tax_max": max(tax_values) if tax_values else None,
        "rounding_amount": sum(rounding_values) if rounding_values else None,
        "discount_amount": sum(discount_values) if discount_values else None,
        "service_amount": sum(service_values) if service_values else None,
        "total_candidates": total_candidates,
        "num_items": num_items,
        "item_line_count": float(item_line_count),
    }


def _math_formula_candidates(components: dict[str, Any]) -> list[tuple[str, float]]:
    subtotal_candidates_raw = components.get("subtotal_candidates")
    subtotal_candidates = (
        [float(value) for value in subtotal_candidates_raw if isinstance(value, (int, float))]
        if isinstance(subtotal_candidates_raw, list)
        else []
    )
    if not subtotal_candidates:
        return []

    addition_components: list[tuple[str, float]] = []
    tax_amount = components.get("tax_amount")
    if isinstance(tax_amount, (int, float)) and abs(float(tax_amount)) >= 0.005:
        addition_components.append(("tax_sum", float(tax_amount)))
    tax_max = components.get("tax_max")
    if isinstance(tax_max, (int, float)) and abs(float(tax_max)) >= 0.005:
        addition_components.append(("tax_max", float(tax_max)))
    service_amount = components.get("service_amount")
    if isinstance(service_amount, (int, float)) and abs(float(service_amount)) >= 0.005:
        addition_components.append(("service", float(service_amount)))
    rounding_amount = components.get("rounding_amount")
    if isinstance(rounding_amount, (int, float)) and abs(float(rounding_amount)) >= 0.005:
        addition_components.append(("rounding", float(rounding_amount)))

    discount_amount = components.get("discount_amount")
    discount_value = (
        float(discount_amount)
        if isinstance(discount_amount, (int, float)) and abs(float(discount_amount)) >= 0.005
        else None
    )

    candidates: dict[str, float] = {}
    for subtotal in subtotal_candidates:
        base_configs = [("subtotal", subtotal)]
        if discount_value is not None:
            base_configs.append(("subtotal_minus_discount", subtotal - discount_value))

        for base_name, base_value in base_configs:
            candidates[base_name] = base_value
            for count in range(1, len(addition_components) + 1):
                for subset in combinations(addition_components, count):
                    name = base_name + "".join(f"_plus_{label}" for label, _value in subset)
                    value = base_value + sum(component_value for _label, component_value in subset)
                    candidates[name] = value
    return sorted(candidates.items(), key=lambda item: (item[0], item[1]))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    q_clamped = min(max(q, 0.0), 1.0)
    position = q_clamped * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = position - lower_index
    return float(lower + ((upper - lower) * weight))


def _default_math_profile() -> dict[str, Any]:
    return {
        "trusted_formulas": [
            "subtotal",
            "subtotal_plus_tax_max",
            "subtotal_plus_tax_sum",
            "subtotal_plus_tax_max_plus_rounding",
            "subtotal_plus_tax_sum_plus_rounding",
            "subtotal_minus_discount_plus_tax_max",
            "subtotal_minus_discount_plus_tax_sum",
        ],
        "formula_rates": {},
        "tolerance_abs": 0.40,
        "tolerance_rel": 0.01,
        "profile_observations": 0,
        "profile_matches": 0,
        "profile_sources": {},
    }


def _math_error_tolerance(total_amount: float | None, math_profile: dict[str, Any] | None) -> float:
    profile = math_profile if isinstance(math_profile, dict) else {}
    abs_tol = float(profile.get("tolerance_abs", 0.15))
    rel_tol = float(profile.get("tolerance_rel", 0.01))
    reference = max(1.0, float(total_amount) if total_amount is not None else 1.0)
    return max(abs_tol, rel_tol * reference)


def _receipt_structure_features(
    text: str,
    total_amount: float | None,
    math_profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    components = _receipt_structure_components(text)
    subtotal_candidates = components.get("subtotal_candidates")
    subtotal_amount = (
        max(float(value) for value in subtotal_candidates)
        if isinstance(subtotal_candidates, list) and subtotal_candidates
        else None
    )
    tax_amount = components.get("tax_amount")
    discount_amount = components.get("discount_amount")
    rounding_amount = components.get("rounding_amount")
    service_amount = components.get("service_amount")
    num_items = components.get("num_items")
    item_line_count = float(components.get("item_line_count", 0.0))

    candidate_formulas = _math_formula_candidates(components)
    trusted_formulas_raw = math_profile.get("trusted_formulas") if isinstance(math_profile, dict) else None
    trusted_formulas = (
        {str(name) for name in trusted_formulas_raw if isinstance(name, str)}
        if isinstance(trusted_formulas_raw, list)
        else set()
    )
    supported_formulas = (
        [(name, value) for name, value in candidate_formulas if not trusted_formulas or name in trusted_formulas]
    )
    formula_rates = math_profile.get("formula_rates") if isinstance(math_profile, dict) else {}
    total_candidates_raw = components.get("total_candidates")
    total_candidates = (
        [float(value) for value in total_candidates_raw if isinstance(value, (int, float))]
        if isinstance(total_candidates_raw, list)
        else []
    )

    math_available = 0.0
    math_consistent = 0.0
    math_abs_error = 0.0
    math_rel_error = 0.0
    math_supported_formula = 0.0
    math_formula_match_rate = 0.0
    total_line_verified = 0.0

    if total_amount is not None and total_candidates:
        total_line_tolerance = max(0.15, 0.005 * max(1.0, total_amount))
        total_line_verified = 1.0 if any(abs(candidate - total_amount) <= total_line_tolerance for candidate in total_candidates) else 0.0

    if total_amount is not None and supported_formulas and total_line_verified >= 0.5:
        best_formula_name, best_formula_value = min(
            supported_formulas,
            key=lambda item: abs(item[1] - total_amount),
        )
        math_abs_error = abs(best_formula_value - total_amount)
        math_rel_error = math_abs_error / max(1.0, total_amount)
        math_supported_formula = 1.0
        math_formula_match_rate = float(formula_rates.get(best_formula_name, 0.0)) if isinstance(formula_rates, dict) else 0.0
        math_available = 1.0
        math_consistent = 1.0 if math_abs_error <= _math_error_tolerance(total_amount, math_profile) else 0.0

    return {
        "subtotal_amount": float(subtotal_amount) if subtotal_amount is not None else 0.0,
        "subtotal_missing": 0.0 if subtotal_amount is not None else 1.0,
        "tax_amount": float(tax_amount) if tax_amount is not None else 0.0,
        "tax_missing": 0.0 if tax_amount is not None else 1.0,
        "discount_amount": float(discount_amount) if discount_amount is not None else 0.0,
        "discount_missing": 0.0 if discount_amount is not None else 1.0,
        "rounding_amount": float(rounding_amount) if rounding_amount is not None else 0.0,
        "rounding_missing": 0.0 if rounding_amount is not None else 1.0,
        "service_amount": float(service_amount) if service_amount is not None else 0.0,
        "service_missing": 0.0 if service_amount is not None else 1.0,
        "num_items": float(num_items) if num_items is not None else 0.0,
        "num_items_missing": 0.0 if num_items is not None else 1.0,
        "item_line_count": item_line_count,
        "math_available": math_available,
        "math_consistent": math_consistent,
        "math_abs_error": float(math_abs_error),
        "math_rel_error": float(math_rel_error),
        "math_supported_formula": math_supported_formula,
        "math_formula_match_rate": math_formula_match_rate,
        "total_line_verified": total_line_verified,
    }


def _build_math_profile_from_observations(observations: list[ReceiptObservation]) -> dict[str, Any]:
    default_profile = _default_math_profile()
    formula_counts: dict[str, int] = {}
    match_errors: list[float] = []
    match_rel_errors: list[float] = []
    candidate_observations = 0

    for observation in observations:
        total_amount = observation.total_amount
        if total_amount is None or not observation.text.strip():
            continue
        components = _receipt_structure_components(observation.text)
        candidates = _math_formula_candidates(components)
        if not candidates:
            continue

        candidate_observations += 1
        best_formula_name, best_formula_value = min(
            candidates,
            key=lambda item: abs(item[1] - total_amount),
        )
        best_abs_error = abs(best_formula_value - total_amount)
        best_rel_error = best_abs_error / max(1.0, total_amount)
        seed_tolerance = max(0.10, 0.01 * max(1.0, total_amount))
        if best_abs_error > seed_tolerance:
            continue

        formula_counts[best_formula_name] = formula_counts.get(best_formula_name, 0) + 1
        match_errors.append(best_abs_error)
        match_rel_errors.append(best_rel_error)

    if candidate_observations < 16 or len(match_errors) < 8:
        return default_profile

    matched = len(match_errors)
    formula_rates = {
        name: count / matched
        for name, count in formula_counts.items()
    }
    trusted_formulas = [
        name
        for name, count in sorted(formula_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 3 and formula_rates.get(name, 0.0) >= 0.04
    ]
    if not trusted_formulas:
        trusted_formulas = default_profile["trusted_formulas"]

    profile = dict(default_profile)
    profile["trusted_formulas"] = trusted_formulas
    profile["formula_rates"] = formula_rates
    profile["tolerance_abs"] = min(max(_quantile(match_errors, 0.95), 0.40), 0.75)
    profile["tolerance_rel"] = min(max(_quantile(match_rel_errors, 0.95), 0.005), 0.03)
    profile["profile_observations"] = candidate_observations
    profile["profile_matches"] = matched
    return profile


def _build_math_profile(
    train_records: list[dict[str, Any]],
    train_root: Path,
    model: dict[str, Any],
) -> dict[str, Any]:
    reference_records_by_source: dict[str, list[dict[str, Any]]] = {"sroie": [], "cord": []}
    for record in train_records:
        source = str(record.get("source", "")).strip().lower()
        if source not in reference_records_by_source:
            continue
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        if int(bool(label.get("is_forged", 0))) != 0:
            continue
        reference_records_by_source[source].append(record)

    known_vendors = [
        _restore_vendor_display(key, model)
        for key in model.get("known_vendor_keys", [])
        if isinstance(key, str)
    ]
    known_vendors = [value for value in known_vendors if isinstance(value, str)]

    observations: list[ReceiptObservation] = []
    rng = random.Random(42)
    sampled_counts: dict[str, int] = {}
    for source, records in reference_records_by_source.items():
        ordered_records = sorted(
            records,
            key=lambda row: str(row.get("id", "")),
        )
        if len(ordered_records) > MATH_PROFILE_SAMPLE_PER_SOURCE:
            ordered_records = sorted(
                rng.sample(ordered_records, MATH_PROFILE_SAMPLE_PER_SOURCE),
                key=lambda row: str(row.get("id", "")),
            )
        sampled_counts[source] = len(ordered_records)
        for record in ordered_records:
            observation = _observe_receipt(
                record=record,
                data_root=train_root,
                model=model,
                known_vendors=known_vendors,
                need_text=True,
                need_lines=False,
            )
            if observation.text.strip():
                observations.append(observation)

    profile = _build_math_profile_from_observations(observations)
    profile["profile_sources"] = sampled_counts
    return profile


def _ela_scalar_features(image_path: Path) -> tuple[float, float, float, float]:
    if not image_path.exists() or not NUMPY_AVAILABLE or not PIL_AVAILABLE:
        return (0.0,) * len(ELA_FEATURE_NAMES)

    try:
        with Image.open(image_path) as pil_img:
            rgb = pil_img.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            with Image.open(buffer) as recompressed:
                rec_rgb = recompressed.convert("RGB")
                orig_arr = np.asarray(rgb, dtype=np.float32) / 255.0
                rec_arr = np.asarray(rec_rgb, dtype=np.float32) / 255.0
    except Exception:
        return (0.0,) * len(ELA_FEATURE_NAMES)

    if orig_arr.shape != rec_arr.shape or orig_arr.ndim != 3:
        return (0.0,) * len(ELA_FEATURE_NAMES)

    diff = np.abs(orig_arr[:, :, :3] - rec_arr[:, :, :3])
    diff_gray = np.mean(diff, axis=2)
    if diff_gray.size == 0:
        return (0.0,) * len(ELA_FEATURE_NAMES)

    return (
        float(np.mean(diff_gray)),
        float(np.max(diff_gray)),
        float(np.std(diff_gray)),
        float(np.mean(diff_gray >= 0.15)),
    )


def _observe_receipt(
    record: dict[str, Any],
    data_root: Path,
    model: dict[str, Any],
    known_vendors: list[str],
    need_text: bool,
    need_lines: bool,
) -> ReceiptObservation:
    vendor, date_raw, total_raw = _extract_from_fields(record)
    text = ""
    ocr_lines: list[OCRLine] = []
    image_path = _resolve_image_path(data_root, record.get("image_path"))

    has_all_fields = vendor is not None and date_raw is not None and total_raw is not None
    if need_text and has_all_fields and not need_lines and image_path.exists():
        text = _run_tesseract(image_path, psm=6)

    if (vendor is None or date_raw is None or total_raw is None or need_lines) or (need_text and not text):
        ocr_vendor, ocr_date, ocr_total, text, ocr_lines = _extract_with_ocr(
            image_path=image_path,
            known_vendors=known_vendors,
            include_lines=need_lines,
        )
        if vendor is None:
            vendor = ocr_vendor
        if date_raw is None:
            date_raw = ocr_date
        if total_raw is None:
            total_raw = ocr_total

    if not text and need_text and image_path.exists():
        text = _run_tesseract(image_path, psm=6)

    vendor = _canonicalize_vendor(vendor, model)
    date = _normalize_date(date_raw) if date_raw else None
    total = _normalize_total(total_raw)
    total_amount = _safe_float(total)
    return ReceiptObservation(
        vendor=vendor,
        date_raw=date_raw,
        date=date,
        total_raw=total_raw,
        total=total,
        total_amount=total_amount,
        text=text,
        ocr_lines=ocr_lines,
        image_path=image_path,
    )


def _outlier_numeric_vector(
    total_amount: float | None,
    tax_amount: float,
    num_items: float,
    model: dict[str, Any],
) -> list[float]:
    medians = model.get("outlier_numeric_medians")
    if not isinstance(medians, dict):
        medians = {}
    return [
        float(total_amount) if total_amount is not None else float(medians.get("total", 0.0)),
        float(tax_amount) if tax_amount > 0.0 else float(medians.get("tax", 0.0)),
        float(num_items) if num_items > 0.0 else float(medians.get("num_items", 0.0)),
    ]


def _forgery_feature_map(
    observation: ReceiptObservation,
    model: dict[str, Any],
    outlier_model: Any | None,
) -> dict[str, float]:
    structure = _receipt_structure_features(
        observation.text,
        observation.total_amount,
        model.get("math_profile") if isinstance(model, dict) else None,
    )
    ela_features = _ela_scalar_features(observation.image_path)
    return _compose_forgery_feature_map(
        observation=observation,
        structure=structure,
        ela_features=ela_features,
        model=model,
        outlier_model=outlier_model,
    )


def _compose_forgery_feature_map(
    observation: ReceiptObservation,
    structure: dict[str, float],
    ela_features: tuple[float, float, float, float],
    model: dict[str, Any],
    outlier_model: Any | None,
) -> dict[str, float]:
    ela_mean, ela_max, ela_std, ela_high_ratio = ela_features
    invalid_date = 1.0 if observation.date_raw and observation.date is None else 0.0

    outlier_vector = _outlier_numeric_vector(
        total_amount=observation.total_amount,
        tax_amount=float(structure["tax_amount"]),
        num_items=float(structure["num_items"]),
        model=model,
    )
    outlier_score = 0.0
    outlier_flag = 0.0
    if outlier_model is not None:
        try:
            decision = float(outlier_model.decision_function([outlier_vector])[0])
            outlier_score = max(0.0, -decision)
            outlier_flag = 1.0 if int(outlier_model.predict([outlier_vector])[0]) == -1 else 0.0
        except Exception:
            outlier_score = 0.0
            outlier_flag = 0.0

    return {
        "missing_vendor": 0.0 if observation.vendor else 1.0,
        "missing_date": 0.0 if observation.date_raw else 1.0,
        "missing_total": 0.0 if observation.total_amount is not None else 1.0,
        "invalid_date": invalid_date,
        "total_amount": float(observation.total_amount) if observation.total_amount is not None else 0.0,
        "subtotal_amount": float(structure["subtotal_amount"]),
        "tax_amount": float(structure["tax_amount"]),
        "discount_amount": float(structure["discount_amount"]),
        "rounding_amount": float(structure["rounding_amount"]),
        "service_amount": float(structure["service_amount"]),
        "num_items": float(structure["num_items"]),
        "subtotal_missing": float(structure["subtotal_missing"]),
        "tax_missing": float(structure["tax_missing"]),
        "discount_missing": float(structure["discount_missing"]),
        "rounding_missing": float(structure["rounding_missing"]),
        "service_missing": float(structure["service_missing"]),
        "num_items_missing": float(structure["num_items_missing"]),
        "item_line_count": float(structure["item_line_count"]),
        "math_available": float(structure["math_available"]),
        "math_consistent": float(structure["math_consistent"]),
        "math_abs_error": float(structure["math_abs_error"]),
        "math_rel_error": float(structure["math_rel_error"]),
        "math_supported_formula": float(structure["math_supported_formula"]),
        "math_formula_match_rate": float(structure["math_formula_match_rate"]),
        "total_line_verified": float(structure["total_line_verified"]),
        "iso_score": float(outlier_score),
        "iso_outlier": float(outlier_flag),
        "ela_mean": float(ela_mean),
        "ela_max": float(ela_max),
        "ela_std": float(ela_std),
        "ela_high_ratio": float(ela_high_ratio),
    }


def _forgery_feature_vector(feature_map: dict[str, float]) -> list[float]:
    ordered_names = (
        "missing_vendor",
        "missing_date",
        "missing_total",
        "invalid_date",
        "total_amount",
        "subtotal_amount",
        "tax_amount",
        "discount_amount",
        "rounding_amount",
        "service_amount",
        "num_items",
        "subtotal_missing",
        "tax_missing",
        "discount_missing",
        "rounding_missing",
        "service_missing",
        "num_items_missing",
        "item_line_count",
        "math_available",
        "math_consistent",
        "math_abs_error",
        "math_rel_error",
        "math_supported_formula",
        "math_formula_match_rate",
        "total_line_verified",
        "iso_score",
        "iso_outlier",
        "ela_mean",
        "ela_max",
        "ela_std",
        "ela_high_ratio",
    )
    return [float(feature_map.get(name, 0.0)) for name in ordered_names]


def _forgery_reasons(feature_map: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if feature_map.get("missing_vendor", 0.0) >= 0.5:
        reasons.append("Missing vendor")
    if feature_map.get("missing_date", 0.0) >= 0.5:
        reasons.append("Missing date")
    elif feature_map.get("invalid_date", 0.0) >= 0.5:
        reasons.append("Invalid date")
    if feature_map.get("missing_total", 0.0) >= 0.5:
        reasons.append("Missing total")
    if feature_map.get("math_available", 0.0) >= 0.5 and feature_map.get("math_consistent", 1.0) < 0.5:
        reasons.append("Receipt totals do not match common genuine math patterns")
    if feature_map.get("iso_outlier", 0.0) >= 0.5:
        reasons.append("Numeric profile is an outlier")
    if feature_map.get("ela_high_ratio", 0.0) >= 0.08 or feature_map.get("ela_max", 0.0) >= 0.25:
        reasons.append("ELA suggests elevated recompression differences")
    return reasons


def _tune_probability_threshold(probabilities: list[float], targets: list[int]) -> float:
    if not probabilities or len(probabilities) != len(targets):
        return 0.5

    base_rate = sum(targets) / max(1, len(targets))
    best_threshold = 0.5
    best_score = -1.0
    for raw in range(5, 96):
        threshold = raw / 100.0
        tp, fp, tn, fn = _classification_counts(probabilities, targets, threshold)
        balanced = _balanced_accuracy(tp, fp, tn, fn)
        f1 = _f1_score(tp, fp, fn)
        predicted_rate = (tp + fp) / max(1, len(targets))
        prevalence_penalty = abs(predicted_rate - base_rate)
        score_value = balanced + 0.10 * f1 - 0.10 * prevalence_penalty
        if score_value > best_score:
            best_score = score_value
            best_threshold = threshold
    return best_threshold


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _prepare_forgery_training_rows(
    train_records: list[dict[str, Any]],
    train_dir: Path,
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    records_for_training = _anomaly_training_records(train_records)
    if not records_for_training:
        return []

    known_vendors = [
        _restore_vendor_display(key, model)
        for key in model.get("known_vendor_keys", [])
        if isinstance(key, str)
    ]
    known_vendors = [value for value in known_vendors if isinstance(value, str)]

    rows: list[dict[str, Any]] = []
    for record in records_for_training:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        target = int(bool(label.get("is_forged", 0)))
        observation = _observe_receipt(
            record=record,
            data_root=train_dir,
            model=model,
            known_vendors=known_vendors,
            need_text=True,
            need_lines=False,
        )
        structure = _receipt_structure_features(
            observation.text,
            observation.total_amount,
            model.get("math_profile") if isinstance(model, dict) else None,
        )
        rows.append(
            {
                "target": target,
                "observation": observation,
                "structure": structure,
                "ela_features": _ela_scalar_features(observation.image_path),
            }
        )
    return rows


def _train_outlier_model(
    training_rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> tuple[Any | None, dict[str, float]]:
    default_medians = {"total": 0.0, "tax": 0.0, "num_items": 0.0}
    if not SKLEARN_AVAILABLE or IsolationForest is None:
        return None, default_medians

    if not training_rows:
        return None, default_medians

    genuine_vectors_raw: list[tuple[float | None, float, float]] = []
    total_values: list[float] = []
    tax_values: list[float] = []
    item_values: list[float] = []
    forged_rate_numerator = 0

    for row in training_rows:
        target = int(row.get("target", 0))
        forged_rate_numerator += target
        if target != 0:
            continue

        observation = row.get("observation")
        structure = row.get("structure")
        if not isinstance(observation, ReceiptObservation) or not isinstance(structure, dict):
            continue
        genuine_vectors_raw.append(
            (
                observation.total_amount,
                float(structure["tax_amount"]),
                float(structure["num_items"]),
            )
        )
        if observation.total_amount is not None:
            total_values.append(float(observation.total_amount))
        if structure["tax_missing"] < 0.5:
            tax_values.append(float(structure["tax_amount"]))
        if structure["num_items_missing"] < 0.5:
            item_values.append(float(structure["num_items"]))

    medians = {
        "total": _median(total_values),
        "tax": _median(tax_values),
        "num_items": _median(item_values),
    }

    if len(genuine_vectors_raw) < 24:
        return None, medians

    vectors = [
        [
            float(total) if total is not None else medians["total"],
            float(tax) if tax > 0.0 else medians["tax"],
            float(num_items) if num_items > 0.0 else medians["num_items"],
        ]
        for total, tax, num_items in genuine_vectors_raw
    ]

    forged_rate = forged_rate_numerator / max(1, len(training_rows))
    contamination = min(max(forged_rate, 0.05), 0.20)
    clf = IsolationForest(
        n_estimators=240,
        contamination=contamination,
        random_state=42,
    )
    clf.fit(vectors)
    return clf, medians


def _train_lightgbm_classifier(
    training_rows: list[dict[str, Any]],
    model: dict[str, Any],
    outlier_model: Any | None,
) -> tuple[Any, float] | None:
    if not LIGHTGBM_AVAILABLE or LGBMClassifier is None:
        return None

    if not training_rows:
        return None

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for row in training_rows:
        observation = row.get("observation")
        structure = row.get("structure")
        ela_features = row.get("ela_features")
        if (
            not isinstance(observation, ReceiptObservation)
            or not isinstance(structure, dict)
            or not isinstance(ela_features, tuple)
            or len(ela_features) != 4
        ):
            continue
        feature_map = _compose_forgery_feature_map(
            observation=observation,
            structure=structure,
            ela_features=ela_features,
            model=model,
            outlier_model=outlier_model,
        )
        x_rows.append(_forgery_feature_vector(feature_map))
        y_rows.append(int(row.get("target", 0)))

    if len(x_rows) < 20 or len(set(y_rows)) < 2:
        return None

    positives = sum(y_rows)
    negatives = max(0, len(y_rows) - positives)
    scale_pos_weight = 1.0
    if positives > 0 and negatives > 0:
        scale_pos_weight = (negatives / positives) * 2.0

    x_train = np.asarray(x_rows, dtype=np.float32) if NUMPY_AVAILABLE else x_rows
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=0.02,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(x_train, y_rows)

    model["anomaly_variant"] = CURRENT_ANOMALY_VARIANT
    model["anomaly_scale_pos_weight"] = float(scale_pos_weight)
    return clf, DEFAULT_ML_THRESHOLD


def _suspicious_reasons(
    vendor: str | None,
    date: str | None,
    total: str | None,
    text: str,
    model: dict[str, Any],
    ocr_quality: tuple[float, float, float, float, float, float] | None = None,
    image_forensics: tuple[float, float, float, float, float, float, float, float] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if not vendor:
        reasons.append("Missing vendor")
    if not date:
        reasons.append("Missing or invalid date")
    if not total:
        reasons.append("Missing total")

    amount = _safe_float(total)
    if amount is not None:
        mean = float(model.get("global_total_mean", 0.0))
        std = float(model.get("global_total_std", 1.0))
        z = abs((amount - mean) / max(std, 1e-6))
        if z > 2.5:
            reasons.append("Total is a strong outlier")

    if text and len(text.strip()) < 20:
        reasons.append("Very low OCR text signal")

    if ocr_quality is not None:
        _alpha_ratio, non_alnum_ratio, noisy_token_ratio, _long_token_ratio, repeat_ratio, _avg_token_len = ocr_quality
        if noisy_token_ratio > 0.35 or non_alnum_ratio > 0.38:
            reasons.append("OCR text appears noisy or artifact-heavy")
        if repeat_ratio > 0.18:
            reasons.append("Repeated OCR character patterns detected")

    if image_forensics is not None:
        _gray_mean, _gray_std, _sharpness, _edge_density, blockiness, _color_std, _color_gap, entropy_norm = image_forensics
        if blockiness > 1.45:
            reasons.append("Image shows elevated compression block artifacts")
        if entropy_norm < 0.35:
            reasons.append("Unusually low image texture/entropy")
    return reasons


def _heuristic_anomaly_summary(prediction: dict[str, Any], debug: dict[str, Any]) -> str:
    probability = float(debug.get("probability", 0.0))
    is_forged = int(prediction.get("is_forged", 0))
    reasons = debug.get("reasons", [])
    reasons_list = [str(reason) for reason in reasons] if isinstance(reasons, list) else []

    if is_forged == 1:
        if reasons_list:
            return (
                f"Likely suspicious receipt ({probability:.1%}). "
                f"Primary signals: {', '.join(reasons_list[:3])}."
            )
        return f"Likely suspicious receipt ({probability:.1%}) based on anomaly scoring."

    if reasons_list:
        return (
            f"Likely genuine receipt ({probability:.1%} suspicious score), "
            f"but keep an eye on: {', '.join(reasons_list[:2])}."
        )
    return f"Likely genuine receipt ({probability:.1%} suspicious score) with no strong tampering signals."


def _llm_anomaly_summary(
    prediction: dict[str, Any],
    debug: dict[str, Any],
    llm_model: str | None,
) -> tuple[str | None, str | None]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DOCFUSION_LLM_API_KEY")
    if not api_key:
        return None, "Missing OPENAI_API_KEY (or DOCFUSION_LLM_API_KEY)."

    model_name = (llm_model or os.environ.get("DOCFUSION_LLM_MODEL") or "gpt-4o-mini").strip()
    if not model_name:
        model_name = "gpt-4o-mini"

    base_url = (os.environ.get("DOCFUSION_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    timeout_seconds = 12.0
    timeout_raw = os.environ.get("DOCFUSION_LLM_TIMEOUT")
    if timeout_raw:
        try:
            timeout_seconds = max(1.0, float(timeout_raw))
        except ValueError:
            timeout_seconds = 12.0

    ocr_lines = debug.get("ocr_lines", [])
    ocr_preview: list[str] = []
    if isinstance(ocr_lines, list):
        for row in ocr_lines[:8]:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if text:
                ocr_preview.append(text)

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 120,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a receipt-fraud analyst. "
                    "Write exactly 1-2 concise sentences explaining whether the receipt appears forged."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "vendor": prediction.get("vendor"),
                        "date": prediction.get("date"),
                        "total": prediction.get("total"),
                        "is_forged": prediction.get("is_forged"),
                        "probability": debug.get("probability"),
                        "reasons": debug.get("reasons"),
                        "ocr_preview": ocr_preview,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        detail = detail.strip()
        if detail:
            return None, f"LLM request failed ({exc.code}): {detail[:200]}"
        return None, f"LLM request failed ({exc.code})."
    except Exception as exc:
        return None, f"LLM request failed: {exc}"

    try:
        payload_out = json.loads(body)
    except json.JSONDecodeError:
        return None, "LLM returned invalid JSON payload."

    choices = payload_out.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "LLM response did not include choices."
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "LLM response did not include text content."
    return content.strip(), None


def _attach_anomaly_summary(
    prediction: dict[str, Any],
    llm_enabled: bool,
    llm_model: str | None,
) -> None:
    debug = prediction.get("_debug")
    if not isinstance(debug, dict):
        return

    heuristic = _heuristic_anomaly_summary(prediction, debug)
    debug["anomaly_summary"] = heuristic
    debug["summary_source"] = "heuristic"

    if not llm_enabled:
        return

    llm_summary, llm_error = _llm_anomaly_summary(prediction, debug, llm_model)
    if llm_summary:
        debug["anomaly_summary"] = llm_summary
        debug["summary_source"] = "llm"
    elif llm_error:
        debug["llm_error"] = llm_error


def _predict_single_record(
    record: dict[str, Any],
    data_root: Path,
    model: dict[str, Any],
    classifier: Any | None,
    outlier_model: Any | None,
    known_vendors: list[str],
    include_debug: bool,
) -> dict[str, Any]:
    record_id = record.get("id")
    if not isinstance(record_id, str):
        record_id = "unknown"
    ml_enabled = bool(model.get("ml_enabled", False) and classifier is not None)
    observation = _observe_receipt(
        record=record,
        data_root=data_root,
        model=model,
        known_vendors=known_vendors,
        need_text=ml_enabled,
        need_lines=include_debug,
    )

    prediction: dict[str, Any] = {
        "id": record_id,
        "vendor": observation.vendor,
        "date": observation.date,
        "total": observation.total,
        "is_forged": 0,
    }

    suspicious_score = 0.0
    threshold = float(model.get("ml_threshold", 0.5))
    feature_map: dict[str, float] = {}
    if ml_enabled:
        feature_map = _forgery_feature_map(observation, model, outlier_model)
        feature_vector = _forgery_feature_vector(feature_map)
        try:
            predict_input = np.asarray([feature_vector], dtype=np.float32) if NUMPY_AVAILABLE else [feature_vector]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                    category=UserWarning,
                )
                suspicious_score = float(classifier.predict_proba(predict_input)[0][1])
        except Exception as exc:
            suspicious_score = 0.0
            model["model_warning"] = (
                "Loaded anomaly model is incompatible with the current feature pipeline. "
                "Retrain or replace the latest model artifacts."
            )
            if include_debug:
                feature_map["_prediction_error"] = 1.0
                feature_map["_prediction_error_msg"] = str(exc)[:240]  # type: ignore[assignment]
        prediction["is_forged"] = int(suspicious_score >= threshold)

    if include_debug:
        boxes = _find_field_boxes(observation.ocr_lines, observation.vendor, observation.date, observation.total)
        prediction["_debug"] = {
            "mode": "feature_ensemble" if ml_enabled else "extraction_only",
            "boxes": boxes,
            "ocr_lines": [
                {
                    "text": line.text,
                    "left": line.left,
                    "top": line.top,
                    "right": line.right,
                    "bottom": line.bottom,
                    "conf": line.conf,
                }
                for line in observation.ocr_lines
            ],
            "suspicious_score": suspicious_score,
            "threshold": threshold,
            "reasons": _forgery_reasons(feature_map),
            "features": feature_map,
            "model_warning": model.get("model_warning"),
        }

    return prediction


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        train_root = Path(train_dir)
        train_data_root, train_path = _resolve_split_dataset_root(train_root, "train")
        model_dir = Path(work_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_records = _load_jsonl(train_path)
        model = _build_model(train_records)
        model["anomaly_pipeline"] = CURRENT_ANOMALY_PIPELINE
        model["anomaly_variant"] = CURRENT_ANOMALY_VARIANT
        model["math_profile"] = _build_math_profile(train_records, train_data_root, model)
        training_rows = _prepare_forgery_training_rows(train_records, train_data_root, model)
        outlier_model, medians = _train_outlier_model(training_rows, model)
        model["outlier_enabled"] = outlier_model is not None
        model["outlier_numeric_medians"] = medians
        model["ml_enabled"] = False
        model["patch_enabled"] = False
        model["ml_threshold"] = DEFAULT_ML_THRESHOLD
        model["patch_threshold"] = 0.60

        classifier_bundle = _train_lightgbm_classifier(
            training_rows=training_rows,
            model=model,
            outlier_model=outlier_model,
        )
        if outlier_model is not None:
            _save_outlier_model(model_dir, outlier_model)
        if classifier_bundle is not None:
            classifier, threshold = classifier_bundle
            _save_anomaly_classifier(model_dir, classifier)
            model["ml_enabled"] = True
            model["ml_threshold"] = threshold

        model_path = model_dir / "model.json"
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2, sort_keys=True)
        return str(model_dir)

    def _load_model(self, model_dir: str) -> tuple[dict[str, Any], Any | None, Any | None, list[str]]:
        model_path = Path(model_dir) / "model.json"
        if model_path.exists():
            with model_path.open("r", encoding="utf-8") as handle:
                model = json.load(handle)
        else:
            model = {
                "known_vendor_keys": [],
                "vendor_display_names": {},
                "global_fraud_rate": 0.5,
                "global_total_mean": 0.0,
                "global_total_std": 1.0,
                "vendor_fraud_rate": {},
                "vendor_total_stats": {},
                "threshold": 0.25,
                "anomaly_pipeline": CURRENT_ANOMALY_PIPELINE,
                "anomaly_variant": CURRENT_ANOMALY_VARIANT,
                "math_profile": _default_math_profile(),
                "ml_enabled": False,
                "ml_threshold": DEFAULT_ML_THRESHOLD,
                "outlier_enabled": False,
                "outlier_numeric_medians": {"total": 0.0, "tax": 0.0, "num_items": 0.0},
                "patch_enabled": False,
                "patch_threshold": 0.60,
            }

        if "ml_enabled" not in model:
            model["ml_enabled"] = False
        if "anomaly_pipeline" not in model:
            model["anomaly_pipeline"] = "legacy"
        if "anomaly_variant" not in model:
            model["anomaly_variant"] = CURRENT_ANOMALY_VARIANT
        if "math_profile" not in model or not isinstance(model.get("math_profile"), dict):
            model["math_profile"] = _default_math_profile()
        if "ml_threshold" not in model:
            model["ml_threshold"] = DEFAULT_ML_THRESHOLD
        if "outlier_enabled" not in model:
            model["outlier_enabled"] = False
        if "outlier_numeric_medians" not in model:
            model["outlier_numeric_medians"] = {"total": 0.0, "tax": 0.0, "num_items": 0.0}
        if "patch_enabled" not in model:
            model["patch_enabled"] = False
        if "patch_threshold" not in model:
            model["patch_threshold"] = 0.60

        if model_path.exists() and model.get("anomaly_pipeline") != CURRENT_ANOMALY_PIPELINE:
            raise ValueError(
                f"Incompatible model pipeline at {model_path}: "
                f"expected {CURRENT_ANOMALY_PIPELINE}, got {model.get('anomaly_pipeline')!r}."
            )

        classifier = _load_anomaly_classifier(Path(model_dir)) if model.get("ml_enabled") else None
        outlier_model = _load_outlier_model(Path(model_dir)) if model.get("outlier_enabled") else None
        if model.get("ml_enabled") and classifier is None:
            raise ValueError(
                f"No compatible anomaly classifier found in {Path(model_dir) / 'anomaly_model.pkl'}."
            )
        if model.get("outlier_enabled") and outlier_model is None:
            raise ValueError(
                f"No compatible outlier model found in {Path(model_dir) / 'outlier_model.pkl'}."
            )
        known_vendors = [
            _restore_vendor_display(key, model)
            for key in model.get("known_vendor_keys", [])
            if isinstance(key, str)
        ]
        known_vendors = [value for value in known_vendors if isinstance(value, str)]
        return model, classifier, outlier_model, known_vendors

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        data_root = Path(data_dir)
        test_data_root, test_path = _resolve_split_dataset_root(data_root, "test")
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        model, classifier, outlier_model, known_vendors = self._load_model(model_dir)

        predictions: list[dict[str, Any]] = []
        for record in _load_jsonl(test_path):
            prediction = _predict_single_record(
                record=record,
                data_root=test_data_root,
                model=model,
                classifier=classifier,
                outlier_model=outlier_model,
                known_vendors=known_vendors,
                include_debug=False,
            )
            predictions.append(
                {
                    "id": prediction["id"],
                    "vendor": prediction["vendor"],
                    "date": prediction["date"],
                    "total": prediction["total"],
                    "is_forged": prediction["is_forged"],
                }
            )

        _write_jsonl(out_file, predictions)

    def analyze_image(
        self,
        model_dir: str,
        image_path: str,
        include_llm_summary: bool = False,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """Local helper for UI/debugging (not used by judge harness)."""
        model, classifier, outlier_model, known_vendors = self._load_model(model_dir)
        record = {
            "id": "uploaded",
            "image_path": str(Path(image_path).resolve()),
        }
        prediction = _predict_single_record(
            record=record,
            data_root=Path("/"),
            model=model,
            classifier=classifier,
            outlier_model=outlier_model,
            known_vendors=known_vendors,
            include_debug=True,
        )
        return prediction
