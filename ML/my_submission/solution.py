"""
DocFusion challenge solution.

Key capabilities:
- Interface-compliant DocFusionSolution for judge harness.
- OCR-based extraction for vendor/date/total with multi-pass tesseract.
- Hybrid anomaly prediction: rule-based score + optional sklearn classifier.
- Extra `analyze_image()` helper for local UI and debugging.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
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
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False


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
AMOUNT_RE = re.compile(r"(?<!\d)(\d+(?:[.,]\d{2}))(?!\d)")
TOTAL_HINT_RE = re.compile(r"(total|amount\s*due|grand\s*total|net\s*total)", re.I)
TOTAL_EXCLUSION_RE = re.compile(
    r"(subtotal|sub[- ]?total|tax|vat|discount|cash|change|paid|balance|rounding|qty|quantity|item)",
    re.I,
)
VENDOR_LABEL_RE = re.compile(
    r"(?:vendor|merchant|store|seller|supplier|from)\s*[:\-]\s*(.+)$",
    re.I,
)


@dataclass
class OCRLine:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    conf: float


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
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    token = match.group(0).replace(",", ".")
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


def _anomaly_training_records(train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = [
        record
        for record in train_records
        if isinstance(record.get("label"), dict) and "is_forged" in record["label"]
    ]
    if not labeled:
        return []

    find_it_again = [
        record
        for record in labeled
        if str(record.get("source", "")).strip().lower() == "find_it_again"
    ]
    find_it_again_classes = {
        int(bool(record.get("label", {}).get("is_forged", 0)))
        for record in find_it_again
    }

    if len(find_it_again) >= 40 and len(find_it_again_classes) >= 2:
        return find_it_again
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
            scored_candidates.append((score, amount))

    if scored_candidates:
        scored_candidates.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return f"{scored_candidates[0][1]:.2f}"

    all_candidates = [_safe_float(token) for token in AMOUNT_RE.findall(text)]
    all_candidates = [value for value in all_candidates if value is not None]
    if all_candidates:
        return f"{max(all_candidates):.2f}"
    return None


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


def _extract_vendor_from_text(text: str, known_vendors: list[str]) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    for line in lines[:16]:
        label_match = VENDOR_LABEL_RE.search(line)
        if not label_match:
            continue
        candidate = label_match.group(1).strip()
        mapped = _best_vendor_match(candidate, known_vendors)
        if mapped:
            return mapped
        cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 3:
            return cleaned

    for line in lines[:12]:
        match = _best_vendor_match(line, known_vendors)
        if match:
            return match

    for line in lines[:12]:
        if any(token in line.lower() for token in ("date", "invoice", "receipt", "total")):
            continue
        if re.search(r"\d", line) and len(re.findall(r"[A-Za-z]", line)) < 6:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", line).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) >= 3 and len(cleaned.split()) <= 7:
            return cleaned
    return None


def _extract_with_ocr(
    image_path: Path,
    known_vendors: list[str],
    include_lines: bool = False,
) -> tuple[str | None, str | None, str | None, str, list[OCRLine]]:
    text = _ocr_text(image_path)
    vendor = _extract_vendor_from_text(text, known_vendors)
    date = _extract_date_from_text(text)
    total = _extract_total_from_text(text)
    lines = _ocr_lines(image_path) if include_lines else []
    return vendor, date, total, text, lines


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


def _build_model(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly_records = _anomaly_training_records(train_records)

    vendor_counts: dict[str, int] = {}
    vendor_anomaly_counts: dict[str, int] = {}
    vendor_fraud_counts: dict[str, int] = {}
    vendor_name_counts: dict[str, dict[str, int]] = {}
    genuine_totals: list[float] = []
    all_totals: list[float] = []
    vendor_genuine_totals: dict[str, list[float]] = {}
    fraud_count = 0

    for record in train_records:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        is_forged = int(bool(label.get("is_forged", 0)))

        vendor, _date, total = _extract_from_fields(record)
        vendor_key = _normalize_vendor(vendor)
        if vendor_key:
            vendor_counts[vendor_key] = vendor_counts.get(vendor_key, 0) + 1
            display_vendor = str(vendor).strip() if vendor else vendor_key.title()
            per_vendor_names = vendor_name_counts.setdefault(vendor_key, {})
            per_vendor_names[display_vendor] = per_vendor_names.get(display_vendor, 0) + 1

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
    patch_classifier: Any | None,
    known_vendors: list[str],
    include_debug: bool,
) -> dict[str, Any]:
    record_id = record.get("id")
    if not isinstance(record_id, str):
        record_id = "unknown"

    vendor, date, total = _extract_from_fields(record)
    text = ""
    ocr_lines: list[OCRLine] = []
    image_rel = record.get("image_path")
    image_path = _resolve_image_path(data_root, image_rel)
    image_forensics = _image_forensics_features(image_path)
    need_ocr_lines = include_debug or patch_classifier is not None

    if vendor is None or date is None or total is None or need_ocr_lines:
        ocr_vendor, ocr_date, ocr_total, text, ocr_lines = _extract_with_ocr(
            image_path=image_path,
            known_vendors=known_vendors,
            include_lines=need_ocr_lines,
        )
        if vendor is None:
            vendor = ocr_vendor
        if date is None:
            date = ocr_date
        if total is None:
            total = ocr_total

    if not text and classifier is not None and image_path.exists():
        text = _run_tesseract(image_path, psm=6)

    vendor = _canonicalize_vendor(vendor, model)
    date = _normalize_date(date) if date else None
    total = _normalize_total(total)
    ocr_quality = _ocr_quality_features(text)

    rule_score = _fraud_score(vendor, date, total, model)
    rule_prob = _sigmoid(rule_score)
    threshold_rule = _sigmoid(float(model.get("threshold", 0.25)))

    probability = rule_prob
    threshold = threshold_rule
    ml_probability = None
    patch_probability = None
    patch_threshold = float(model.get("patch_threshold", 0.60))
    patch_candidate_count = 0
    patch_best_bbox: dict[str, int] | None = None

    if classifier is not None:
        feats = _feature_vector(
            vendor,
            date,
            total,
            text,
            model,
            ocr_quality=ocr_quality,
            image_forensics=image_forensics,
        )
        try:
            ml_probability = float(classifier.predict_proba([feats])[0][1])
            probability = 0.65 * ml_probability + 0.35 * rule_prob
            threshold = float(model.get("ml_threshold", 0.5))
        except Exception:
            ml_probability = None

    if patch_classifier is not None and image_path.exists():
        rgb_arr = _image_rgb_array(image_path)
        if rgb_arr is not None:
            image_height = int(rgb_arr.shape[0])
            image_width = int(rgb_arr.shape[1])
            if not ocr_lines:
                ocr_lines = _ocr_lines(image_path)
            if ocr_lines and image_width > 2 and image_height > 2:
                candidate_bboxes = _candidate_patch_bboxes_from_ocr_lines(
                    ocr_lines,
                    image_width=image_width,
                    image_height=image_height,
                )
                patch_candidate_count = len(candidate_bboxes)
                best_prob = -1.0
                best_bbox: tuple[int, int, int, int] | None = None
                for bbox in candidate_bboxes:
                    patch_feats = _patch_features_from_bbox(rgb_arr, bbox)
                    if patch_feats is None:
                        continue
                    try:
                        patch_prob = float(patch_classifier.predict_proba([patch_feats])[0][1])
                    except Exception:
                        continue
                    if not math.isfinite(patch_prob):
                        continue
                    if patch_prob > best_prob:
                        best_prob = patch_prob
                        best_bbox = bbox
                if best_prob >= 0.0:
                    patch_probability = min(max(best_prob, 0.0), 1.0)
                    if best_bbox is not None:
                        patch_best_bbox = {
                            "left": int(best_bbox[0]),
                            "top": int(best_bbox[1]),
                            "right": int(best_bbox[2]),
                            "bottom": int(best_bbox[3]),
                        }

    if patch_probability is not None:
        if patch_probability >= patch_threshold:
            probability = max(probability, (0.35 * probability) + (0.65 * patch_probability))
        else:
            probability = (0.92 * probability) + (0.08 * patch_probability)
        probability = min(max(probability, 0.0), 1.0)

    is_forged = int(probability >= threshold)

    prediction: dict[str, Any] = {
        "id": record_id,
        "vendor": vendor,
        "date": date,
        "total": total,
        "is_forged": is_forged,
    }

    if include_debug:
        boxes = _find_field_boxes(ocr_lines, vendor, date, total)
        if patch_best_bbox is not None:
            boxes["patch_alert"] = patch_best_bbox
        reasons = _suspicious_reasons(
            vendor,
            date,
            total,
            text,
            model,
            ocr_quality=ocr_quality,
            image_forensics=image_forensics,
        )
        if patch_probability is not None and patch_probability >= patch_threshold:
            reasons.append("Local visual tampering signal near numeric text")
        prediction["_debug"] = {
            "probability": probability,
            "rule_probability": rule_prob,
            "ml_probability": ml_probability,
            "patch_probability": patch_probability,
            "patch_threshold": patch_threshold,
            "patch_candidate_count": patch_candidate_count,
            "threshold": threshold,
            "reasons": reasons,
            "ocr_quality_features": {
                name: float(value) for name, value in zip(OCR_QUALITY_FEATURE_NAMES, ocr_quality)
            },
            "image_forensics_features": {
                name: float(value) for name, value in zip(IMAGE_FORENSICS_FEATURE_NAMES, image_forensics)
            },
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
                for line in ocr_lines
            ],
        }

    return prediction


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        train_root = Path(train_dir)
        train_path = train_root / "train.jsonl"
        model_dir = Path(work_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_records = _load_jsonl(train_path)
        model = _build_model(train_records)

        trained = _train_ml_classifier(train_records, train_root, model)
        if trained is not None:
            classifier, ml_threshold = trained
            _save_anomaly_classifier(model_dir, classifier)
            model["ml_enabled"] = True
            model["ml_threshold"] = float(ml_threshold)

        trained_patch = _train_patch_classifier(train_records, train_root)
        if trained_patch is not None:
            patch_classifier, patch_threshold = trained_patch
            _save_patch_classifier(model_dir, patch_classifier)
            model["patch_enabled"] = True
            model["patch_threshold"] = float(patch_threshold)

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
                "ml_enabled": False,
                "ml_threshold": 0.5,
                "patch_enabled": False,
                "patch_threshold": 0.60,
            }

        if "ml_enabled" not in model:
            model["ml_enabled"] = False
        if "ml_threshold" not in model:
            model["ml_threshold"] = 0.5
        if "patch_enabled" not in model:
            model["patch_enabled"] = False
        if "patch_threshold" not in model:
            model["patch_threshold"] = 0.60

        classifier = _load_anomaly_classifier(Path(model_dir)) if model.get("ml_enabled") else None
        patch_classifier = _load_patch_classifier(Path(model_dir)) if model.get("patch_enabled") else None
        known_vendors = [
            _restore_vendor_display(key, model)
            for key in model.get("known_vendor_keys", [])
            if isinstance(key, str)
        ]
        known_vendors = [value for value in known_vendors if isinstance(value, str)]
        return model, classifier, patch_classifier, known_vendors

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        data_root = Path(data_dir)
        test_path = data_root / "test.jsonl"
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        model, classifier, patch_classifier, known_vendors = self._load_model(model_dir)

        predictions: list[dict[str, Any]] = []
        for record in _load_jsonl(test_path):
            prediction = _predict_single_record(
                record=record,
                data_root=data_root,
                model=model,
                classifier=classifier,
                patch_classifier=patch_classifier,
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
        model, classifier, patch_classifier, known_vendors = self._load_model(model_dir)
        record = {
            "id": "uploaded",
            "image_path": str(Path(image_path).resolve()),
        }
        prediction = _predict_single_record(
            record=record,
            data_root=Path("/"),
            model=model,
            classifier=classifier,
            patch_classifier=patch_classifier,
            known_vendors=known_vendors,
            include_debug=True,
        )
        _attach_anomaly_summary(
            prediction=prediction,
            llm_enabled=include_llm_summary,
            llm_model=llm_model,
        )
        return prediction
