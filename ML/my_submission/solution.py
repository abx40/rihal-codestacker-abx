"""
DocFusion baseline submission.

This baseline is designed to be robust for the public checker and practical as a
starting point for the private benchmark:
- Uses provided `fields` directly when available.
- Falls back to OCR (tesseract CLI) + regex extraction when fields are absent.
- Trains lightweight fraud heuristics from train labels and numeric patterns.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any


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


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(variance)
    if std < 1e-6:
        std = 1.0
    return mean, std


def _logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


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


def _run_tesseract(image_path: Path, psm: int = 6) -> str:
    if not image_path.exists():
        return ""
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--psm",
                str(psm),
                "--oem",
                "1",
                "-l",
                "eng",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=6,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


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
    return _merge_texts([primary, secondary])


def _extract_date_from_text(text: str) -> str | None:
    for match in DATE_CANDIDATE_RE.findall(text):
        normalized = _normalize_date(match)
        if normalized:
            return normalized

    for match in DATE_TEXTUAL_RE.findall(text):
        normalized = _normalize_date(match)
        if normalized:
            return normalized

    # OCR often outputs digits with spaces: "2024 01 09".
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

        line_lower = line.lower()
        line_score = 0.0
        if TOTAL_HINT_RE.search(line):
            line_score += 3.0
        if TOTAL_EXCLUSION_RE.search(line):
            line_score -= 1.5

        for token in tokens:
            amount = _safe_float(token)
            if amount is None:
                continue
            score = line_score
            if amount > 0:
                score += min(amount / 1000.0, 0.75)
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

    # First attempt: map first lines to known vendor names.
    for line in lines[:12]:
        match = _best_vendor_match(line, known_vendors)
        if match:
            return match

    # Fallback: top-most alphabetic line without numeric clutter.
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


def _extract_with_ocr(image_path: Path, known_vendors: list[str]) -> tuple[str | None, str | None, str | None, str]:
    text = _ocr_text(image_path)
    vendor = _extract_vendor_from_text(text, known_vendors)
    date = _extract_date_from_text(text)
    total = _extract_total_from_text(text)
    return vendor, date, total, text


def _build_model(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    vendor_counts: dict[str, int] = {}
    vendor_fraud_counts: dict[str, int] = {}
    vendor_name_counts: dict[str, dict[str, int]] = {}
    genuine_totals: list[float] = []
    all_totals: list[float] = []
    vendor_genuine_totals: dict[str, list[float]] = {}
    fraud_count = 0

    for record in train_records:
        label = record.get("label") if isinstance(record.get("label"), dict) else {}
        is_forged = int(bool(label.get("is_forged", 0)))
        fraud_count += is_forged

        vendor, date, total = _extract_from_fields(record)
        _ = date  # Date is used at inference but not required in train statistics.
        vendor_key = _normalize_vendor(vendor)
        if vendor_key:
            vendor_counts[vendor_key] = vendor_counts.get(vendor_key, 0) + 1
            vendor_fraud_counts[vendor_key] = vendor_fraud_counts.get(vendor_key, 0) + is_forged
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

    known_vendor_keys = sorted(vendor_counts, key=lambda key: (-vendor_counts[key], key))
    overall_rate = fraud_count / max(1, len(train_records))
    overall_mean, overall_std = _mean_std(genuine_totals if genuine_totals else all_totals)

    vendor_fraud_rate: dict[str, float] = {}
    for vendor_key, count in vendor_counts.items():
        forged = vendor_fraud_counts.get(vendor_key, 0)
        # Laplace smoothing.
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
    }

    # Calibrate threshold on train for basic stability.
    candidate_thresholds = [round(v / 20.0, 2) for v in range(-20, 21)]
    best_threshold = model["threshold"]
    best_accuracy = -1.0
    for threshold in candidate_thresholds:
        correct = 0
        for record in train_records:
            label = record.get("label") if isinstance(record.get("label"), dict) else {}
            target = int(bool(label.get("is_forged", 0)))
            vendor, date, total = _extract_from_fields(record)
            score = _fraud_score(vendor, date, total, model)
            prediction = int(score >= threshold)
            if prediction == target:
                correct += 1
        accuracy = correct / max(1, len(train_records))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
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
    for key in known_keys:
        if key in vendor_key or vendor_key in key:
            return _restore_vendor_display(key, model)
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


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        train_path = Path(train_dir) / "train.jsonl"
        model_dir = Path(work_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_records = _load_jsonl(train_path)
        model = _build_model(train_records)

        model_path = model_dir / "model.json"
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2, sort_keys=True)
        return str(model_dir)

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        data_root = Path(data_dir)
        test_path = data_root / "test.jsonl"
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        model_path = Path(model_dir) / "model.json"
        if model_path.exists():
            with model_path.open("r", encoding="utf-8") as handle:
                model = json.load(handle)
        else:
            # Safe fallback if train artifacts are unavailable.
            model = {
                "known_vendor_keys": [],
                "vendor_display_names": {},
                "global_fraud_rate": 0.5,
                "global_total_mean": 0.0,
                "global_total_std": 1.0,
                "vendor_fraud_rate": {},
                "vendor_total_stats": {},
                "threshold": 0.25,
            }

        known_vendors = [
            _restore_vendor_display(key, model)
            for key in model.get("known_vendor_keys", [])
            if isinstance(key, str)
        ]
        known_vendors = [value for value in known_vendors if isinstance(value, str)]

        predictions: list[dict[str, Any]] = []
        for record in _load_jsonl(test_path):
            record_id = record.get("id")
            if not isinstance(record_id, str):
                continue

            vendor, date, total = _extract_from_fields(record)

            # OCR fallback for records without structured fields.
            if vendor is None or date is None or total is None:
                image_rel = record.get("image_path")
                image_path = data_root / str(image_rel) if image_rel else Path("")
                ocr_vendor, ocr_date, ocr_total, _ = _extract_with_ocr(image_path, known_vendors)
                if vendor is None:
                    vendor = ocr_vendor
                if date is None:
                    date = ocr_date
                if total is None:
                    total = ocr_total

            vendor = _canonicalize_vendor(vendor, model)
            date = _normalize_date(date) if date else None
            total = _normalize_total(total)

            score = _fraud_score(vendor, date, total, model)
            threshold = float(model.get("threshold", 0.25))
            is_forged = int(score >= threshold)

            predictions.append(
                {
                    "id": record_id,
                    "vendor": vendor,
                    "date": date,
                    "total": total,
                    "is_forged": is_forged,
                }
            )

        _write_jsonl(out_file, predictions)
