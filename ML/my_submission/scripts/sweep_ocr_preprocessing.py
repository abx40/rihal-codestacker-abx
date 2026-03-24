#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

import solution


RANDOM_SEED = 42


@dataclass(frozen=True)
class EvalRecord:
    dataset: str
    split: str
    image_path: Path
    vendor: str | None
    date: str | None
    total: str | None


def _otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    sum_total = float(np.dot(np.arange(256), hist))
    sum_back = 0.0
    weight_back = 0.0
    best_var = -1.0
    threshold = 128
    for idx in range(256):
        weight_back += hist[idx]
        if weight_back <= 0:
            continue
        weight_fore = total - weight_back
        if weight_fore <= 0:
            break
        sum_back += idx * hist[idx]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if between > best_var:
            best_var = between
            threshold = idx
    return int(threshold)


def _adaptive_mean_binarize(gray_u8: np.ndarray, window: int = 25, bias: float = 10.0) -> np.ndarray:
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(gray_u8.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    h, w = gray_u8.shape

    top = np.arange(0, h)
    left = np.arange(0, w)
    bottom = top + window
    right = left + window

    sums = (
        integral[bottom[:, None], right[None, :]]
        - integral[top[:, None], right[None, :]]
        - integral[bottom[:, None], left[None, :]]
        + integral[top[:, None], left[None, :]]
    )
    means = sums / float(window * window)
    out = np.where(gray_u8.astype(np.float32) >= (means - bias), 255, 0).astype(np.uint8)
    return out


def _deskew_angle_from_binary(binary_u8: np.ndarray) -> float:
    centered = (255.0 - binary_u8.astype(np.float32)) / 255.0
    best_angle = 0.0
    best_score = -1.0
    for angle in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
        rotated = ndimage.rotate(centered, angle, reshape=False, order=1, mode="constant", cval=0.0)
        projection = rotated.sum(axis=1)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _prepare_variant(image_path: Path, variant: str) -> Path:
    if variant == "raw":
        return image_path

    with Image.open(image_path) as pil_img:
        base = ImageOps.exif_transpose(pil_img)
        gray = base.convert("L")

        if variant == "gray":
            prepared = gray
        elif variant == "resize_raw":
            prepared = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.BICUBIC)
        elif variant == "light":
            prepared = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.8)
            prepared = prepared.filter(ImageFilter.SHARPEN)
        elif variant == "exif_light":
            prepared = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.8)
            prepared = prepared.filter(ImageFilter.SHARPEN)
        elif variant == "resize_light":
            prepared = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.8)
            prepared = prepared.filter(ImageFilter.SHARPEN)
            prepared = prepared.resize((prepared.width * 2, prepared.height * 2), Image.Resampling.BICUBIC)
        elif variant == "adaptive_mean":
            boosted = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.2)
            arr = np.asarray(boosted, dtype=np.uint8)
            prepared = Image.fromarray(_adaptive_mean_binarize(arr), mode="L")
        elif variant == "otsu":
            boosted = ImageOps.autocontrast(gray, cutoff=2)
            arr = np.asarray(boosted, dtype=np.uint8)
            threshold = _otsu_threshold(arr)
            bw = np.where(arr >= threshold, 255, 0).astype(np.uint8)
            prepared = Image.fromarray(bw, mode="L")
        elif variant == "median_otsu":
            filtered = gray.filter(ImageFilter.MedianFilter(size=3))
            boosted = ImageOps.autocontrast(filtered, cutoff=2)
            arr = np.asarray(boosted, dtype=np.uint8)
            threshold = _otsu_threshold(arr)
            bw = np.where(arr >= threshold, 255, 0).astype(np.uint8)
            prepared = Image.fromarray(bw, mode="L")
        elif variant == "deskew_light":
            boosted = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.4)
            arr = np.asarray(boosted, dtype=np.uint8)
            deskew_angle = _deskew_angle_from_binary(_adaptive_mean_binarize(arr))
            prepared = boosted.rotate(deskew_angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
            prepared = prepared.filter(ImageFilter.SHARPEN)
        elif variant == "deskew_adaptive":
            boosted = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=2)).enhance(1.2)
            arr = np.asarray(boosted, dtype=np.uint8)
            deskew_angle = _deskew_angle_from_binary(_adaptive_mean_binarize(arr))
            rotated = boosted.rotate(deskew_angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
            rotated_arr = np.asarray(rotated, dtype=np.uint8)
            prepared = Image.fromarray(_adaptive_mean_binarize(rotated_arr), mode="L")
        else:
            raise ValueError(f"Unknown OCR variant: {variant}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = Path(tmp.name)
        prepared.save(temp_path)
        return temp_path


def _run_tesseract_variant(image_path: Path, psm: int, variant: str) -> str:
    temp_path: Path | None = None
    run_path = image_path
    if variant != "raw":
        temp_path = _prepare_variant(image_path, variant)
        run_path = temp_path
    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(run_path),
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
            timeout=10,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout or ""
    except subprocess.TimeoutExpired:
        return ""
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _merge_texts(texts: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for text in texts:
        for line in text.splitlines():
            cleaned = " ".join(line.strip().split())
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return "\n".join(merged)


def _candidate_text_for_variant(
    image_path: Path,
    variant: str,
    cache: dict[tuple[str, str], str],
    psms: tuple[int, ...],
) -> str:
    cache_key = (str(image_path), f"{variant}:{','.join(str(value) for value in psms)}")
    if cache_key in cache:
        return cache[cache_key]
    texts = [_run_tesseract_variant(image_path, psm=psm, variant=variant) for psm in psms]
    merged = _merge_texts(texts)
    cache[cache_key] = merged
    return merged


def _text_candidate_score(
    text: str,
    known_vendors: list[str],
) -> tuple[float, tuple[str | None, str | None, str | None]]:
    vendor = solution._extract_vendor_from_text(text, known_vendors)
    date = solution._extract_date_from_text(text)
    total = solution._extract_total_from_text(text)
    line_count, char_count, digit_ratio = solution._text_features(text)
    alpha_ratio, non_alnum_ratio, noisy_ratio, _long_token_ratio, repeat_ratio, _avg_len = solution._ocr_quality_features(text)

    score = 0.0
    score += 2.0 if vendor else 0.0
    score += 2.0 if date else 0.0
    score += 3.0 if total else 0.0
    score += 0.4 if solution.TOTAL_HINT_RE.search(text) else 0.0
    score += min(line_count / 20.0, 0.5)
    score += min(char_count / 800.0, 0.5)
    score += alpha_ratio * 0.4
    score += (1.0 - min(non_alnum_ratio, 1.0)) * 0.3
    score += (1.0 - min(noisy_ratio, 1.0)) * 0.4
    score += (1.0 - min(repeat_ratio, 1.0)) * 0.2
    if digit_ratio > 0.85:
        score -= 0.6
    return score, (vendor, date, total)


def _extract_with_pipeline(
    image_path: Path,
    pipeline_name: str,
    known_vendors: list[str],
    cache: dict[tuple[str, str], str],
    psms: tuple[int, ...],
) -> tuple[str | None, str | None, str | None]:
    if pipeline_name == "current_live":
        text = _candidate_text_for_variant(image_path, "raw", cache, psms)
        if not text.strip():
            text = _candidate_text_for_variant(image_path, "light", cache, psms)
        return (
            solution._extract_vendor_from_text(text, known_vendors),
            solution._extract_date_from_text(text),
            solution._extract_total_from_text(text),
        )

    if pipeline_name == "raw_only":
        text = _candidate_text_for_variant(image_path, "raw", cache, psms)
        return (
            solution._extract_vendor_from_text(text, known_vendors),
            solution._extract_date_from_text(text),
            solution._extract_total_from_text(text),
        )

    variant_sets = {
        "light_only": ["light"],
        "resize_raw_only": ["resize_raw"],
        "adaptive_mean_only": ["adaptive_mean"],
        "deskew_light_only": ["deskew_light"],
        "deskew_adaptive_only": ["deskew_adaptive"],
        "raw_or_light_best": ["raw", "light"],
        "raw_or_exif_light_best": ["raw", "exif_light"],
        "raw_or_adaptive_best": ["raw", "adaptive_mean"],
        "raw_or_deskew_light_best": ["raw", "deskew_light"],
        "raw_or_deskew_adaptive_best": ["raw", "deskew_adaptive"],
        "raw_or_otsu_best": ["raw", "otsu"],
        "raw_or_resize_light_best": ["raw", "resize_light"],
        "raw_light_otsu_best": ["raw", "light", "otsu"],
        "raw_light_median_otsu_best": ["raw", "light", "median_otsu"],
        "raw_gray_light_best": ["raw", "gray", "light"],
        "raw_all_best": ["raw", "gray", "light", "resize_light", "adaptive_mean", "deskew_light", "otsu", "median_otsu"],
    }
    variants = variant_sets[pipeline_name]
    best_score = -1e9
    best_triplet = (None, None, None)
    for variant in variants:
        text = _candidate_text_for_variant(image_path, variant, cache, psms)
        score, triplet = _text_candidate_score(text, known_vendors)
        if score > best_score:
            best_score = score
            best_triplet = triplet
    return best_triplet


def _load_known_vendors(model_json: Path) -> list[str]:
    payload = json.loads(model_json.read_text())
    return [
        solution._restore_vendor_display(key, payload)
        for key in payload.get("known_vendor_keys", [])
        if isinstance(key, str)
    ]


def _load_sroie_records() -> list[EvalRecord]:
    entities_dir = Path("/Users/abx/rihal/SROIE2019/test/entities")
    images_dir = Path("/Users/abx/rihal/SROIE2019/test/img")
    rows: list[EvalRecord] = []
    for entity_path in sorted(entities_dir.glob("*.txt")):
        try:
            payload = json.loads(entity_path.read_text())
        except Exception:
            continue
        image_path = images_dir / f"{entity_path.stem}.jpg"
        rows.append(
            EvalRecord(
                dataset="sroie",
                split="test",
                image_path=image_path,
                vendor=solution._normalize_vendor(payload.get("company")),
                date=solution._normalize_date(payload.get("date")),
                total=solution._normalize_total(payload.get("total")),
            )
        )
    return rows


def _cord_total_from_ground_truth(raw_ground_truth: str) -> str | None:
    try:
        payload = json.loads(raw_ground_truth)
    except Exception:
        return None
    gt_parse = payload.get("gt_parse") if isinstance(payload, dict) else {}
    total_block = gt_parse.get("total") if isinstance(gt_parse, dict) else {}
    total_raw = total_block.get("total_price") if isinstance(total_block, dict) else None
    return solution._normalize_total(total_raw)


def _load_cord_records(split: str) -> list[EvalRecord]:
    jsonl_path = Path(f"/Users/abx/rihal/rihal-codestacker/ML/my_submission/data/raw/cord/cord_{split}.jsonl")
    rows: list[EvalRecord] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        image_path = Path("/Users/abx/rihal/rihal-codestacker/ML/my_submission/data/raw/cord") / payload["image_path"]
        rows.append(
            EvalRecord(
                dataset="cord",
                split=split,
                image_path=image_path,
                vendor=None,
                date=None,
                total=_cord_total_from_ground_truth(payload.get("ground_truth", "")),
            )
        )
    return rows


def _sample_rows(rows: list[EvalRecord], limit: int | None) -> list[EvalRecord]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(RANDOM_SEED)
    return sorted(rng.sample(rows, limit), key=lambda row: str(row.image_path))


def _evaluate_pipeline(
    rows: list[EvalRecord],
    pipeline_name: str,
    known_vendors: list[str],
    cache: dict[tuple[str, str], str],
    psms: tuple[int, ...],
) -> dict[str, Any]:
    vendor_total = vendor_correct = 0
    date_total = date_correct = 0
    total_total = total_correct = 0
    all_exact_total = all_exact_correct = 0

    for row in rows:
        vendor_pred, date_pred, total_pred = _extract_with_pipeline(row.image_path, pipeline_name, known_vendors, cache, psms)

        checks = []
        if row.vendor is not None:
            vendor_total += 1
            normalized_pred = solution._normalize_vendor(vendor_pred)
            vendor_correct += int(normalized_pred == row.vendor)
            checks.append(normalized_pred == row.vendor)
        if row.date is not None:
            date_total += 1
            normalized_pred = solution._normalize_date(date_pred) if date_pred else None
            date_correct += int(normalized_pred == row.date)
            checks.append(normalized_pred == row.date)
        if row.total is not None:
            total_total += 1
            normalized_pred = solution._normalize_total(total_pred) if total_pred else None
            total_correct += int(normalized_pred == row.total)
            checks.append(normalized_pred == row.total)
        if checks:
            all_exact_total += 1
            all_exact_correct += int(all(checks))

    def pct(num: int, den: int) -> float:
        return 100.0 * num / den if den else 0.0

    return {
        "vendor_accuracy": pct(vendor_correct, vendor_total),
        "date_accuracy": pct(date_correct, date_total),
        "total_accuracy": pct(total_correct, total_total),
        "all_exact_accuracy": pct(all_exact_correct, all_exact_total),
        "counts": {
            "vendor_total": vendor_total,
            "date_total": date_total,
            "total_total": total_total,
            "all_exact_total": all_exact_total,
        },
    }


def _score_pipeline(result: dict[str, Any]) -> float:
    return (
        0.25 * result["sroie"]["vendor_accuracy"]
        + 0.20 * result["sroie"]["date_accuracy"]
        + 0.25 * result["sroie"]["total_accuracy"]
        + 0.15 * result["cord_validation"]["total_accuracy"]
        + 0.15 * result["cord_test"]["total_accuracy"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep OCR preprocessing pipelines for extraction.")
    parser.add_argument("--sample-sroie", type=int, default=120)
    parser.add_argument("--sample-cord", type=int, default=60)
    parser.add_argument("--full-top-k", type=int, default=3)
    parser.add_argument(
        "--pipelines",
        default="current_live,raw_only,light_only,raw_or_light_best,raw_or_otsu_best,raw_or_resize_light_best,raw_light_otsu_best",
    )
    parser.add_argument(
        "--report-json",
        default="/Users/abx/rihal/rihal-codestacker/ML/my_submission/reports/benchmark/ocr_preprocessing_sweep_20260317.json",
    )
    parser.add_argument("--psms", default="6")
    args = parser.parse_args()
    psms = tuple(int(part.strip()) for part in args.psms.split(",") if part.strip())

    known_vendors = _load_known_vendors(
        Path("/Users/abx/rihal/rihal-codestacker/ML/my_submission/models/latest/model/model.json")
    )

    sroie_rows = _load_sroie_records()
    cord_validation_rows = _load_cord_records("validation")
    cord_test_rows = _load_cord_records("test")

    sampled = {
        "sroie": _sample_rows(sroie_rows, args.sample_sroie),
        "cord_validation": _sample_rows(cord_validation_rows, args.sample_cord),
        "cord_test": _sample_rows(cord_test_rows, args.sample_cord),
    }

    pipelines = [item.strip() for item in args.pipelines.split(",") if item.strip()]

    cache: dict[tuple[str, str], str] = {}
    sample_results: list[dict[str, Any]] = []
    for pipeline_name in pipelines:
        print(f"[sample] {pipeline_name}", flush=True)
        result = {
            "pipeline": pipeline_name,
            "sroie": _evaluate_pipeline(sampled["sroie"], pipeline_name, known_vendors, cache, psms),
            "cord_validation": _evaluate_pipeline(sampled["cord_validation"], pipeline_name, known_vendors, cache, psms),
            "cord_test": _evaluate_pipeline(sampled["cord_test"], pipeline_name, known_vendors, cache, psms),
        }
        result["score"] = _score_pipeline(result)
        sample_results.append(result)

    sample_results.sort(key=lambda item: item["score"], reverse=True)
    top_pipelines = [row["pipeline"] for row in sample_results[: max(0, args.full_top_k)]]

    full_results: list[dict[str, Any]] = []
    for pipeline_name in top_pipelines:
        print(f"[full] {pipeline_name}", flush=True)
        result = {
            "pipeline": pipeline_name,
            "sroie": _evaluate_pipeline(sroie_rows, pipeline_name, known_vendors, cache, psms),
            "cord_validation": _evaluate_pipeline(cord_validation_rows, pipeline_name, known_vendors, cache, psms),
            "cord_test": _evaluate_pipeline(cord_test_rows, pipeline_name, known_vendors, cache, psms),
        }
        result["score"] = _score_pipeline(result)
        full_results.append(result)
    full_results.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "sample_sizes": {
            "sroie": len(sampled["sroie"]),
            "cord_validation": len(sampled["cord_validation"]),
            "cord_test": len(sampled["cord_test"]),
        },
        "sample_results": sample_results,
        "full_results": full_results,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
