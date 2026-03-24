#!/usr/bin/env python3
"""
Build a unified DocFusion training dataset from raw sources.

Input sources (any subset):
- data/raw/sroie/
- data/raw/cord/
- data/raw/find_it_again/

Output schema:
- <out-root>/train/images/*
- <out-root>/train/train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DATE_CANDIDATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
AMOUNT_RE = re.compile(
    r"(?<!\d)(-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|-?\d+(?:[.,]\d{2})|-?\d{2,})(?!\d)"
)
TOTAL_HINT_RE = re.compile(r"(total|amount\s*due|grand\s*total|net\s*total)", re.I)


@dataclass
class UnifiedRecord:
    source: str
    source_id: str
    image_path: Path
    vendor: str | None
    date: str | None
    total: str | None
    is_forged: int | None
    forgery_regions: list[dict[str, int]] | None = None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _normalize_total(value: Any) -> str | None:
    if value is None:
        return None
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
        amount = float(token)
    except ValueError:
        return None
    return f"{amount:.2f}"


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().replace(",", " "))
    fmts = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m.%d.%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_vendor(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_fields_from_ocr_text(text: str) -> tuple[str | None, str | None, str | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    vendor: str | None = None
    for line in lines[:10]:
        if re.search(r"\d", line) and len(re.findall(r"[A-Za-z]", line)) < 6:
            continue
        low = line.lower()
        if any(tok in low for tok in ("invoice", "date", "total", "tax", "cash", "change")):
            continue
        cleaned = re.sub(r"[^A-Za-z0-9 &.-]+", " ", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 3 and len(cleaned.split()) <= 9:
            vendor = cleaned
            break

    date: str | None = None
    for match in DATE_CANDIDATE_RE.findall(text):
        normalized = _normalize_date(match)
        if normalized:
            date = normalized
            break

    total: str | None = None
    scored: list[tuple[float, str]] = []
    for line in lines:
        tokens = AMOUNT_RE.findall(line)
        if not tokens:
            continue
        score = 0.0
        if TOTAL_HINT_RE.search(line):
            score += 3.0
        for token in tokens:
            scored.append((score, token))

    if scored:
        scored.sort(key=lambda item: (item[0], _safe_num(item[1])), reverse=True)
        total = _normalize_total(scored[0][1])
    return vendor, date, total


def _safe_num(value: str) -> float:
    token = value.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return -1.0


def _parse_forgery_regions(value: Any) -> list[dict[str, int]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "0":
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    else:
        parsed = value

    if not isinstance(parsed, dict):
        return []
    regions = parsed.get("regions")
    if not isinstance(regions, list):
        return []

    out: list[dict[str, int]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        shape = region.get("shape_attributes")
        if not isinstance(shape, dict):
            continue
        try:
            x = int(shape.get("x", 0))
            y = int(shape.get("y", 0))
            width = int(shape.get("width", 0))
            height = int(shape.get("height", 0))
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        out.append({"x": x, "y": y, "width": width, "height": height})
    return out


def _try_parse_json_text(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _find_first_value_by_key(obj: Any, patterns: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(pattern in key_l for pattern in patterns):
                if isinstance(value, (str, int, float)):
                    return value
            nested = _find_first_value_by_key(value, patterns)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_first_value_by_key(item, patterns)
            if nested is not None:
                return nested
    return None


def _parse_sroie_dir(root: Path) -> list[UnifiedRecord]:
    if not root.exists():
        return []

    images = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpeg"))
    records: list[UnifiedRecord] = []
    for image in images:
        stem = image.stem
        key_candidates = list(root.rglob(f"{stem}.txt")) + list(root.rglob(f"{stem}.json"))
        vendor = None
        date = None
        total = None
        for key_file in key_candidates:
            text = key_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            obj = _try_parse_json_text(text)
            if obj is None:
                kv = {}
                for line in text.splitlines():
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    kv[k.strip()] = v.strip()
                obj = kv
            vendor = _normalize_vendor(
                _find_first_value_by_key(obj, ("company", "vendor", "store", "supplier", "merchant"))
            )
            date = _normalize_date(_find_first_value_by_key(obj, ("date",)))
            total = _normalize_total(_find_first_value_by_key(obj, ("total", "amount")))
            if vendor or date or total:
                break

        records.append(
            UnifiedRecord(
                source="sroie",
                source_id=stem,
                image_path=image,
                vendor=vendor,
                date=date,
                total=total,
                is_forged=0,
            )
        )
    return records


def _parse_sroie(raw_root: Path) -> list[UnifiedRecord]:
    root = raw_root / "sroie"
    return _parse_sroie_dir(root)


def _parse_cord(raw_root: Path) -> list[UnifiedRecord]:
    root = raw_root / "cord"
    if not root.exists():
        return []

    jsonls = sorted(root.glob("cord_*.jsonl"))
    records: list[UnifiedRecord] = []
    for jsonl in jsonls:
        for row in _load_jsonl(jsonl):
            source_id = str(row.get("id", ""))
            image_rel = row.get("image_path")
            if not source_id or not image_rel:
                continue
            image_path = root / str(image_rel)
            if not image_path.exists():
                continue

            gt_raw = row.get("ground_truth")
            gt_obj: Any = None
            if isinstance(gt_raw, str):
                gt_obj = _try_parse_json_text(gt_raw)
            elif isinstance(gt_raw, dict):
                gt_obj = gt_raw

            vendor = _normalize_vendor(
                _find_first_value_by_key(gt_obj, ("vendor", "store", "company", "supplier", "merchant"))
            )
            date = _normalize_date(_find_first_value_by_key(gt_obj, ("date",)))
            total = _normalize_total(_find_first_value_by_key(gt_obj, ("total", "amount", "sum")))

            records.append(
                UnifiedRecord(
                    source="cord",
                    source_id=source_id,
                    image_path=image_path,
                    vendor=vendor,
                    date=date,
                    total=total,
                    is_forged=0,
                )
            )
    return records


def _normalize_find_it_again_splits(raw: str) -> set[str]:
    token = str(raw).strip().lower()
    if not token:
        return {"train"}
    if token in {"all", "*"}:
        return {"train", "val", "test"}

    pieces = [part.strip().lower() for part in token.split(",") if part.strip()]
    allowed = {"train", "val", "test"}
    selected = {part for part in pieces if part in allowed}
    if not selected:
        return {"train"}
    return selected


def _collect_find_it_again_stems(root: Path, include_splits: set[str] | None = None) -> set[str]:
    if not root.exists():
        return set()
    selected_splits = include_splits or {"train", "val", "test"}
    stems: set[str] = set()
    for split in ("train", "val", "test"):
        if split not in selected_splits:
            continue
        split_meta = root / f"{split}.txt"
        if not split_meta.exists():
            continue
        with split_meta.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("image") or row.get("filename") or row.get("image_path")
                if image_name:
                    stems.add(Path(str(image_name)).stem)
    return stems


def _parse_find_it_again_dir(root: Path, include_splits: set[str] | None = None) -> list[UnifiedRecord]:
    if not root.exists():
        return []
    selected_splits = include_splits or {"train", "val", "test"}

    records: list[UnifiedRecord] = []
    labels_jsonl = root / "labels.jsonl"
    labels_csv = root / "labels.csv"

    if labels_jsonl.exists():
        rows = _load_jsonl(labels_jsonl)
        for row in rows:
            image_rel = row.get("image_path")
            source_id = str(row.get("id", Path(str(image_rel or "sample")).stem))
            image_path = (root / str(image_rel)).resolve() if image_rel else None
            if image_path is None or not image_path.exists():
                continue
            records.append(
                UnifiedRecord(
                    source="find_it_again",
                    source_id=source_id,
                    image_path=image_path,
                    vendor=_normalize_vendor(row.get("vendor")),
                    date=_normalize_date(row.get("date")),
                    total=_normalize_total(row.get("total")),
                    is_forged=int(bool(row.get("is_forged", 0))),
                    forgery_regions=_parse_forgery_regions(row.get("forgery_annotations")),
                )
            )
        return records

    if labels_csv.exists():
        with labels_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_rel = row.get("image_path")
                source_id = str(row.get("id", Path(str(image_rel or "sample")).stem))
                image_path = (root / str(image_rel)).resolve() if image_rel else None
                if image_path is None or not image_path.exists():
                    continue
                records.append(
                    UnifiedRecord(
                        source="find_it_again",
                        source_id=source_id,
                        image_path=image_path,
                        vendor=_normalize_vendor(row.get("vendor")),
                        date=_normalize_date(row.get("date")),
                        total=_normalize_total(row.get("total")),
                        is_forged=int(bool(int(row.get("is_forged", "0")))),
                        forgery_regions=_parse_forgery_regions(row.get("forgery_annotations")),
                    )
                )
        return records

    excluded_stems: set[str] = set()
    for split in ("train", "val", "test"):
        if split in selected_splits:
            continue
        split_meta = root / f"{split}.txt"
        if not split_meta.exists():
            continue
        with split_meta.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("image") or row.get("filename") or row.get("image_path")
                if image_name:
                    excluded_stems.add(Path(str(image_name)).stem)

    for split in ("train", "val", "test"):
        if split not in selected_splits:
            continue
        split_meta = root / f"{split}.txt"
        split_dir = root / split
        if not split_meta.exists() or not split_dir.exists():
            continue
        with split_meta.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("image") or row.get("filename") or row.get("image_path")
                if not image_name:
                    continue
                image_path = split_dir / str(image_name)
                if not image_path.exists():
                    continue

                stem = image_path.stem
                if stem in excluded_stems:
                    continue
                txt_path = split_dir / f"{stem}.txt"
                text = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
                vendor, date, total = _extract_fields_from_ocr_text(text)

                forged_value = row.get("forged", "0")
                try:
                    is_forged = int(bool(int(str(forged_value))))
                except ValueError:
                    is_forged = 0
                forgery_regions = _parse_forgery_regions(row.get("forgery annotations"))

                records.append(
                    UnifiedRecord(
                        source="find_it_again",
                        source_id=stem,
                        image_path=image_path.resolve(),
                        vendor=vendor,
                        date=date,
                        total=total,
                        is_forged=is_forged,
                        forgery_regions=forgery_regions,
                    )
                )
    return records


def _parse_find_it_again(raw_root: Path, include_splits: set[str]) -> list[UnifiedRecord]:
    root = raw_root / "find_it_again"
    return _parse_find_it_again_dir(root, include_splits=include_splits)


def _parse_public_dummy(dummy_root: Path) -> list[UnifiedRecord]:
    train_path = dummy_root / "train" / "train.jsonl"
    if not train_path.exists():
        return []
    records: list[UnifiedRecord] = []
    for row in _load_jsonl(train_path):
        image_rel = row.get("image_path")
        if not image_rel:
            continue
        image_path = (dummy_root / "train" / str(image_rel)).resolve()
        if not image_path.exists():
            continue
        fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
        label = row.get("label") if isinstance(row.get("label"), dict) else {}
        records.append(
            UnifiedRecord(
                source="public_dummy",
                source_id=str(row.get("id", image_path.stem)),
                image_path=image_path,
                vendor=_normalize_vendor(fields.get("vendor")),
                date=_normalize_date(fields.get("date")),
                total=_normalize_total(fields.get("total")),
                is_forged=int(bool(label.get("is_forged", 0))),
            )
        )
    return records


def _write_unified(records: list[UnifiedRecord], out_root: Path, seed: int) -> None:
    random.seed(seed)
    out_train = out_root / "train"
    out_images = out_train / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    random.shuffle(records)
    out_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        image_ext = record.image_path.suffix.lower() if record.image_path.suffix else ".png"
        out_name = f"{record.source}_{record.source_id}_{idx:06d}{image_ext}"
        dst_path = out_images / out_name
        shutil.copy2(record.image_path, dst_path)

        label_payload: dict[str, Any] = {
            "is_forged": int(record.is_forged) if record.is_forged is not None else 0,
        }
        if record.forgery_regions:
            label_payload["forgery_regions"] = record.forgery_regions

        out_rows.append(
            {
                "id": f"{record.source}_{record.source_id}_{idx:06d}",
                "image_path": f"images/{out_name}",
                "fields": {
                    "vendor": record.vendor,
                    "date": record.date,
                    "total": record.total,
                },
                "label": label_payload,
                "source": record.source,
            }
        )

    train_jsonl = out_train / "train.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    stats: dict[str, int] = {}
    for row in out_rows:
        source = str(row.get("source"))
        stats[source] = stats.get(source, 0) + 1

    stats_path = out_root / "stats.json"
    stats_payload = {
        "records": len(out_rows),
        "sources": stats,
        "path": str(out_root),
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build] wrote {len(out_rows)} records -> {train_jsonl}")
    print(f"[build] source stats -> {stats_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified DocFusion dataset")
    parser.add_argument("--raw-root", default="./data/raw", help="Raw sources directory")
    parser.add_argument("--out-root", default="./data/processed/unified", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse sources and print counts without copying files or writing outputs",
    )
    parser.add_argument(
        "--sroie-dir",
        default="",
        help="Optional absolute path to SROIE root (e.g., /Users/abx/rihal/SROIE2019)",
    )
    parser.add_argument(
        "--find-it-again-dir",
        default="",
        help="Optional absolute path to Find-It-Again root (e.g., /Users/abx/rihal/findit2)",
    )
    parser.add_argument(
        "--find-it-again-splits",
        default="train",
        help="Comma-separated splits to include from Find-It-Again (train,val,test) or 'all'. Default: train",
    )
    parser.add_argument(
        "--dummy-root",
        default="../dummy_data",
        help="Optional fallback source when real datasets are missing",
    )
    parser.add_argument(
        "--allow-source-overlap",
        action="store_true",
        help="Keep overlapping SROIE / Find-It-Again records instead of filtering them out.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    dummy_root = Path(args.dummy_root).resolve()
    external_sroie = Path(args.sroie_dir).resolve() if args.sroie_dir else None
    external_find_it_again = Path(args.find_it_again_dir).resolve() if args.find_it_again_dir else None
    include_find_it_again_splits = _normalize_find_it_again_splits(args.find_it_again_splits)

    find_it_again_stems: set[str] = set()
    find_it_again_roots = [raw_root / "find_it_again"]
    if external_find_it_again and external_find_it_again.exists():
        find_it_again_roots.append(external_find_it_again)
    for candidate_root in find_it_again_roots:
        find_it_again_stems.update(_collect_find_it_again_stems(candidate_root, include_splits={"train", "val", "test"}))

    records: list[UnifiedRecord] = []
    sroie_records = _parse_sroie(raw_root)
    if find_it_again_stems and not args.allow_source_overlap:
        before = len(sroie_records)
        sroie_records = [record for record in sroie_records if record.source_id not in find_it_again_stems]
        removed = before - len(sroie_records)
        if removed:
            print(f"[build] filtered {removed} SROIE records overlapping Find-It-Again stems")
    records.extend(sroie_records)
    records.extend(_parse_cord(raw_root))
    records.extend(_parse_find_it_again(raw_root, include_splits=include_find_it_again_splits))

    if external_sroie and external_sroie.exists():
        external_sroie_records = _parse_sroie_dir(external_sroie)
        if find_it_again_stems and not args.allow_source_overlap:
            before = len(external_sroie_records)
            external_sroie_records = [
                record for record in external_sroie_records if record.source_id not in find_it_again_stems
            ]
            removed = before - len(external_sroie_records)
            if removed:
                print(f"[build] filtered {removed} external SROIE records overlapping Find-It-Again stems")
        records.extend(external_sroie_records)
    if external_find_it_again and external_find_it_again.exists():
        records.extend(
            _parse_find_it_again_dir(
                external_find_it_again,
                include_splits=include_find_it_again_splits,
            )
        )

    if args.dry_run:
        source_counts: dict[str, int] = {}
        forged_counts: dict[str, int] = {}
        for record in records:
            source_counts[record.source] = source_counts.get(record.source, 0) + 1
            if record.is_forged:
                forged_counts[record.source] = forged_counts.get(record.source, 0) + 1

        payload = {
            "records": len(records),
            "source_counts": source_counts,
            "forged_counts": forged_counts,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not records and dummy_root.exists():
        print("[build] no raw sources found; using public dummy_data fallback.")
        records.extend(_parse_public_dummy(dummy_root))

    if not records:
        print("[build] no records found. Populate data/raw first via scripts/download_data.py.")
        return 1

    _write_unified(records, out_root, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
