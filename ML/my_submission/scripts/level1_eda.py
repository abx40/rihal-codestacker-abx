#!/usr/bin/env python3
"""
Level 1 EDA helper for the DocFusion challenge.

Outputs:
- summary.md
- vendor_frequency.csv
- monthly_totals.csv
- outlier_candidates.csv
- (optional) vendor_frequency.png / total_distribution.png if matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def parse_total(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    token = "".join(ch for ch in str(value) if ch.isdigit() or ch in {".", ","})
    if not token:
        return None
    token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
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
    )
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    std = statistics.stdev(values)
    return std if std > 1e-9 else 1.0


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_optional_plots(
    out_dir: Path,
    vendor_counts: Counter[str],
    totals: list[float],
) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    top_vendors = vendor_counts.most_common(10)
    if top_vendors:
        labels = [v for v, _ in top_vendors]
        counts = [c for _, c in top_vendors]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, counts)
        plt.xticks(rotation=30, ha="right")
        plt.title("Top Vendors by Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "vendor_frequency.png", dpi=150)
        plt.close()

    if totals:
        plt.figure(figsize=(10, 4))
        plt.hist(totals, bins=min(20, max(5, int(math.sqrt(len(totals))))), edgecolor="black")
        plt.title("Total Amount Distribution")
        plt.xlabel("Total")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "total_distribution.png", dpi=150)
        plt.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Level 1 EDA for DocFusion JSONL data")
    parser.add_argument(
        "--train-jsonl",
        required=True,
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write EDA artifacts",
    )
    args = parser.parse_args()

    train_path = Path(args.train_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(train_path)

    vendor_counts: Counter[str] = Counter()
    monthly_counts: Counter[str] = Counter()
    monthly_total_sum: dict[str, float] = {}
    totals: list[float] = []
    forgery_labels: list[int] = []
    outlier_rows: list[list[Any]] = []

    normalized_rows: list[dict[str, Any]] = []
    for record in records:
        fields = record.get("fields") if isinstance(record.get("fields"), dict) else {}
        label = record.get("label") if isinstance(record.get("label"), dict) else {}

        vendor = fields.get("vendor")
        vendor_name = str(vendor).strip() if vendor else "UNKNOWN"
        vendor_counts[vendor_name] += 1

        date_obj = parse_date(fields.get("date"))
        if date_obj:
            month_key = date_obj.strftime("%Y-%m")
            monthly_counts[month_key] += 1

        total = parse_total(fields.get("total"))
        if total is not None:
            totals.append(total)
            if date_obj:
                month_key = date_obj.strftime("%Y-%m")
                monthly_total_sum[month_key] = monthly_total_sum.get(month_key, 0.0) + total

        is_forged = int(bool(label.get("is_forged", 0)))
        forgery_labels.append(is_forged)

        normalized_rows.append(
            {
                "id": record.get("id"),
                "vendor": vendor_name,
                "date": date_obj.strftime("%Y-%m-%d") if date_obj else None,
                "total": total,
                "is_forged": is_forged,
            }
        )

    mean_total = safe_mean(totals)
    std_total = safe_stdev(totals)

    for row in normalized_rows:
        amount = row["total"]
        if amount is None:
            continue
        z = abs((amount - mean_total) / std_total)
        if z >= 2.0:
            outlier_rows.append(
                [
                    row["id"],
                    row["vendor"],
                    row["date"],
                    f"{amount:.2f}",
                    f"{z:.2f}",
                    row["is_forged"],
                ]
            )

    forgery_rate = safe_mean(forgery_labels)
    plot_created = save_optional_plots(out_dir, vendor_counts, totals)

    write_csv(
        out_dir / "vendor_frequency.csv",
        ["vendor", "count"],
        [[vendor, count] for vendor, count in vendor_counts.most_common()],
    )
    write_csv(
        out_dir / "monthly_totals.csv",
        ["month", "document_count", "total_sum", "avg_total"],
        [
            [
                month,
                monthly_counts.get(month, 0),
                f"{monthly_total_sum.get(month, 0.0):.2f}",
                (
                    f"{monthly_total_sum.get(month, 0.0) / monthly_counts[month]:.2f}"
                    if monthly_counts.get(month, 0) > 0
                    else "0.00"
                ),
            ]
            for month in sorted(monthly_counts)
        ],
    )
    write_csv(
        out_dir / "outlier_candidates.csv",
        ["id", "vendor", "date", "total", "z_score", "is_forged"],
        outlier_rows,
    )

    summary = [
        "# Level 1 EDA Summary",
        "",
        f"- Records analyzed: {len(records)}",
        f"- Vendor cardinality: {len(vendor_counts)}",
        f"- Average total: {mean_total:.2f}",
        f"- Std total: {std_total:.2f}",
        f"- Forgery ratio: {forgery_rate:.2%}",
        f"- Outlier candidates (|z| >= 2.0): {len(outlier_rows)}",
        "",
        "## Top Vendors",
    ]
    for vendor, count in vendor_counts.most_common(10):
        summary.append(f"- {vendor}: {count}")

    summary.extend(
        [
            "",
            "## Generated Artifacts",
            f"- {out_dir / 'vendor_frequency.csv'}",
            f"- {out_dir / 'monthly_totals.csv'}",
            f"- {out_dir / 'outlier_candidates.csv'}",
            f"- {out_dir / 'vendor_frequency.png'} (optional)",
            f"- {out_dir / 'total_distribution.png'} (optional)",
            "",
            f"- Plot images created: {'yes' if plot_created else 'no (matplotlib not installed)'}",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[eda] wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
