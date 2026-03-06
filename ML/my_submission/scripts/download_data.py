#!/usr/bin/env python3
"""
Download helper for DocFusion challenge datasets.

This script attempts:
- CORD (HuggingFace) via `datasets` package.
- SROIE (Kaggle) via `kaggle` CLI if credentials are configured.
- Find-It-Again is currently manual download (official website requires manual flow).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, text


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _download_cord(raw_root: Path, max_samples: int | None) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        print("[download] CORD skipped: install `datasets` first.")
        return

    cord_root = raw_root / "cord"
    image_root = cord_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]
    for split in splits:
        stream_mode = True
        try:
            ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
        except Exception as exc:
            stream_mode = False
            try:
                ds = load_dataset("naver-clova-ix/cord-v2", split=split)
            except Exception as inner_exc:
                print(f"[download] CORD split '{split}' unavailable: {inner_exc}")
                continue

        rows: list[dict[str, Any]] = []
        written = 0
        for idx, sample in enumerate(ds):
            if max_samples is not None and written >= max_samples:
                break

            sample_id = str(sample.get("id", f"{split}_{idx:06d}"))
            image_name = f"{split}_{sample_id}.png"
            image_path = image_root / image_name

            image_obj = sample.get("image")
            if image_obj is None:
                continue
            try:
                image_obj.save(image_path)
            except Exception:
                continue

            row: dict[str, Any] = {
                "id": sample_id,
                "split": split,
                "image_path": f"images/{image_name}",
            }
            for key, value in sample.items():
                if key == "image":
                    continue
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    row[key] = value
                else:
                    row[key] = str(value)

            rows.append(row)
            written += 1

        if rows:
            out_jsonl = cord_root / f"cord_{split}.jsonl"
            _save_jsonl(out_jsonl, rows)
            mode = "streaming" if stream_mode else "materialized"
            print(f"[download] CORD {split} ({mode}): {len(rows)} samples -> {out_jsonl}")


def _download_sroie(raw_root: Path) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        print("[download] SROIE skipped: `kaggle` CLI not found.")
        return

    sroie_root = raw_root / "sroie"
    sroie_root.mkdir(parents=True, exist_ok=True)
    code, output = _run(
        [
            kaggle,
            "datasets",
            "download",
            "-d",
            "urbikn/sroie-datasetv2",
            "-p",
            str(sroie_root),
            "--unzip",
        ]
    )
    if code != 0:
        print("[download] SROIE download failed. Usually this means Kaggle credentials are missing.")
        print(output.strip())
        return
    print(f"[download] SROIE downloaded into {sroie_root}")


def _init_find_it_again_placeholder(raw_root: Path) -> None:
    fit_root = raw_root / "find_it_again"
    fit_root.mkdir(parents=True, exist_ok=True)
    note = fit_root / "README_MANUAL_DOWNLOAD.txt"
    if note.exists():
        return
    note.write_text(
        "\n".join(
            [
                "Find-It-Again dataset requires manual download from:",
                "https://l3i-share.univ-lr.fr/2023Finditagain/index.html",
                "",
                "Place files under this folder and include a labels file such as:",
                "- labels.jsonl OR labels.csv",
                "Expected columns/keys: id,image_path,is_forged,vendor,date,total",
                "",
                "Then run scripts/build_unified_dataset.py to normalize all sources.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[download] Created manual download note: {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DocFusion source datasets")
    parser.add_argument(
        "--raw-root",
        default="./data/raw",
        help="Directory to store raw datasets",
    )
    parser.add_argument(
        "--skip-cord",
        action="store_true",
        help="Skip CORD download from HuggingFace",
    )
    parser.add_argument(
        "--skip-sroie",
        action="store_true",
        help="Skip SROIE download from Kaggle",
    )
    parser.add_argument(
        "--cord-max-samples",
        type=int,
        default=None,
        help="Optional limit per CORD split for quick experiments",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_cord:
        _download_cord(raw_root, args.cord_max_samples)
    if not args.skip_sroie:
        _download_sroie(raw_root)
    _init_find_it_again_placeholder(raw_root)

    print(f"[download] done. Raw datasets root: {raw_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
