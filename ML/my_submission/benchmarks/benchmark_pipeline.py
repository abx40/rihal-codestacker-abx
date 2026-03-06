#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any


def load_solution(submission_dir: Path):
    solution_path = submission_dir / "solution.py"
    spec = importlib.util.spec_from_file_location("_docfusion_submission", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {solution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_docfusion_submission"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "DocFusionSolution"):
        raise RuntimeError("DocFusionSolution class not found")
    return module.DocFusionSolution()


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def write_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "benchmark.json"
    md_path = report_dir / "benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Benchmark Report",
        "",
        f"- Train time (s): {payload['train_seconds']:.4f}",
        f"- Predict time (s): {payload['predict_seconds']:.4f}",
        f"- Peak train memory (MB): {payload['train_peak_mb']:.4f}",
        f"- Peak predict memory (MB): {payload['predict_peak_mb']:.4f}",
        f"- Model size (MB): {payload['model_size_mb']:.4f}",
        f"- Predictions written: {payload['predictions_count']}",
        "",
        "## Paths",
        f"- model_dir: {payload['model_dir']}",
        f"- predictions_path: {payload['predictions_path']}",
        f"- json_report: {json_path}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DocFusion train/predict pipeline")
    parser.add_argument("--submission-dir", default=".", help="Path containing solution.py")
    parser.add_argument("--data-dir", default="../dummy_data", help="Path containing train/ and test/")
    parser.add_argument("--work-dir", default="./tmp_benchmark", help="Scratch directory")
    parser.add_argument("--report-dir", default="./reports/benchmark", help="Report output directory")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    report_dir = Path(args.report_dir).resolve()

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    predictions_path = work_dir / "predictions.jsonl"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    solution = load_solution(submission_dir)

    tracemalloc.start()
    train_start = time.perf_counter()
    model_dir = solution.train(str(train_dir), str(work_dir))
    train_seconds = time.perf_counter() - train_start
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    predict_start = time.perf_counter()
    solution.predict(model_dir, str(test_dir), str(predictions_path))
    predict_seconds = time.perf_counter() - predict_start
    _, predict_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred_count = 0
    if predictions_path.exists():
        pred_count = sum(1 for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip())

    payload = {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "train_peak_mb": train_peak / (1024 * 1024),
        "predict_peak_mb": predict_peak / (1024 * 1024),
        "model_size_mb": dir_size_bytes(Path(model_dir)) / (1024 * 1024),
        "predictions_count": pred_count,
        "model_dir": model_dir,
        "predictions_path": str(predictions_path),
    }
    write_reports(report_dir, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
