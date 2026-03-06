# DocFusion ML Challenge Solution (Python 3.13+)

Production-oriented baseline for the **Rihal CodeStacker 2026 ML challenge**.

This project now includes:

- End-to-end `DocFusionSolution` (`train`/`predict`) for autograder harness
- Unified data pipeline scripts for SROIE + CORD + Find-It-Again
- OCR-based field extraction (`vendor`, `date`, `total`)
- Hybrid anomaly detection (`is_forged`) using rules + optional sklearn model
- Streamlit web UI with receipt upload and field highlight boxes
- Benchmark script with latency/memory/model-size report output
- EDA + extraction + anomaly notebooks

## 1) Project Layout

```text
my_submission/
  solution.py
  app.py
  benchmarks/
    benchmark_pipeline.py
  data/
    raw/
    processed/
  notebooks/
    level1_eda.ipynb
    level2_extraction_experiments.ipynb
    level3_anomaly_modeling.ipynb
  reports/
    eda/
    benchmark/
  scripts/
    download_data.py
    build_unified_dataset.py
    level1_eda.py
  requirements.txt
  requirements-dev.txt
  pyproject.toml
```

## 2) Python Environment (Required)

Challenge requires **Python 3.13+**.

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

System dependency for OCR:

- `tesseract` binary must be installed and in `PATH`.

## 3) Data Pipeline

### 3.1 Download Public Sources

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
.venv/bin/python scripts/download_data.py --raw-root ./data/raw --cord-max-samples 300
```

Notes:

- **CORD** is downloaded automatically from HuggingFace (streaming mode supported).
- **SROIE** auto-download requires Kaggle CLI + Kaggle credentials.
- **Find-It-Again** is manual download; script creates guidance file:
  `data/raw/find_it_again/README_MANUAL_DOWNLOAD.txt`

### 3.2 Build Unified Training Schema

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
.venv/bin/python scripts/build_unified_dataset.py \
  --raw-root ./data/raw \
  --out-root ./data/processed/unified \
  --dummy-root /Users/abx/rihal/rihal-codestacker/ML/dummy_data
```

Output:

- `data/processed/unified/train/train.jsonl`
- `data/processed/unified/train/images/*`
- `data/processed/unified/stats.json`

## 4) Level 1 EDA

Generate EDA artifacts:

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
MPLBACKEND=Agg \
MPLCONFIGDIR=/Users/abx/rihal/rihal-codestacker/ML/my_submission/.cache/matplotlib \
XDG_CACHE_HOME=/Users/abx/rihal/rihal-codestacker/ML/my_submission/.cache \
.venv/bin/python scripts/level1_eda.py \
  --train-jsonl /Users/abx/rihal/rihal-codestacker/ML/my_submission/data/processed/unified/train/train.jsonl \
  --out-dir /Users/abx/rihal/rihal-codestacker/ML/my_submission/reports/eda
```

Notebook:

- `notebooks/level1_eda.ipynb`

## 5) Level 2 + Level 4 Harness Integration

### 5.1 Local Contract Validation

```bash
cd /Users/abx/rihal/rihal-codestacker/ML
python3.13 check_submission.py --submission ./my_submission --data ./dummy_data
```

`solution.py` supports:

- OCR multi-pass extraction (`psm 6/11/4`)
- date normalization (numeric + month text)
- total scoring with keyword/exclusion heuristics
- vendor fuzzy matching
- hybrid anomaly model (rules + sklearn random forest when available)

## 6) Level 3 Web UI

Run dashboard:

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
.venv/bin/streamlit run app.py
```

Features:

- Upload receipt image
- Extracted fields display (`vendor/date/total`)
- Anomaly status (`is_forged`) with confidence
- Bounding-box highlights for detected fields
- In-app model train/refresh trigger
- Optional LLM-generated anomaly narrative (1-2 sentence explanation)

### 6.1 Optional LLM Narrative (Bonus)

Enable in the UI sidebar with **Enable LLM anomaly summary**.

Environment variables:

- `OPENAI_API_KEY` (or `DOCFUSION_LLM_API_KEY`) - required for API access
- `DOCFUSION_LLM_MODEL` - optional model override (default: `gpt-4o-mini`)
- `DOCFUSION_LLM_BASE_URL` - optional OpenAI-compatible endpoint (default: `https://api.openai.com/v1`)
- `DOCFUSION_LLM_TIMEOUT` - optional request timeout seconds (default: `12`)

When LLM is disabled or unavailable, the app falls back to a deterministic heuristic summary.

## 7) Benchmarking (Latency / Memory / Model Size)

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
.venv/bin/python benchmarks/benchmark_pipeline.py \
  --submission-dir . \
  --data-dir /Users/abx/rihal/rihal-codestacker/ML/dummy_data \
  --work-dir ./tmp_benchmark \
  --report-dir ./reports/benchmark
```

Generated:

- `reports/benchmark/benchmark.json`
- `reports/benchmark/benchmark.md`

Current dummy benchmark snapshot:

- Train: ~1.28s
- Predict: ~0.48s
- Peak memory: ~1.76 MB (train), ~1.03 MB (predict)
- Model size: ~0.23 MB

## 8) Additional Notebooks

- `notebooks/level2_extraction_experiments.ipynb`
- `notebooks/level3_anomaly_modeling.ipynb`

## 9) Submission Packaging Notes

For final standalone challenge submission repo:

1. Put `solution.py` at repo root.
2. Include this README (or adapted version) with full run instructions.
3. Include notebooks, UI code (`app.py`), benchmark report, dependencies.
4. Confirm `python3.13 check_submission.py --submission .` passes in your final repo layout.

## 10) Remaining External Blockers

These are external-data constraints, not code gaps:

- SROIE full download needs Kaggle credentials.
- Find-It-Again labels/images need manual download from official link.
- Full CORD materialization can require significant disk; streaming mode is implemented for low-disk environments.
