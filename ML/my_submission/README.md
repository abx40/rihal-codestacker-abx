# DocFusion Baseline Submission

This folder contains a practical baseline for the CodeStacker 2026 ML challenge.

## Python Version

The challenge requires **Python 3.13+**.

Recommended setup:

```bash
cd /Users/abx/rihal/rihal-codestacker/ML/my_submission
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# optional notebook/plots tooling
python -m pip install -r requirements-dev.txt
```

## What it does

- Implements the required `DocFusionSolution` class in `solution.py`.
- Uses `fields` from JSONL directly when available.
- Falls back to multi-pass OCR with `tesseract` CLI when fields are missing.
- Predicts `is_forged` using lightweight train-derived heuristics.

## Level 2 Extraction Notes

- OCR uses multiple page segmentation modes (`psm 6/11/4`) and merges results.
- Date parsing supports numeric and month-name formats.
- Total extraction scores line candidates using keyword/exclusion hints.
- Vendor extraction uses known-vendor matching with fuzzy fallback.

## Local validation

From repository root:

```bash
cd ML
python3.13 check_submission.py --submission ./my_submission
```

## Notes

- This is a starter baseline, not a tuned final model.
- For better private score, upgrade OCR and anomaly modeling.
- OCR fallback requires the `tesseract` system binary to be installed and available in `PATH`.

## Level 1 EDA

Generate EDA artifacts from training JSONL:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/Users/abx/rihal/rihal-codestacker/ML/my_submission/.cache/matplotlib \
XDG_CACHE_HOME=/Users/abx/rihal/rihal-codestacker/ML/my_submission/.cache \
python3.13 scripts/level1_eda.py \
  --train-jsonl /Users/abx/rihal/rihal-codestacker/ML/dummy_data/train/train.jsonl \
  --out-dir /Users/abx/rihal/rihal-codestacker/ML/my_submission/reports/eda
```

Notebook:

- `/Users/abx/rihal/rihal-codestacker/ML/my_submission/notebooks/level1_eda.ipynb`
