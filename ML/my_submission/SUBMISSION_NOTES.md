# Submission Notes

Prepared on **March 17, 2026**.

## Completed

- Level 1 EDA notebooks and reports
- Level 2 extraction pipeline for `vendor`, `date`, `total`
- Level 3A anomaly pipeline with `is_forged`
- Level 3B Streamlit UI
- Level 4 local harness integration via `DocFusionSolution`
- Local `check_submission.py` validation passed

## Current State

- Extraction is strongest on `vendor` and `date`
- `total` extraction still has errors on some noisy CORD layouts
- Anomaly detection is implemented and usable, but not fully optimized
- Field highlighting is present in the UI, but exact forged-region localization is limited

## Unfinished / Partial Components

- Final anomaly benchmarking was not fully rerun after the latest extraction-only heuristic changes
- Forged-region localization is not high-precision
- Extraction quality, especially `total`, can still be improved further

## Relevant Reports

- Extraction snapshot:
  - `reports/benchmark/extraction_metrics_line_rank_20260317.json`
- Earlier anomaly benchmark snapshot:
  - `reports/benchmark/final_tuned_metrics_20260317.json`
  - `reports/benchmark/final_tuned_metrics_20260317.md`

## Packaging Note

Raw datasets are not bundled in the submission package because they are large and externally hosted. The repository includes scripts and instructions to rebuild the dataset views locally.
