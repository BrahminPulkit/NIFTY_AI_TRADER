# Repository Reset Report

Date: 2026-07-22

## Outcome

The active repository was reduced from 166 files to a minimal five-year foundation. A total of 151 previous-development files were moved to `backup/project_reset_before_5year/` with their original repository-relative hierarchy preserved.

Nothing was permanently deleted. No raw dataset was edited. No features, labels, processed datasets, models, backtests, or experiments were generated.

## Retention policy

Retained:

- Five-year NIFTY Index raw CSV.
- Dhan Security Master.
- Dhan historical downloader and continuous-futures validator.
- Minimal project path configuration.
- Risk utility.
- Downloader and risk tests.
- `notebooks/a.ipynb`, unchanged and in its original location.

Archived:

- Every active model and model artifact.
- All processed and feature datasets.
- Previous training, feature, label, live, backtest and experimental pipelines.
- Old reports and documentation.
- Duplicate/V1/V2 implementations.
- Previous production-dataset entry points and tests.
- Caches and temporary artifacts.

The project phase is now `fresh_five_year_development`; dataset status is `raw_validation_pending`.
