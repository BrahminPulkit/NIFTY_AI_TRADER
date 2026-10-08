# Remaining Active Modules

| Module | Responsibility |
|---|---|
| `src.config` | Safe repository-relative paths for the five-year raw file, Security Master and reset backup |
| `src.dhan_data_downloader` | Security discovery, chunk planning, resumable Dhan REST acquisition, retries and chunk validation |
| `src.continuous_futures_builder` | Continuous-contract construction and raw-series quality validation |
| `src.risk_manager` | Existing reusable capital, position and session risk controls |
| `main_download_dhan_data.py` | Dhan downloader CLI |

No active module performs feature engineering, label generation, model training, prediction, backtesting, execution, or experimentation.
