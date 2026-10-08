# Final Production Tree

```text
NIFTY_AI_TRADER/
|-- backup/
|   `-- project_reset_before_5year/   # Complete reset archive
|-- config/
|   |-- dhan_historical_data.json
|   `-- project_data.json
|-- data/
|   `-- raw/
|       |-- dhan_security_master.csv
|       `-- nifty_50_5years_1min.csv
|-- docs/
|   |-- backup_tree.md
|   |-- files_moved_to_reset_backup.md
|   |-- final_production_tree.md
|   |-- remaining_active_modules.md
|   |-- repository_reset_report.md
|   `-- reset_validation_report.md
|-- notebooks/
|   `-- a.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- continuous_futures_builder.py
|   |-- dhan_data_downloader.py
|   `-- risk_manager.py
|-- tests/
|   |-- test_dhan_data_pipeline.py
|   `-- test_risk_manager.py
|-- main_download_dhan_data.py
|-- pytest.ini
`-- requirements.txt
```

Empty conventional directories may remain on disk, but contain no active model, processed dataset, report, or experimental code.
