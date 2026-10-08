# Reset Backup Tree

```text
backup/project_reset_before_5year/
|-- config/                 # Previous dataset pipeline configurations
|-- data/
|   |-- features/           # Previous feature contract
|   |-- models/             # All previous active model artifacts
|   |-- processed/          # Previous processed datasets
|   `-- raw/                # Previous short Futures test dataset
|-- docs/                   # Superseded documentation
|-- reports/                # Superseded reports
|-- src/                    # Previous pipelines and engine implementations
|-- tests/                  # Tests for archived code
|-- __pycache__/            # Root caches
|-- main_build_dataset.py
|-- main_build_dual_datasets.py
|-- main_build_production_dataset.py
`-- main_train.py
```

This reset archive is additive to earlier backups. No earlier backup was moved, rewritten, or removed.
