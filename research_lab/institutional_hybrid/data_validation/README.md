# Institutional Data Validation Engine

Strategy-free, deterministic audit of three pinned historical CSV sources. It
reads source files without mutation, exports only below `outputs/`, permits exact
or backward-only timestamp alignment, and writes the gate consumed by every
research runner.

Run:

```powershell
python data_validation/run_audit.py
```

Exit code `0` means PASS; exit code `2` means FAIL. A FAIL gate prohibits all
Phase-1 experiments, comparison-center runs and hybrid runs.
