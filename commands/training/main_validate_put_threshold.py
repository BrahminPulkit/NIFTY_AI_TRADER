"""Select a PUT threshold on early OOF folds and evaluate the latest holdout."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def main() -> None:
    root = Path("models/stage2_put_candidate_v1")
    oof = pd.read_parquet(root / "oof_predictions.parquet").reset_index(drop=True)
    oof["fold"] = np.repeat(range(1, 5), len(oof) // 4)
    candidates = [.70, .75, .80, .85, .90]
    diagnostics = []
    for threshold in candidates:
        for fold, frame in oof.groupby("fold"):
            selected = frame.probability.ge(threshold)
            diagnostics.append({
                "threshold": threshold, "fold": int(fold), "signals": int(selected.sum()),
                "precision": float(precision_score(frame.target, selected, zero_division=0)),
                "recall": float(recall_score(frame.target, selected, zero_division=0)),
            })
    table = pd.DataFrame(diagnostics)
    development = table.loc[table.fold.le(3)]
    eligible = []
    for threshold, frame in development.groupby("threshold"):
        if frame.signals.ge(30).all() and frame.precision.ge(.75).all():
            eligible.append(float(threshold))
    selected_threshold = max(eligible) if eligible else None
    holdout = table.loc[(table.fold.eq(4)) & table.threshold.eq(selected_threshold)].iloc[0]
    passed = bool(holdout.signals >= 20 and holdout.precision >= .75)
    result = {
        "status": "FORWARD_PAPER_CANDIDATE" if passed else "THRESHOLD_VALIDATION_FAILED",
        "live_orders_enabled": False, "selected_threshold": selected_threshold,
        "selection_data": "OOF_FOLDS_1_TO_3", "holdout_data": "LATEST_OOF_FOLD_4",
        "development_rule": "highest threshold with >=30 signals and >=0.75 precision in every development fold",
        "holdout_rule": ">=20 signals and >=0.75 precision",
        "holdout": holdout.to_dict(),
        "warning": "Paper-only forward monitoring is required; this is not broker-order approval.",
    }
    table.to_csv("reports/stage2_put_model/threshold_by_fold.csv", index=False)
    (root / "threshold_validation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
