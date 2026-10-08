# Step 23 — Model Validation Report

## Freeze result

- Version: `production_candidate_v1.0.0`
- Dataset: `prediction_dataset_v1`
- Rows: 7,643
- Features: 83
- Training period: 2021-07-26T10:36:00+05:30 to 2026-07-22T13:11:00+05:30
- Target: ATM CALL +5% within 15 minutes
- Approved threshold: 0.90

## Approved research performance

- ROC-AUC: 0.853913
- PR-AUC: 0.809656
- Precision: 0.729438
- Recall: 0.747236
- F1: 0.738230
- Balanced accuracy: 0.769433

## Validation

- Frozen input hashes unchanged: PASS
- Feature names and order frozen: PASS
- Missing/unknown/reordered features rejected: PASS
- Forbidden future/outcome features absent: PASS
- Preprocessing fitted once on the approved full production dataset: PASS
- Repeated probability inference bitwise identical: PASS
- Native CatBoost model persisted: PASS

## Readiness boundary

This is the first **production candidate**, but Step 20L classified V1 as
**Research Ready** and its Decision Engine evidence gate failed the minimum
trade requirement. Live deployment and order execution remain unapproved.
