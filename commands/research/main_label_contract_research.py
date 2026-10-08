from pathlib import Path
import json
import pandas as pd

from src.label_contract_research import OpportunityMatrix, contract_grid, summarize_contract

INPUT = Path("data/setups/stage1_setup_dataset.parquet")
REPORTS = Path("reports/label_contract_research")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(INPUT)
    matrix = OpportunityMatrix(frame, 20)
    overall_parts=[]; direction_parts=[]; year_parts=[]; regime_parts=[]; excursion_parts=[]
    for spec in contract_grid():
        result = matrix.evaluate(spec)
        overall_parts.append(summarize_contract(result, ["contract_id", "family", "horizon"]))
        direction_parts.append(summarize_contract(result, ["contract_id", "direction"]))
        year_parts.append(summarize_contract(result, ["contract_id", "direction", "year"]))
        regime_parts.append(summarize_contract(result, ["contract_id", "direction", "regime"]))
        excursion_parts.append(summarize_contract(result, ["contract_id", "direction"])[
            ["contract_id","direction","observations","average_mfe_r","average_mae_r"]])
    overall=pd.concat(overall_parts,ignore_index=True); directional=pd.concat(direction_parts,ignore_index=True)
    yearly=pd.concat(year_parts,ignore_index=True); regimes=pd.concat(regime_parts,ignore_index=True)
    excursions=pd.concat(excursion_parts,ignore_index=True)
    direction_pivot=directional.pivot(index="contract_id",columns="direction",values="expected_r")
    stability=yearly.groupby("contract_id").expected_r.agg(
        yearly_mean="mean",yearly_std="std",worst_year_direction="min",best_year_direction="max")
    regime_stability=regimes.groupby("contract_id").expected_r.agg(regime_std="std",worst_regime="min")
    leaderboard=overall.merge(direction_pivot,on="contract_id").merge(stability,on="contract_id").merge(
        regime_stability,on="contract_id")
    leaderboard["positive_both_directions"]=(leaderboard.BUY_CALL>0)&(leaderboard.BUY_PUT>0)
    leaderboard["robust_positive"]=(leaderboard.positive_both_directions)&(leaderboard.worst_year_direction>0)&(
        leaderboard.worst_regime>0)
    leaderboard=leaderboard.sort_values(["robust_positive","positive_both_directions","expected_r","yearly_std"],
                                        ascending=[False,False,False,True])
    leaderboard.to_csv(REPORTS/"contract_leaderboard.csv",index=False)
    directional.to_csv(REPORTS/"directional_contract_results.csv",index=False)
    yearly.to_csv(REPORTS/"yearly_contract_stability.csv",index=False)
    regimes.to_csv(REPORTS/"regime_contract_stability.csv",index=False)
    excursions.to_csv(REPORTS/"mfe_mae_contract_statistics.csv",index=False)
    for family,name in [("FIXED","fixed_percentage_results.csv"),("ATR","atr_contract_results.csv"),
                        ("ATR_TRAILING","dynamic_exit_results.csv")]:
        leaderboard[leaderboard.family.eq(family)].to_csv(REPORTS/name,index=False)
    leaderboard[leaderboard.family.str.startswith("TIME")].to_csv(REPORTS/"time_exit_results.csv",index=False)
    candidates=leaderboard[leaderboard.positive_both_directions].head(20)
    payload={"status":"RESEARCH_ONLY","production_contract_changed":False,
             "contracts_evaluated":len(leaderboard),"positive_both_directions":int(leaderboard.positive_both_directions.sum()),
             "robust_positive_contracts":int(leaderboard.robust_positive.sum()),
             "candidate_contract_ids":candidates.contract_id.tolist()}
    (REPORTS/"candidate_contracts.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    top=leaderboard.iloc[0]
    verdict=("At least one contract is positive in both directions and all evaluated year/regime slices."
             if leaderboard.robust_positive.any() else
             "No contract passes the strict all-year/all-regime robustness gate; do not promote a contract yet.")
    (REPORTS/"recommendation.md").write_text(f"""# Step 17.5 Recommendation

**{verdict}**

- Contracts evaluated: **{len(leaderboard)}**
- Positive Expected R in both directions: **{leaderboard.positive_both_directions.sum()}**
- Strict robust-positive contracts: **{leaderboard.robust_positive.sum()}**
- Highest-ranked research contract: `{top.contract_id}`
- Combined Expected R: **{top.expected_r:.4f}R**
- BUY CALL / BUY PUT: **{top.BUY_CALL:.4f}R / {top.BUY_PUT:.4f}R**

This is contract research over overlapping Stage-1 opportunities. It is not a backtest and does not change the production label contract.
""",encoding="utf-8")
    (REPORTS/"label_contract_research_report.md").write_text(f"""# Production Label Contract Research

Stage-1 V2 remains frozen. The study evaluated fixed-percentage barriers, ATR-scaled barriers, causal ATR trailing exits, and pure time exits at 5/10/15/20 candles. Barrier ambiguity is stop-first, fills are gap-aware, and paths never cross sessions.

`win_rate` means positive realized R and `loss_rate` means negative realized R across every family. `stop_rate` identifies the exit mechanism, so a profitable trailing-stop exit is both a win and a trailing stop; those fields are intentionally not mutually exclusive for dynamic exits.

{verdict}

See `contract_leaderboard.csv` for the complete ranking and `recommendation.md` for the research verdict. No model, calibration, production-label change, or execution backtest was performed.
""",encoding="utf-8")
    print(f"Step 17.5 complete: {len(leaderboard)} contracts; both-direction positive: {leaderboard.positive_both_directions.sum()}")

if __name__=="__main__": main()
