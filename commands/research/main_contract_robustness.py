from pathlib import Path
import json
import pandas as pd

from src.contract_robustness import apply_scenario, execution_scenarios, metrics
from src.label_contract_research import ContractSpec, OpportunityMatrix

INPUT=Path("data/setups/stage1_setup_dataset.parquet")
REPORTS=Path("reports/contract_robustness")
CANDIDATE=ContractSpec("TRAIL_H15_ATR0.50","ATR_TRAILING",15,.5,None,"TRAILING")


def main():
    REPORTS.mkdir(parents=True,exist_ok=True)
    source=pd.read_parquet(INPUT)
    gross=OpportunityMatrix(source,15).evaluate(CANDIDATE)
    scenarios=execution_scenarios()
    overall=[];direction=[];year=[];regime=[]
    baseline_er=gross.realized_r.mean()
    for _,scenario in scenarios.iterrows():
        net=apply_scenario(gross,scenario)
        overall.append(metrics(net,["scenario_id"]))
        direction.append(metrics(net,["scenario_id","direction"]))
        year.append(metrics(net,["scenario_id","direction","year"]))
        regime.append(metrics(net,["scenario_id","direction","regime"]))
    overall=pd.concat(overall,ignore_index=True).merge(scenarios,on="scenario_id")
    direction=pd.concat(direction,ignore_index=True);year=pd.concat(year,ignore_index=True)
    regime=pd.concat(regime,ignore_index=True)
    piv=direction.pivot(index="scenario_id",columns="direction",values="expected_r")
    stability=year.groupby("scenario_id").expected_r.agg(worst_year_direction="min",year_std="std")
    regime_stability=regime.groupby("scenario_id").expected_r.agg(worst_regime_direction="min",regime_std="std")
    summary=overall.merge(piv,on="scenario_id").merge(stability,on="scenario_id").merge(
        regime_stability,on="scenario_id")
    summary["expected_r_change_from_gross"]=summary.expected_r-baseline_er
    summary["positive_combined"]=summary.expected_r>0
    summary["positive_both_directions"]=(summary.BUY_CALL>0)&(summary.BUY_PUT>0)
    summary["stable_all_year_regime"]=(summary.worst_year_direction>0)&(summary.worst_regime_direction>0)
    summary.to_csv(REPORTS/"execution_stress_scenarios.csv",index=False)
    direction.to_csv(REPORTS/"directional_robustness.csv",index=False)
    year.to_csv(REPORTS/"yearly_robustness.csv",index=False)
    regime.to_csv(REPORTS/"regime_robustness.csv",index=False)

    # Isolate requested dimensions at zero values for all other assumptions.
    summary[(summary.spread_ticks_roundtrip==0)&(summary.tracking_points_per_side==0)&(
        summary.ambiguity_penalty_ticks==0)].to_csv(REPORTS/"slippage_analysis.csv",index=False)
    summary[(summary.slippage_ticks_per_side==0)&(summary.tracking_points_per_side==0)&(
        summary.ambiguity_penalty_ticks==0)].to_csv(REPORTS/"spread_analysis.csv",index=False)
    summary[(summary.slippage_ticks_per_side==0)&(summary.spread_ticks_roundtrip==0)&(
        summary.ambiguity_penalty_ticks==0)].to_csv(REPORTS/"index_futures_difference_analysis.csv",index=False)
    summary[(summary.slippage_ticks_per_side==0)&(summary.spread_ticks_roundtrip==0)&(
        summary.tracking_points_per_side==0)].to_csv(REPORTS/"ohlc_ambiguity_analysis.csv",index=False)

    # Gap-through stops are already filled at the first open beyond the prior-bar
    # trailing level. Quantify realized losses beyond -1R as a conservative proxy.
    gap=gross[gross.realized_r < -1].copy()
    gap[["direction","year","regime","realized_r","entry_price","exit_price","risk_points"]].to_csv(
        REPORTS/"gap_through_stop_analysis.csv",index=False)
    reference=summary[summary.is_realistic_reference].iloc[0]
    passed=bool(reference.expected_r>0 and reference.profit_factor>1 and reference.BUY_CALL>0 and
                reference.BUY_PUT>0 and reference.worst_year_direction>0 and reference.worst_regime_direction>0)
    status="production_candidate" if passed else "rejected_by_robustness"
    verdict={"candidate":CANDIDATE.contract_id,"status":status,"promoted":False,
             "realistic_scenario":reference.scenario_id,"expected_r":reference.expected_r,
             "profit_factor":reference.profit_factor,"buy_call_expected_r":reference.BUY_CALL,
             "buy_put_expected_r":reference.BUY_PUT,"worst_year_direction":reference.worst_year_direction,
             "worst_regime_direction":reference.worst_regime_direction,
             "gross_expected_r":baseline_er,"maximum_adverse_expected_r_change":summary.expected_r_change_from_gross.min(),
             "gap_through_observations":len(gap)}
    (REPORTS/"candidate_verdict.json").write_text(json.dumps(verdict,indent=2),encoding="utf-8")
    conclusion=("The candidate survives the defined realistic execution scenario and is marked "
                "`production_candidate`; this is not production promotion." if passed else
                "The candidate fails the defined realistic robustness gate and is rejected.")
    (REPORTS/"contract_robustness_report.md").write_text(f"""# Contract Robustness Validation

Candidate: `{CANDIDATE.contract_id}`. **{conclusion}**

## Execution definitions

- NIFTY Futures tick: 0.05 point
- Slippage: 0/1/2/3 ticks per side
- Spread: 0/1/2/4 ticks, full round-trip cost
- Index-to-Futures difference: 0/0.25/0.50/1.00 adverse point per side
- OHLC ambiguity stress: 0/1 additional adverse tick on trailing exit
- Trailing stop uses only the prior completed bar; stop is checked before any current-bar trail update
- Gap-through stop fills at the first available candle open

## Realistic reference

`{reference.scenario_id}`: Expected R **{reference.expected_r:.4f}**, Profit Factor **{reference.profit_factor:.3f}**, BUY CALL **{reference.BUY_CALL:.4f}R**, BUY PUT **{reference.BUY_PUT:.4f}R**, worst year-direction **{reference.worst_year_direction:.4f}R**, worst regime-direction **{reference.worst_regime_direction:.4f}R**.

Gross Expected R is **{baseline_er:.4f}R**. Worst stress-envelope change is **{summary.expected_r_change_from_gross.min():.4f}R**. No model, calibration, label promotion, or production change occurred.
""",encoding="utf-8")
    (REPORTS/"recommendation.md").write_text(f"# Recommendation\n\n**{conclusion}**\n",encoding="utf-8")
    print(f"Step 17.6 complete: {status}; realistic Expected R {reference.expected_r:.4f}")

if __name__=="__main__":main()

