"""Deterministic audit exports and data-quality visualisations."""
from __future__ import annotations

import html
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import LAB

OUTPUT=Path(__file__).resolve().parent/"outputs"


def _guard(path: Path) -> Path:
    resolved=path.resolve()
    if LAB not in resolved.parents:
        raise ValueError(f"Output outside research lab refused: {resolved}")
    return resolved


def _chart(path: Path,title: str,draw=None):
    fig,ax=plt.subplots(figsize=(12,5))
    if draw: draw(fig,ax)
    else:
        ax.text(.5,.5,"Unavailable: required timestamp-aligned fields are absent",
                ha="center",va="center",transform=ax.transAxes)
    ax.set_title(title); fig.tight_layout(); fig.savefig(_guard(path),dpi=180,bbox_inches="tight"); plt.close(fig)


def charts(result: dict,loaded: dict,tables: dict) -> None:
    chart_dir=OUTPUT/"charts"; chart_dir.mkdir(parents=True,exist_ok=True)
    index=loaded["index"].frame
    daily=tables["index"][0]
    _chart(chart_dir/"timeline_coverage.png","Timeline Coverage",
           lambda f,a:a.scatter(index.index,index["close"],s=.25) if isinstance(index.index,pd.DatetimeIndex) else None)
    def missing_heat(fig,ax):
        if daily.empty: return
        d=daily.copy(); d["month"]=pd.to_datetime(d.date).dt.to_period("M").astype(str)
        d["day"]=pd.to_datetime(d.date).dt.day
        pivot=d.pivot_table(index="month",columns="day",values="completeness_pct",aggfunc="mean").fillna(0)
        im=ax.imshow(pivot,aspect="auto",cmap="RdYlGn",vmin=0,vmax=100); fig.colorbar(im,ax=ax)
    _chart(chart_dir/"missing_candle_heatmap.png","Missing Candle Heatmap",missing_heat)
    _chart(chart_dir/"trading_day_completeness.png","Trading-day Completeness",
           lambda f,a:a.plot(pd.to_datetime(daily.date),daily.completeness_pct) if not daily.empty else None)
    aligned=tables.get("_alignment",pd.DataFrame())
    _chart(chart_dir/"timestamp_alignment.png","Timestamp Alignment",
           lambda f,a:a.plot(aligned.index,aligned.exact_timestamp_match.astype(int))
           if not aligned.empty else None)
    _chart(chart_dir/"spot_vs_index_overlay.png","Spot vs Index Overlay",
           lambda f,a:(a.plot(aligned.index,aligned.close_index,label="Index"),
                       a.plot(aligned.index,aligned.spot,label="Option spot",alpha=.7),a.legend())
           if not aligned.empty and {"close_index","spot"}.issubset(aligned) else None)
    def correlation_plot(fig,ax):
        if not aligned.empty and {"close_index","spot"}.issubset(aligned):
            sample=aligned[["close_index","spot"]].dropna().iloc[::max(len(aligned)//10000,1)]
            ax.scatter(sample.close_index,sample.spot,s=2,alpha=.3)
    _chart(chart_dir/"correlation_plot.png","Index Close vs Option Spot Correlation",correlation_plot)
    for name,column in (("volume_distribution","volume"),("oi_distribution","oi"),
                        ("iv_distribution","iv"),("strike_distribution","strike")):
        source=loaded["option_full"].frame
        def draw(fig,ax,c=column,s=source):
            if c in s:
                values=pd.to_numeric(s[c],errors="coerce").dropna()
                if len(values): ax.hist(values,bins=50)
                else: ax.text(.5,.5,f"No non-NaN {c} values",ha="center",va="center",transform=ax.transAxes)
        _chart(chart_dir/f"{name}.png",name.replace("_"," ").title(),draw)
    def dashboard(fig,ax):
        score=result["data_quality_score"]
        ax.barh(["Quality score"],[score],color=("green" if score>=90 else "orange" if score>=75 else "red"))
        ax.set_xlim(0,100); ax.axvline(90,color="black",linestyle="--")
        ax.text(score,0,f" {score:.2f} / 100 — {result['status']}",va="center")
    _chart(chart_dir/"data_quality_dashboard.png","Data Quality Dashboard",dashboard)


def export(result: dict,loaded: dict,tables: dict) -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    canonical=json.dumps(result,indent=2,sort_keys=True,default=str)
    (OUTPUT/"data_validation_summary.json").write_text(canonical,encoding="utf-8")
    (OUTPUT/"research_data_gate.json").write_text(json.dumps({
        "approved":result["can_research_continue"],"status":result["status"],
        "data_quality_score":result["data_quality_score"],
        "critical_issues":result["critical_issues"],
        "dataset_hashes":{k:v.digest for k,v in loaded.items()},
    },indent=2,sort_keys=True),encoding="utf-8")
    rows=[]
    for name,s in result["datasets"].items():
        row={k:v for k,v in s.items() if not isinstance(v,(list,dict))}
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT/"data_validation_summary.csv",index=False)
    failures=[]
    index=loaded["index"].frame
    if {"open","high","low","close"}.issubset(index):
        corrupt=(index.high<index[["open","close","low"]].max(axis=1))|(
            index.low>index[["open","close","high"]].min(axis=1))
        for ts,row in index.loc[corrupt].iterrows():
            failures.append({"dataset":"index","row_index":str(ts),"timestamp":str(ts),
                             "failure":"corrupted_ohlc","details":json.dumps(row.to_dict(),default=str)})
    option=loaded["option_full"].frame
    if "iv" in option:
        iv=pd.to_numeric(option.iv,errors="coerce")
        invalid=(iv<=0)|(iv>200)
        for ts,row in option.loc[invalid].iterrows():
            failures.append({"dataset":"option_full","row_index":str(ts),"timestamp":str(ts),
                             "failure":"iv_outside_(0,200]","details":json.dumps(row.to_dict(),default=str)})
    if {"strike","spot"}.issubset(option):
        distance=(pd.to_numeric(option.strike,errors="coerce")-
                  pd.to_numeric(option.spot,errors="coerce")).abs()
        for ts,row in option.loc[distance>100].iterrows():
            failures.append({"dataset":"option_full","row_index":str(ts),"timestamp":str(ts),
                             "failure":"strike_distance_over_100","details":json.dumps(row.to_dict(),default=str)})
    pd.DataFrame(failures,columns=["dataset","row_index","timestamp","failure","details"]).to_csv(
        OUTPUT/"row_level_failures.csv",index=False)
    index=loaded["index"].frame
    aligned=tables.get("_alignment")
    if aligned is not None:
        aligned.reset_index().to_csv(OUTPUT/"minute_alignment_report.csv",index=False)
    issues="\n".join(f"- {x}" for x in result["critical_issues"]) or "- None"
    dataset_lines="\n".join(
        f"- **{k}:** {v['total_candles']} candles, {v['total_trading_days']} days, "
        f"{v['duplicate_timestamps']} duplicates, {v['nan_required_cells']} required-field NaNs"
        for k,v in result["datasets"].items()
    )
    recommendation=("YES\n\nResearch datasets are validated and approved for institutional historical backtesting."
                    if result["can_research_continue"] else "NO\n\nCritical issues must be resolved before any backtest.")
    md=f"""# Institutional Data Validation Report

## Decision

**{result['status']} — Data Quality Score: {result['data_quality_score']}/100**

Can research continue? **{recommendation}**

## Critical issues

{issues}

## Dataset summary

{dataset_lines}

## Alignment

{json.dumps(result['alignment'],indent=2) if result['alignment'] else 'Not computable: option timestamps are absent.'}

## Spot versus index

Correlation: `{result['spot_index_correlation']}`  
Consistency: `{result['spot_index_consistency']}`

## Integrity statement

This audit is deterministic, read-only, contains no strategy execution and uses
no model-derived outcomes. Absent weekdays are reported as holiday-or-missing-day
candidates because no authoritative exchange calendar was supplied.
"""
    (OUTPUT/"Institutional_Data_Validation_Report.md").write_text(md,encoding="utf-8")
    (OUTPUT/"Institutional_Data_Validation_Report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Data Validation</title></head>"
        f"<body><pre>{html.escape(md)}</pre></body></html>",encoding="utf-8")
    charts(result,loaded,tables)
