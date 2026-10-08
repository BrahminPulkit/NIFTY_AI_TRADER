"""Publication-quality deterministic charts for every methodology."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .core import guard_output

plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(guard_output(path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def all_charts(trades: pd.DataFrame, rejected: pd.DataFrame, output: Path, title: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    r = trades.get("net_return", pd.Series(dtype=float))
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1 if not eq.empty else eq
    specs = [
        ("equity_curve", eq, "line"), ("drawdown_curve", dd, "line"),
        ("entry_distribution", pd.to_datetime(trades.get("entry_time", pd.Series(dtype=str))).dt.hour, "hist"),
        ("exit_distribution", trades.get("exit_reason", pd.Series(dtype=str)).value_counts(), "bar"),
        ("premium_expansion_histogram", trades.get("premium_expansion", pd.Series(dtype=float)), "hist"),
        ("win_loss_distribution", pd.Series(np.where(r > 0, "Win", "Loss")).value_counts(), "bar"),
        ("trade_duration_histogram", trades.get("hold_bars", pd.Series(dtype=float)), "hist"),
    ]
    paths = []
    for name, series, kind in specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        if kind == "line": ax.plot(series.reset_index(drop=True), linewidth=1.6)
        elif kind == "hist": ax.hist(pd.Series(series).dropna(), bins=25, edgecolor="white")
        else: pd.Series(series).plot.bar(ax=ax)
        ax.set_title(f"{title} — {name.replace('_', ' ').title()}")
        path = output / f"{name}.png"; _save(fig, path); paths.append(path)
    t = trades.copy()
    if not t.empty:
        t["entry_time"] = pd.to_datetime(t.entry_time)
        t["month"] = t.entry_time.dt.to_period("M").astype(str)
        t["week"] = t.entry_time.dt.to_period("W").astype(str)
    for name, key in (("monthly_performance", "month"), ("weekly_performance", "week")):
        fig, ax = plt.subplots(figsize=(12, 5))
        if not t.empty: t.groupby(key).net_return.sum().plot.bar(ax=ax)
        ax.set_title(f"{title} — {name.replace('_', ' ').title()}")
        path = output / f"{name}.png"; _save(fig, path); paths.append(path)
    fig, ax = plt.subplots(figsize=(12, 4))
    if not t.empty:
        heat = t.assign(hour=t.entry_time.dt.hour, dow=t.entry_time.dt.day_name()).pivot_table(
            index="dow", columns="hour", values="net_return", aggfunc="mean"
        )
        image = ax.imshow(heat.fillna(0), aspect="auto", cmap="RdYlGn")
        fig.colorbar(image, ax=ax)
    ax.set_title(f"{title} — Time-of-Day Heatmap")
    path = output / "time_of_day_heatmap.png"; _save(fig, path); paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not t.empty:
        wins = t.net_return.gt(0)
        runs = wins.ne(wins.shift()).cumsum()
        wins.groupby(runs).size().plot.bar(ax=ax)
    ax.set_title(f"{title} — Consecutive Wins/Losses")
    path = output / "consecutive_wins_losses.png"; _save(fig, path); paths.append(path)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    labels = ["Win Rate", "Expectancy", "Profit Factor", "Stability", "Premium"]
    values = [float((r > 0).mean()) if len(r) else 0, max(float(r.mean()), 0) if len(r) else 0,
              min(float(r[r > 0].sum() / -r[r < 0].sum()), 3) / 3 if (r < 0).any() else 0,
              1 / (1 + float(r.std())) if len(r) else 0,
              max(float(trades.get("premium_expansion", pd.Series(dtype=float)).mean()), 0)
              if not trades.empty else 0]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    ax.plot(angles, values); ax.fill(angles, values, alpha=.2); ax.set_xticks(angles[:-1], labels)
    ax.set_title(f"{title} — Strategy Score Radar")
    path = output / "strategy_score_radar.png"; _save(fig, path); paths.append(path)
    for name, column in (
        ("institutional_score_distribution", "strategy_score"),
        ("volume_analysis", "volume"), ("oi_analysis", "open_interest"),
        ("holding_time_analysis", "hold_bars"), ("premium_expansion_analysis", "premium_expansion"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        if column in trades: ax.hist(trades[column].dropna(), bins=25)
        ax.set_title(f"{title} — {name.replace('_', ' ').title()}")
        path = output / f"{name}.png"; _save(fig, path); paths.append(path)
    for name, time_col, price_col in (
        ("entry_chart", "entry_time", "entry_premium"), ("exit_chart", "exit_time", "exit_premium"),
    ):
        fig, ax = plt.subplots(figsize=(12, 5))
        if not trades.empty:
            ax.scatter(pd.to_datetime(trades[time_col]), trades[price_col],
                       c=np.where(trades.net_return > 0, "green", "red"), s=18)
        ax.set_title(f"{title} — {name.replace('_', ' ').title()}")
        path = output / f"{name}.png"; _save(fig, path); paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not trades.empty: trades.groupby("regime").net_return.agg(["mean", "count"])["mean"].plot.bar(ax=ax)
    ax.set_title(f"{title} — Regime-wise Performance")
    path = output / "regime_wise_performance.png"; _save(fig, path); paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not rejected.empty and "rejection_reason" in rejected:
        rejected.rejection_reason.value_counts().plot.bar(ax=ax)
    ax.set_title(f"{title} — Rejected Setup Analysis")
    path = output / "rejected_setup_analysis.png"; _save(fig, path); paths.append(path)
    return paths
