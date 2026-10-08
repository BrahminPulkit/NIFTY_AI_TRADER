"""Diagnostic research charts, including rejected setups."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_research_dashboard(features: pd.DataFrame, output: Path, title: str = "Institutional Hybrid") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    ax = axes[0]
    ax.plot(features.index, features["close"], label="Futures", linewidth=1)
    for ema in ("ema20", "ema50", "ema200"):
        ax.plot(features.index, features[ema], label=ema.upper(), alpha=.8)
    entries = features["signal"] != 0
    rejects = features["rejection_reason"].ne("")
    ax.scatter(features.index[entries], features.loc[entries, "close"], marker="^", color="green", label="Entry")
    ax.scatter(features.index[rejects], features.loc[rejects, "close"], marker="x", color="grey", s=12, label="Rejected")
    ax.legend(ncol=6)
    axes[1].plot(features.index, features["atr_expansion"], label="ATR expansion")
    axes[1].plot(features.index, features["range_expansion"], label="Range expansion")
    axes[1].legend()
    axes[2].plot(features.index, features["institutional_score"], label="Institutional score")
    axes[2].plot(features.index, features["premium_expansion_score"], label="Premium expansion score")
    axes[2].axhline(65, color="red", linestyle="--")
    axes[2].legend()
    axes[3].plot(features.index, features["premium"], label="ATM premium")
    axes[3].fill_between(features.index, 0, features["relative_volume"], alpha=.25, label="Relative volume")
    axes[3].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output
