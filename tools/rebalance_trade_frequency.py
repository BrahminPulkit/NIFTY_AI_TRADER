"""Surgically rebalance redundant hard gates without weakening core safety filters."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
backup = path.with_name("NIFTY_Scalping_Paper_Trading.pre_frequency_rebalance.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
shape = [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs")) for c in nb["cells"]]
if not backup.exists():
    backup.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

def replace(cell, old, new, label):
    text = "".join(nb["cells"][cell]["source"])
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, found {text.count(old)}")
    nb["cells"][cell]["source"] = text.replace(old, new, 1).splitlines(keepends=True)

# Retain quality scoring, but stop demanding institutional-grade confidence on every scalp.
replace(3, "    minimum_confidence_score: float = 70.0\n",
        "    minimum_confidence_score: float = 65.0\n", "confidence")
replace(3, "    early_entry_confidence: float = 74.0\n",
        "    early_entry_confidence: float = 68.0\n", "early confidence")
replace(3, "    first_hour_min_confidence: float = 75.0\n",
        "    first_hour_min_confidence: float = 70.0\n", "first-hour confidence")

# A 15m HH/HL trend remains active until a completed 15m bar confirms the opposite
# structure. This avoids treating every inside candle as a trend reset.
replace(4,
"""        bars[f'{prefix}_direction'] = np.select(
            [bars[f'{prefix}_hh'] & bars[f'{prefix}_hl'],
             bars[f'{prefix}_lh'] & bars[f'{prefix}_ll']], [1, -1], default=0)
        bars[f'{prefix}_structure'] = np.select(
            [bars[f'{prefix}_direction'] == 1, bars[f'{prefix}_direction'] == -1],
            ['HH_HL', 'LH_LL'], default='TRANSITION')
""",
"""        raw_direction = pd.Series(np.select(
            [bars[f'{prefix}_hh'] & bars[f'{prefix}_hl'],
             bars[f'{prefix}_lh'] & bars[f'{prefix}_ll']], [1, -1], default=0),
            index=bars.index)
        bars[f'{prefix}_direction'] = (raw_direction.replace(0,np.nan)
            .groupby(bar_session).ffill().fillna(0).astype(int))
        bars[f'{prefix}_structure'] = np.select(
            [bars[f'{prefix}_direction'] == 1, bars[f'{prefix}_direction'] == -1],
            ['HH_HL', 'LH_LL'], default='TRANSITION')
""", "persistent 15m structure")

# Supertrend and the separate 1m pivot structure remain features and score inputs,
# but are no longer redundant vetoes after 15m/5m/1m alignment has already passed.
replace(4,
"""    if not mtf_ok or not supertrend_ok:
        return None
""",
"""    if not mtf_ok:
        return None
""", "supertrend veto")
replace(4,
"""    if not (structure_ok and regime_ok and vwap_ok and adx_ok and atr_ok
            and one_minute_confirmation and not fake_against):
""",
"""    if not (regime_ok and vwap_ok and adx_ok and atr_ok
            and one_minute_confirmation and not fake_against):
""", "structure veto")

# ADX remains a mandatory >22 gate. With that hard gate in place, one additional
# volatility/width/EMA-distance vote is sufficient to classify a tradable trend.
replace(4,
"""    data['market_regime'] = np.where(regime_votes >= 3, 'TRENDING', 'SIDEWAYS')
""",
"""    data['market_regime'] = np.where(regime_votes >= 2, 'TRENDING', 'SIDEWAYS')
""", "regime vote")

# Early EMA/VWAP reclaim setups now use the higher first-hour score instead of a
# blanket rejection. Opening risk cap and cooldown remain unchanged.
replace(4,
"""    if confidence < threshold or weak_early_reclaim:
        return None
""",
"""    if confidence < threshold:
        return None
""", "early reclaim veto")

if shape != [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs"))
             for c in nb["cells"]]:
    raise RuntimeError("Notebook structure changed")
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
print(backup)
