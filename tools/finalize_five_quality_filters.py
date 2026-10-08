"""Small follow-up hardening; preserves notebook structure exactly."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
shape = [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs")) for c in nb["cells"]]

def replace(cell_index, old, new):
    text = "".join(nb["cells"][cell_index]["source"])
    if text.count(old) != 1:
        raise RuntimeError(f"cell {cell_index}: expected one match, found {text.count(old)}")
    nb["cells"][cell_index]["source"] = text.replace(old, new, 1).splitlines(keepends=True)

replace(4,
"""    if prefix == 'trend15':
        bars[f'{prefix}_hh'] = bars['high'] > bars['high'].shift(1)
        bars[f'{prefix}_hl'] = bars['low'] > bars['low'].shift(1)
        bars[f'{prefix}_lh'] = bars['high'] < bars['high'].shift(1)
        bars[f'{prefix}_ll'] = bars['low'] < bars['low'].shift(1)
""",
"""    if prefix == 'trend15':
        bar_session = bars['timestamp'].dt.strftime('%Y-%m-%d')
        previous_high = bars.groupby(bar_session)['high'].shift(1)
        previous_low = bars.groupby(bar_session)['low'].shift(1)
        bars[f'{prefix}_hh'] = bars['high'] > previous_high
        bars[f'{prefix}_hl'] = bars['low'] > previous_low
        bars[f'{prefix}_lh'] = bars['high'] < previous_high
        bars[f'{prefix}_ll'] = bars['low'] < previous_low
""")

replace(4,
"""        if not lot_size or entry_price <= 0 or bid <= 0:
            continue
""",
"""        if not lot_size or entry_price <= 0 or bid <= 0 or ask <= 0:
            continue
""")

replace(8,
"""        drawdown_pct = max(0.0,-daily_r*CFG.cost_r_per_trade)
""",
"""        # Existing replay is R-based; 1R is conservatively budgeted as 1% of paper capital.
        drawdown_pct = max(0.0,-daily_r*0.01)
""")

if shape != [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs")) for c in nb["cells"]]:
    raise RuntimeError("Notebook structure changed")
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
