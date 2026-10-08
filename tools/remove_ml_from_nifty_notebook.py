"""Remove ML/CatBoost gating from the institutional scalping notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(PATH.read_text(encoding="utf-8"))
if len(nb["cells"]) != 12:
    raise RuntimeError("Unexpected notebook cell count; refusing to edit")


def set_source(index: int, text: str) -> None:
    nb["cells"][index]["source"] = text.splitlines(keepends=True)


cell0 = "".join(nb["cells"][0]["source"]).replace(
    "volume profile, option-chain confirmation, frozen-ML gate and portfolio risk controls.",
    "volume profile, option-chain confirmation, confidence scoring and portfolio risk controls.",
)
set_source(0, cell0)

cell2 = "".join(nb["cells"][2]["source"])
cell2 = cell2.replace(
    """    ml = _ml_confirmation(index_bars, option, setup['direction'])
    if not ml['approved']:
        print(f"NO TRADE — {ml['reason']}"); return None
""",
    "",
)
cell2 = cell2.replace(
    "    result.update(option=option, ml=ml, option_entry=premium_entry,\n",
    "    result.update(option=option, option_entry=premium_entry,\n",
)
cell2 = cell2.replace(
    """    print(f"PCR {option['pcr']:.2f} | ML {ml['probability']:.3f} | premium {premium_entry:.2f} | SL {premium_stop:.2f} | target {premium_target:.2f}")
""",
    """    print(f"PCR {option['pcr']:.2f} | premium {premium_entry:.2f} | SL {premium_stop:.2f} | target {premium_target:.2f}")
""",
)
set_source(2, cell2)

cell3 = "".join(nb["cells"][3]["source"])
cell3 = cell3.replace("    ml_enabled: bool = True\n", "")
cell3 = cell3.replace("    ml_probability_threshold: float = 0.90\n", "")
set_source(3, cell3)

cell4 = "".join(nb["cells"][4]["source"])
start = cell4.find("def _ml_confirmation(")
end = cell4.find("def evaluate_baseline_setup(", start)
if start >= 0 and end >= 0:
    cell4 = cell4[:start] + cell4[end:]
set_source(4, cell4)

cell10 = "".join(nb["cells"][10]["source"])
cell10 = cell10.replace(
    '"""Unified closed-candle signal, option selection, ML and premium management."""',
    '"""Unified closed-candle rule-based signal and option-premium management."""',
)
old_entry = """                    if option is not None:
                        ml = _ml_confirmation(index_bars,option,setup['direction'])
                        if ml['approved']:
                            entry = float(option['entry_price'])
                            risk_pct = float(np.clip(setup['risk']/setup['entry']*option['delta']*4,.15,.30))
                            initial_risk = entry*risk_pct
                            symbol = f"NIFTY {option['expiry']} {option['strike']:.0f} {option['option_type']}"
                            position = {**setup, **option, 'symbol':symbol, 'entry':entry,
                                'stop':entry-initial_risk,
                                'target':entry+initial_risk*setup['planned_r'],
                                'initial_risk':initial_risk, 'best_price':entry,
                                'underlying_entry':setup['entry'], 'ml_probability':ml['probability']}
                            used_events.add(setup['event_time']); last_entry = row.timestamp
                            print(f"\\nPAPER ENTRY {symbol} | {setup['setup']} | "
                                  f"confidence {setup['confidence_score']:.1f} | ML {ml['probability']:.3f}")
                        else:
                            print(f"\\nML SAFE BLOCK: {ml['reason']}")
"""
new_entry = """                    if option is not None:
                        entry = float(option['entry_price'])
                        risk_pct = float(np.clip(setup['risk']/setup['entry']*option['delta']*4,.15,.30))
                        initial_risk = entry*risk_pct
                        symbol = f"NIFTY {option['expiry']} {option['strike']:.0f} {option['option_type']}"
                        position = {**setup, **option, 'symbol':symbol, 'entry':entry,
                            'stop':entry-initial_risk,
                            'target':entry+initial_risk*setup['planned_r'],
                            'initial_risk':initial_risk, 'best_price':entry,
                            'underlying_entry':setup['entry']}
                        used_events.add(setup['event_time']); last_entry = row.timestamp
                        print(f"\\nPAPER ENTRY {symbol} | {setup['setup']} | "
                              f"confidence {setup['confidence_score']:.1f}/100 | PCR {option['pcr']:.2f}")
"""
if old_entry not in cell10:
    raise RuntimeError("Expected live ML entry block not found")
cell10 = cell10.replace(old_entry, new_entry)
set_source(10, cell10)

PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Removed ML logic from {PATH}")
