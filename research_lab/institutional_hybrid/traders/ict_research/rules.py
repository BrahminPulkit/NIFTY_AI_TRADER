"""Restricted ICT research: liquidity sweep plus structure shift only."""
def entry(x):
    return x["false_break"].shift().fillna(False) & (x["market_structure_shift"] != 0)
def exit_name(): return "fixed_target_stop_time"
