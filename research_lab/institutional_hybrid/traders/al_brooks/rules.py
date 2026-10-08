"""Fixed Brooks-inspired price-action rules."""
def entry(x):
    return x["trend_bar"] & (x["pullback_bar"].shift().fillna(False)) & (x["trend_direction"] != 0)
def exit_name(): return "fixed_target_stop_time"
