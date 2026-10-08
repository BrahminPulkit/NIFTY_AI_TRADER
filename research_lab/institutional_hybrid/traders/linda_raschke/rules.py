"""Fixed Raschke-inspired trend-continuation rules."""
def entry(x):
    return x["ema_pullback"] & (x["trend_direction"] != 0) & x["continuation"]
def exit_name(): return "fixed_target_stop_time"
