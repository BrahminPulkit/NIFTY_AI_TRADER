"""Fixed Grimes-inspired pullback/complexity rules."""
def entry(x):
    return x["ema_pullback"] & (x["momentum_3_atr"] >= 1.0) & (x["trend_direction"] != 0)
def exit_name(): return "fixed_target_stop_time"
