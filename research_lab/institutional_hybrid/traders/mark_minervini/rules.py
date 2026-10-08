"""Fixed intraday adaptation of Minervini-inspired contraction/breakout."""
def entry(x):
    return x["compression_breakout"] & (x["breakout_direction"] != 0) & (x["relative_volume"] >= 1.5)
def exit_name(): return "fixed_target_stop_time"
