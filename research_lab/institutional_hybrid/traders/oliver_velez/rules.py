"""Fixed Velez-inspired trend and expansion rules."""
def entry(x):
    return (x["trend_direction"] != 0) & x["continuation"] & (x["relative_volume"] >= 1.5)
def exit_name(): return "fixed_target_stop_time"
