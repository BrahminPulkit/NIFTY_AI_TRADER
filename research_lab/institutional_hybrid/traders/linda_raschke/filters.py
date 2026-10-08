def apply(x, setup):
    return setup & x["regime"].isin(["bull_trend", "bear_trend", "expansion"])
