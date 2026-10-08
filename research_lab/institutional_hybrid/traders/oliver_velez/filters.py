def apply(x,setup):
    return setup & (x["ema_spread_atr"] >= .5) & (x["range_expansion"] >= 1.2)
