def apply(x,setup):
    direction=x["market_structure_shift"]
    aligned=((direction>0)&(x["close"]>x["ema20"]))|((direction<0)&(x["close"]<x["ema20"]))
    return setup & aligned
