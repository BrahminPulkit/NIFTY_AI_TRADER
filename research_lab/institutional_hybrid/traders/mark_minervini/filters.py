def apply(x,setup):
    aligned=((x["breakout_direction"]>0)&(x["trend_direction"]>=0))|((x["breakout_direction"]<0)&(x["trend_direction"]<=0))
    return setup & aligned
