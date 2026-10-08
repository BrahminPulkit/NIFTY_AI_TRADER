import numpy as np
from .rules import entry
from .filters import apply
def signal(x):
    out=x.copy(); raw=entry(out).fillna(False); accepted=apply(out,raw).fillna(False)
    out["signal"]=np.where(accepted,out["breakout_direction"],0).astype(int)
    out["strategy_score"]=(out["score_breakout"]+out["score_volume"]+out["score_trend"]).clip(0,100)
    out["rejection_reason"]=np.where(raw & ~accepted,"trend_template_filter","")
    return out
