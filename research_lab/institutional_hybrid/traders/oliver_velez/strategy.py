import numpy as np
from .rules import entry
from .filters import apply
def signal(x):
    out=x.copy(); raw=entry(out).fillna(False); accepted=apply(out,raw).fillna(False)
    out["signal"]=np.where(accepted,out["trend_direction"],0).astype(int)
    out["strategy_score"]=(out["score_trend"]+out["score_volume"]+out["score_momentum"]).clip(0,100)
    out["rejection_reason"]=np.where(raw & ~accepted,"trend_expansion_filter","")
    return out
