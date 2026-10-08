import { Bot, Check, CircleAlert, ShieldCheck } from "lucide-react";
import type { DeskState, Signal } from "../../types/api";
import { percent, readable } from "../../utils/format";

export function SignalPanel({signal,desk,compact=false}:{signal?:Signal|null;desk?:DeskState;compact?:boolean}){
  const action=signal?.action||desk?.action||"NO SIGNAL";const confidence=signal?.confidence??desk?.confidence;
  const strategy=signal?.strategy||desk?.strategy?.name||"Waiting for completed candles";
  const reason=signal?.reason||desk?.reason||"No inference result yet";
  return <aside className={`signal-panel ${action==="BUY CE"?"trade":""} ${compact?"compact":""}`}>
    <header><span><Bot/>CURRENT SIGNAL</span><b>{signal||desk?"VERIFIED OUTPUT":"WAITING"}</b></header>
    <h2>{action}</h2><p>{readable(reason)}</p>
    <div className="signal-facts"><span>Confidence<b>{percent(confidence)}</b></span><span>Strategy<b>{readable(strategy)}</b></span><span>Risk<b>{signal?.risk_status||"Unavailable"}</b></span><span>Direction<b>{action==="BUY CE"?"CALL":"--"}</b></span></div>
    {!compact&&<><div className="primary-reason"><CircleAlert/><div><small>PRIMARY DECISION REASON</small><strong>{readable(reason)}</strong></div></div><div className="signal-safety"><ShieldCheck/><span>Frozen model and decision gates</span>{action==="BUY CE"?<Check/>:<b>BLOCKED</b>}</div></>}
  </aside>
}
