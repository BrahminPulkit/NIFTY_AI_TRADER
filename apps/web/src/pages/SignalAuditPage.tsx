import { Activity, AlertTriangle, CheckCircle2, SearchX } from "lucide-react";
import { PageHead } from "../components/ui/PageHead";
import { ErrorState, LoadingState } from "../components/ui/States";
import { useSignalAudit } from "../hooks/useTradingData";
import { percent, readable, time } from "../utils/format";
import "../signal-audit.css";

export function SignalAuditPage(){
  const audit=useSignalAudit();
  if(audit.isLoading)return <LoadingState/>;
  if(audit.error)return <ErrorState message={audit.error.message}/>;
  const data=audit.data!;const s=data.summary;
  return <><PageHead eyebrow="LIVE MODEL OBSERVABILITY" title="Signal Audit" detail="See exactly where each completed candle stopped in the production pipeline." side={<div className="audit-scope"><i/> {data.scope}</div>}/>
    <section className="audit-banner"><div><Activity/><span><b>{s.approved_signals?`${s.approved_signals} approved signal${s.approved_signals===1?"":"s"}`:"No signal approved"}</b><small>{s.model_evaluations} of {s.audited_candles} audited candles reached the model.</small></span></div><p>{data.methodology.note}</p></section>
    <div className="audit-metrics">
      <article><small>INFERENCE COVERAGE</small><b>{s.coverage_pct.toFixed(1)}%</b><span>{s.audited_candles} / {s.market_candles} market candles</span></article>
      <article><small>MODEL EVALUATED</small><b>{s.model_evaluations}</b><span>Stage-1 setups passed</span></article>
      <article><small>APPROVED SIGNALS</small><b>{s.approved_signals}</b><span>At {percent(data.threshold)} frozen threshold</span></article>
      <article className={s.review_candidates?"review":""}><small>REVIEW CANDIDATES</small><b>{s.review_candidates}</b><span>Retrospective, not signals</span></article>
      <article className={s.data_gaps?"warning":""}><small>UNAUDITED CANDLES</small><b>{s.data_gaps}</b><span>Coverage gap to investigate</span></article>
    </div>
    <div className="audit-grid"><section className="panel audit-funnel"><header><div><span>DECISION FUNNEL</span><h3>Why the model stayed silent</h3></div></header><div className="funnel-row"><span>Market candles</span><i style={{width:"100%"}}/><b>{s.market_candles}</b></div><div className="funnel-row"><span>Audited candles</span><i style={{width:`${Math.max(s.coverage_pct,2)}%`}}/><b>{s.audited_candles}</b></div><div className="funnel-row"><span>Model evaluated</span><i style={{width:`${Math.max(s.audited_candles?s.model_evaluations/s.audited_candles*100:0,2)}%`}}/><b>{s.model_evaluations}</b></div><div className="funnel-row approved"><span>Signals approved</span><i style={{width:`${Math.max(s.model_evaluations?s.approved_signals/s.model_evaluations*100:0,2)}%`}}/><b>{s.approved_signals}</b></div></section>
      <section className="panel audit-reasons"><header><div><span>TOP BLOCKERS</span><h3>Recorded rejection reasons</h3></div></header>{data.reason_breakdown.length?data.reason_breakdown.map(row=><div key={row.reason}><span>{readable(row.reason)}</span><b>{row.count}</b></div>):<p>No inference records for this session.</p>}</section></div>
    <section className="audit-table"><header><div><span>CANDLE-BY-CANDLE TRACE</span><h3>Latest production decisions</h3></div><small>Forward window: {data.methodology.forward_minutes} min | Review move: {data.methodology.review_move_pct}%</small></header><div className="table-scroll"><table><thead><tr><th>Time</th><th>Stage-1</th><th>Strategy</th><th>Model</th><th>Probability</th><th>Decision</th><th>Exact reason</th><th>Next {data.methodology.forward_minutes}m</th><th>Audit</th></tr></thead><tbody>{data.events.map(event=><tr key={event.timestamp} className={event.review_candidate?"review-row":""}><td>{time(event.timestamp)}</td><td><span className={`audit-state ${event.stage1==="PASS"?"pass":"blocked"}`}>{event.stage1==="PASS"?<CheckCircle2/>:<SearchX/>}{event.stage1}</span></td><td>{readable(event.setup)}</td><td>{event.model}</td><td>{event.probability==null?"Not run":percent(event.probability)}</td><td><b>{event.action}</b></td><td>{readable(event.reason)}</td><td>{event.forward_up_pct==null?"Window open":`+${event.forward_up_pct}% / ${event.forward_down_pct}%`}</td><td>{event.review_candidate?<span className="review-tag"><AlertTriangle/>Review</span>:"-"}</td></tr>)}</tbody></table></div>{!data.events.length&&<div className="audit-empty">No live inference records are available for {data.session_date}.</div>}</section>
  </>;
}
