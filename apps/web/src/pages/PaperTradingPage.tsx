import { ArrowDown, BookOpen, LogOut, ShieldCheck, WalletCards } from "lucide-react";
import { useState } from "react";
import { JournalTable } from "../components/data/JournalTable";
import { Metric } from "../components/ui/Metric";
import { PageHead } from "../components/ui/PageHead";
import { ErrorState, LoadingState } from "../components/ui/States";
import { ManualPaperOrder } from "../features/paper/ManualPaperOrder";
import { OrderReviewModal } from "../features/paper/OrderReviewModal";
import { useDesk, usePaperAccount, usePaperActions } from "../hooks/useTradingData";
import type { PaperReview } from "../types/api";
import { money, percent, readable, time } from "../utils/format";

const flow = ["Signal", "Review", "Paper Entry", "Open Position", "Exit", "Journal"];

export function PaperTradingPage() {
  const account = usePaperAccount();
  const desk = useDesk();
  const actions = usePaperActions("live");
  const [mode, setMode] = useState<"ai"|"manual">("ai");
  const [review, setReview] = useState<PaperReview|null>(null);
  if (account.isLoading) return <LoadingState />;
  if (account.error) return <ErrorState message={account.error.message} />;
  const data = account.data!;
  const position = data.position;
  const stats = data.statistics;
  const signalReady = desk.data?.desk?.action === "BUY CE"
    && Boolean(desk.data?.desk?.contract);
  const startReview = () => actions.review.mutate(
    undefined, {onSuccess:setReview});
  const confirm = () => actions.open.mutate(
    undefined, {onSuccess:()=>setReview(null)});
  const positionLabel = position
    ? `${money(position.strike)} ${position.option_type || "CE"}`
    : "No open paper position";

  return <>
    <PageHead eyebrow="SIMULATION" title="Paper Trading"
      detail="AI-approved or manually selected options, tracked on live market quotes."
      side={<div className="paper-mode"><ShieldCheck/>PAPER ONLY</div>}/>
    <div className="paper-entry-mode segments">
      <button className={mode==="ai"?"active":""} onClick={()=>setMode("ai")}>AI Signal</button>
      <button className={mode==="manual"?"active":""} onClick={()=>setMode("manual")}>Manual Option</button>
    </div>
    <div className="paper-flow">{flow.map((step,index)=><span
      className={position&&index<=3?"active":!position&&index===0&&signalReady?"active":""}
      key={step}><i>{index===0?<ShieldCheck/>:index===5?<BookOpen/>:<WalletCards/>}</i>
      <b>{step}</b>{index<flow.length-1&&<ArrowDown/>}</span>)}</div>
    <div className="metric-row">
      <Metric label="Paper capital" value={`₹${money(data.capital)}`} detail={`Initial ₹${money(data.initial_capital)}`}/>
      <Metric label="Win rate" value={percent(Number(stats.win_rate)||0)} detail={`${Number(stats.total_trades)||0} completed trades`}/>
      <Metric label="Net P&L" value={`₹${money(Number(stats.net_pnl)||0)}`} detail={`Maximum ${data.rules.maximum_daily_trades} trades/day`}/>
    </div>
    <section className="position-workspace">
      <div className={`panel live-position ${position?"open":""}`}>
        <header><div><span>CURRENT POSITION</span><h3>{positionLabel}</h3></div><b>{position?"OPEN":"INACTIVE"}</b></header>
        <div className="position-grid">
          <span>Entry<b>₹{money(position?.entry_premium)}</b></span>
          <span>Current bid<b>₹{money(position?.current_premium)}</b></span>
          <span>Live P&amp;L<b className={(position?.current_pnl||0)>=0?"profit":"loss"}>₹{money(position?.current_pnl)}</b></span>
          <span>Quantity<b>{position?.quantity||"--"}</b></span>
          <span>Stop<b>₹{money(position?.stop_loss)}</b></span>
          <span>Target<b>₹{money(position?.target)}</b></span>
        </div>
        {position&&<div className="position-meta">
          <span>Entry time<b>{time(position.entry_timestamp)}</b></span>
          <span>Mode<b>{readable(position.entry_mode)}</b></span>
          <span>MFE<b>{percent(position.maximum_favourable_excursion)}</b></span>
          <span>MAE<b>{percent(position.maximum_adverse_excursion)}</b></span>
        </div>}
        <button className="danger-action" disabled={!position||actions.exit.isPending}
          onClick={()=>actions.exit.mutate(undefined)}><LogOut/>
          {actions.exit.isPending?"Closing...":"Exit paper position"}</button>
      </div>
      {mode==="manual"?<ManualPaperOrder positionOpen={Boolean(position)}/>:<div className="panel">
        <header><div><span>NEXT AI PAPER ORDER</span><h3>{desk.data?.desk?.action||"No approved signal"}</h3></div><b>{desk.data?.desk?.freshness.label}</b></header>
        <div className="paper-rules"><span>Quantity<b>{data.rules.quantity}</b></span><span>Stop<b>{percent(data.rules.stop_loss_pct)}</b></span><span>Target<b>{percent(data.rules.target_pct)}</b></span><span>Maximum hold<b>{data.rules.maximum_holding_minutes} min</b></span></div>
        <p className="muted-copy">AI entry unlocks only after strategy, probability, risk, live contract and candle alignment pass.</p>
        <button className="primary-action" disabled={Boolean(position)||!signalReady||actions.review.isPending}
          onClick={startReview}>{actions.review.isPending?"Checking...":"Review AI paper signal"}</button>
        {!signalReady&&<p className="form-message">Waiting for an approved BUY CE signal.</p>}
        {actions.review.error&&<p className="form-message">{actions.review.error.message}</p>}
      </div>}
    </section>
    <section className="panel"><header><div><span>TRADE JOURNAL</span><h3>Completed live paper positions</h3></div><b>{data.trades.length} records</b></header>
      {data.trades.length?<JournalTable rows={data.trades}/>:<div className="empty compact"><BookOpen/><h3>No completed paper trades</h3><p>AI and manual entries will be journaled automatically.</p></div>}
    </section>
    {review&&<OrderReviewModal review={review} busy={actions.open.isPending}
      error={actions.open.error?.message} onConfirm={confirm} onClose={()=>setReview(null)}/>} 
  </>;
}
