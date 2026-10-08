import {
  Activity, Bot, Clock3, Gauge, ShieldCheck, Target, TrendingUp, WalletCards,
} from "lucide-react";
import { useState } from "react";
import { EmptyState, ErrorState, LoadingState } from "../components/ui/States";
import { PageHead } from "../components/ui/PageHead";
import { OrderReviewModal } from "../features/paper/OrderReviewModal";
import { useDesk, useOptions, usePaperActions } from "../hooks/useTradingData";
import type { PaperReview } from "../types/api";
import { money, percent, readable, time } from "../utils/format";

const APPROVED_THRESHOLD = 0.9;

export function TradingDeskPage() {
  const query = useDesk();
  const options = useOptions();
  const paper = usePaperActions("live");
  const [review, setReview] = useState<PaperReview | null>(null);
  if (query.isLoading) return <LoadingState />;
  if (query.error) return <ErrorState message={query.error.message} />;
  const desk = query.data?.desk;
  const market = query.data?.market;
  const strategySignal = market?.multistrategy_signal?.best_signal;
  if (!desk?.available) {
    return <><PageHead eyebrow="AI OPERATIONS" title="Trading Desk" detail="Your live decision workspace" />
      <EmptyState title="Waiting for a verified prediction" detail={desk?.reason || "No prediction is available"} /></>;
  }

  const tradeReady = desk.action !== "NO TRADE" && Boolean(desk.contract);
  const strategyName = desk.strategy.name === "Strategy unavailable"
    ? "No Active Setup" : desk.strategy.name;
  const strategySummary = strategyName === "No Active Setup"
    ? "Live scanner is running; no complete strategy pattern is present now."
    : desk.strategy.summary;
  const strategyWatch = strategyName === "No Active Setup"
    ? "Continue scanning; the model runs after a complete setup forms."
    : desk.strategy.watch;
  const contractId = String(desk.contract?.security_id || "");
  const liveContract = options.data?.rows.find(
    row => String(row.security_id) === contractId);
  const entry = Number(liveContract?.ask || liveContract?.ltp || 0);
  const planEntry = tradeReady ? entry : 0;
  const recommendedContract = tradeReady ? desk.contract : null;
  const noSetup = strategyName === "No Active Setup" || desk.reason.toUpperCase().includes("NO SETUP");
  const scanner = ["Pullback Breakout","Momentum Breakout","Extended Breakout","Continuation Breakout"];
  const writerTotal = desk.writer_activity.call_total_oi + desk.writer_activity.put_total_oi;
  const callShare = writerTotal ? desk.writer_activity.call_total_oi / writerTotal : 0.5;
  const putShare = 1 - callShare;
  const writerLead = Math.abs(desk.writer_activity.call_total_oi - desk.writer_activity.put_total_oi);
  const writerSentence = writerTotal === 0
    ? "Writer activity is waiting for live option-chain OI."
    : callShare > putShare
      ? `Call writers lead by ${money(writerLead)} OI; resistance pressure is stronger near ${money(desk.writer_activity.strongest_call?.strike)}.`
      : putShare > callShare
        ? `Put writers lead by ${money(writerLead)} OI; support pressure is stronger near ${money(desk.writer_activity.strongest_put?.strike)}.`
        : "Call and put writer activity is balanced.";
  const funnel = [
    ["Market data", market?.connected && !desk.freshness.stale ? "PASS" : "WAITING"],
    ["Stage-1 setup", noSetup ? "SCANNING" : "PASS"],
    ["Strategy selection", noSetup ? "NOT RUN" : strategyName],
    ["Decision Engine", noSetup ? "NOT RUN" : desk.action === "NO TRADE" ? "BLOCKED" : "PASS"],
    ["Model probability", noSetup ? "NOT RUN" : desk.probability >= APPROVED_THRESHOLD ? "PASS" : "BELOW THRESHOLD"],
    ["Risk check", desk.checks.find(item=>item.label==="Risk context")?.ready ? "READY" : "WAITING"],
    ["Option contract", desk.checks.find(item=>item.label==="Option contract")?.ready ? "READY" : "WAITING"],
  ];
  const startReview = () => paper.review.mutate(
    undefined, {onSuccess: setReview});
  const confirm = () => paper.open.mutate(
    undefined, {onSuccess: () => setReview(null)});
  const primaryBlocker = desk.checks.find(check => !check.ready)?.label;
  const blockerText: Record<string, string> = {
    "Trend direction": "Trend direction is not confirmed",
    "Strategy selected": "Strategy selection is not ready",
    "Decision gate": "The decision engine has not approved this setup",
    "Probability threshold": "Model confidence is below the approved threshold",
    "Risk context": "Verified risk values are unavailable",
    "Option contract": "A liquid option contract is not verified",
    "Fresh market data": "Market data is stale or disconnected",
  };
  const primaryReason = desk.action === "NO TRADE"
    ? primaryBlocker ? blockerText[primaryBlocker] || `${primaryBlocker} is not ready` : desk.reason
    : desk.reason;

  return <div className="trading-desk-v3">
    <PageHead eyebrow="AI OPERATIONS" title="Trading Desk"
      detail="One verified decision, its risk, and the next safe action."
      side={<div className={`desk-live-state ${market?.connected ? "online" : ""}`}>
        <i /><span>{market?.connected ? "LIVE MARKET" : "RESEARCH MODE"}</span>
        <small>{desk.freshness.age}</small>
      </div>} />

    <section className={`decision-hero ${desk.action === "BUY CE" ? "approved" : "blocked"}`}>
      <div className="decision-copy">
        <span className="decision-label"><Bot /> AI RECOMMENDATION</span>
        <h2>{desk.action}</h2>
        <p>{primaryReason}</p>
        <div className="next-action">
          <Clock3 />
          <span><small>WHAT HAPPENS NEXT</small><b>{desk.action === "NO TRADE" ? strategyWatch : "Review the verified option before opening a paper position."}</b></span>
        </div>
      </div>

      <div className="confidence-block">
        <div className="confidence-ring" style={{"--score": `${desk.probability * 360}deg`} as React.CSSProperties}>
          <div><strong>{percent(desk.probability)}</strong><small>MODEL CONFIDENCE</small></div>
        </div>
        <div className="threshold-line"><span style={{width:`${Math.min(desk.probability * 100, 100)}%`}} /><i style={{left:`${APPROVED_THRESHOLD * 100}%`}} /></div>
        <footer><span>Current {percent(desk.probability)}</span><b>Required {percent(APPROVED_THRESHOLD)}</b></footer>
      </div>

      <div className="decision-facts">
        <div><TrendingUp /><span>Strategy<b>{strategyName}</b></span></div>
        <div><Activity /><span>Market regime<b>{readable(desk.regime)}</b></span></div>
        <div><ShieldCheck /><span>Expected drawdown<b>{desk.expected_drawdown == null ? "Unavailable" : percent(desk.expected_drawdown)}</b></span></div>
      </div>
    </section>

    <section className={`put-shadow-strip ${market?.put_signal?.action === "BUY PE" ? "ready" : ""}`}>
      <div><Bot/><span><small>PUT MODEL - FORWARD PAPER WATCH</small><b>{market?.put_signal?.action || "WAITING FOR PE DATA"}</b></span></div>
      <span>Probability<b>{market?.put_signal?.probability == null ? "Not evaluated" : percent(market.put_signal.probability)}</b></span>
      <span>Paper threshold<b>{percent(market?.put_signal?.threshold || .85)}</b></span>
      <span>Reason<b>{readable(market?.put_signal?.reason || market?.put_signal_status)}</b></span>
      <em>Paper-only validation. Broker orders remain disabled.</em>
    </section>

    <section className={`put-shadow-strip strategy-shadow ${strategySignal ? "ready" : ""}`}>
      <div><Bot/><span><small>RETRAINED MULTI-STRATEGY - PAPER ONLY</small><b>{strategySignal?.action || "NO APPROVED SETUP"}</b></span></div>
      <span>Strategy<b>{readable(strategySignal?.strategy || "Scanning pullback and continuation")}</b></span>
      <span>Probability<b>{strategySignal ? percent(strategySignal.probability) : "Not approved"}</b></span>
      <span>Threshold<b>{strategySignal ? percent(strategySignal.threshold) : "Strategy specific"}</b></span>
      <em>{market?.multistrategy_signal?.checks.length || 0} active checks. Paper-only; broker orders disabled.</em>
    </section>

    <section className="desk-action-grid">
      <article className="option-command">
        <header><div><span>TRADE PLAN</span><h3>{recommendedContract ? `NIFTY ${money(recommendedContract.strike)} ${recommendedContract.option_type}` : "Locked until trade approval"}</h3></div><b>{recommendedContract ? "LIVE QUOTE" : "LOCKED"}</b></header>
        <div className="option-facts"><span>Expiry<b>{recommendedContract?readable(recommendedContract.expiry):"--"}</b></span><span>Entry ask<b>{planEntry?`₹${money(planEntry)}`:"--"}</b></span><span>Stop loss<b>{planEntry?`₹${money(planEntry*.8)}`:"--"}</b></span><span>Target<b>{planEntry?`₹${money(planEntry*1.05)}`:"--"}</b></span></div>
        <p>{recommendedContract ? "Indicative paper plan; API revalidates the latest quote before entry." : "No CE/PE recommendation is shown until every decision gate passes."}</p>
      </article>
      <article className="risk-command">
        <header><div><span>RISK &amp; CAPITAL</span><h3>{tradeReady ? "Review required" : "Trade controls locked"}</h3></div><Gauge /></header>
        <div className="risk-facts"><span><Target />Maximum loss<b>{planEntry?`₹${money(planEntry*.2)}`:"--"}</b></span><span><ShieldCheck />Reward / risk<b>{planEntry?"0.25 R":"--"}</b></span><span><WalletCards />Capital required<b>{planEntry?`₹${money(planEntry)}`:"--"}</b></span></div>
        <button disabled={!tradeReady || paper.review.isPending} onClick={startReview}>
          <WalletCards />{paper.review.isPending ? "Checking signal..." : "Review paper order"}
        </button>
        {!tradeReady && <small>Available only after signal and contract verification.</small>}
        {paper.review.error && <small>{paper.review.error.message}</small>}
      </article>
    </section>

    <section className="desk-intelligence-grid">
      <article className="panel strategy-now">
        <header><div><span>MODEL STRATEGY</span><h3>{strategyName}</h3></div>
          <b>{desk.action === "NO TRADE" ? "NOT APPROVED" : "ACTIVE"}</b></header>
        <p>{strategySummary}</p>
        <div className="scanner-status"><Activity /><span>{noSetup
          ? `Scanning ${scanner.length} approved strategy playbooks`
          : `${strategyName} is the active model playbook`}</span>
          <b>{noSetup ? "SCANNING" : "ACTIVE"}</b></div>
        <footer><Activity/><span>{strategyWatch}</span></footer>
      </article>
      <article className="panel writer-activity">
        <header><div><span>OPTION WRITER ACTIVITY</span><h3>{readable(desk.writer_activity.dominant)}</h3></div>
          <b>OI PROXY</b></header>
        <div className="writer-grid">
          <div className="call-writer"><span>Strongest call writer</span>
            <strong>{money(desk.writer_activity.strongest_call?.strike)} CE</strong>
            <small>OI {money(desk.writer_activity.strongest_call?.oi)} · Resistance proxy</small></div>
          <div className="put-writer"><span>Strongest put writer</span>
            <strong>{money(desk.writer_activity.strongest_put?.strike)} PE</strong>
            <small>OI {money(desk.writer_activity.strongest_put?.oi)} · Support proxy</small></div>
        </div>
        <div className="writer-comparison"><div><span style={{width:`${callShare * 100}%`}} /><i style={{width:`${putShare * 100}%`}} /></div>
          <small>CALL {percent(callShare)}<b>PUT {percent(putShare)}</b></small></div>
        <p className="writer-reading">{writerSentence}</p>
      </article>
    </section>

    <section className="desk-funnel panel">
      <header><div><span>DECISION FUNNEL</span><h3>Where the current evaluation stands</h3></div><b>{desk.action}</b></header>
      <div>{funnel.map(([label,status],index)=><div className={status==="PASS"||status==="READY"?"ready":status==="BLOCKED"||status==="BELOW THRESHOLD"?"blocked":"waiting"} key={label}>
        <i>{index+1}</i><span>{label}</span><b>{status}</b></div>)}</div>
    </section>

    <section className="trading-diary panel">
      <header><div><span>AI TRADING DIARY</span><h3>Current model reading</h3></div><b>{time(desk.timestamp)} IST</b></header>
      <div className="diary-lines">
        <div><i>1</i><span><small>MODEL VERDICT</small><b>{desk.action}</b><p>{readable(desk.reason)}</p></span></div>
        <div><i>2</i><span><small>STRATEGY</small><b>{strategyName}</b><p>{strategySummary}</p></span></div>
        <div><i>3</i><span><small>WRITER READING</small><b>{readable(desk.writer_activity.dominant)}</b><p>{writerSentence}</p></span></div>
        <div><i>4</i><span><small>NEXT ACTION</small><b>{tradeReady ? "Review paper order" : "Wait; do not enter"}</b><p>{tradeReady ? "A verified live trade plan is ready for review." : strategyWatch}</p></span></div>
      </div>
    </section>
    {review && <OrderReviewModal review={review} busy={paper.open.isPending}
      error={paper.open.error?.message} onConfirm={confirm}
      onClose={() => setReview(null)} />}
  </div>;
}
