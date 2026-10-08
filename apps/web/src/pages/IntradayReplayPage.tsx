import {
  Activity, CalendarDays, Check, ChevronRight, Clock3, Database,
  Pause, Play, RotateCcw, ShieldCheck, SkipForward, Target,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { PageHead } from "../components/ui/PageHead";
import { EmptyState, ErrorState, LoadingState } from "../components/ui/States";
import { MarketChart } from "../features/charts/MarketChart";
import { usePaperAccount, usePaperActions, useReplaySession, useReplaySessions } from "../hooks/useTradingData";
import { money, percent, readable, time } from "../utils/format";

type ReplaySource = "rolling" | "dhan" | "frozen";

const sourceLabel: Record<ReplaySource, string> = {
  rolling: "Rolling ATM",
  dhan: "Dhan Cache",
  frozen: "Legacy OOF",
};

export function IntradayReplayPage() {
  const [source, setSource] = useState<ReplaySource>("rolling");
  const [date, setDate] = useState("");
  const [cursor, setCursor] = useState(30);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const marked = useRef("");
  const sessions = useReplaySessions(source);
  const replay = useReplaySession(date, source);
  const paper = usePaperAccount("replay", date);
  const paperActions = usePaperActions("replay", date);

  useEffect(() => setDate(sessions.data?.latest || ""), [sessions.data?.latest]);

  const candles = replay.data?.candles || [];
  useEffect(() => {
    setCursor(Math.min(30, Math.max(candles.length, 1)));
    setPlaying(false);
    marked.current = "";
  }, [date, candles.length]);

  useEffect(() => {
    if (!playing || !candles.length) return;
    const id = window.setInterval(() => setCursor(value => {
      if (value >= candles.length) {
        setPlaying(false);
        return value;
      }
      return value + 1;
    }), Math.max(60, 1000 / speed));
    return () => window.clearInterval(id);
  }, [playing, speed, candles.length]);

  const visible = candles.slice(0, cursor);
  const current = visible.at(-1);
  const currentTime = current ? Date.parse(current.timestamp) : 0;
  const productionEvents = useMemo(() => replay.data?.events.filter(
    event => Date.parse(event.timestamp) <= currentTime) || [], [replay.data, currentTime]);
  const legacyPutEvents = useMemo(() => replay.data?.put_events.filter(
    event => Date.parse(event.timestamp) <= currentTime) || [], [replay.data, currentTime]);
  const strategyEvents = useMemo(() => replay.data?.strategy_events.filter(
    event => Date.parse(event.timestamp) <= currentTime) || [], [replay.data, currentTime]);
  const strategySource = source !== "frozen";
  const callEvents = strategySource
    ? strategyEvents.filter(event => event.side === "CALL") : productionEvents;
  const putEvents = strategySource
    ? strategyEvents.filter(event => event.side === "PUT") : legacyPutEvents;
  const entries = useMemo(() => [...callEvents, ...putEvents].filter(
    event => event.action === "BUY CE" || event.action === "BUY PE"), [callEvents, putEvents]);
  const latestCall = callEvents.at(-1);
  const latestPut = putEvents.at(-1);
  const chartSignal = latestCall ? { ...latestCall, markers: entries } : undefined;
  const complete = candles.length > 0 && cursor >= candles.length;
  const progress = candles.length ? Math.round(cursor / candles.length * 100) : 0;
  const modelBlocked = strategySource && replay.data?.summary.model_status === "MODEL_VALIDATION_BLOCKED";

  useEffect(() => {
    const position = paper.data?.position;
    if (source !== "frozen" || !position || !current ||
        Date.parse(current.timestamp) <= Date.parse(position.entry_timestamp) ||
        marked.current === current.timestamp || paperActions.mark.isPending) return;
    marked.current = current.timestamp;
    paperActions.mark.mutate(current.timestamp);
  }, [source, current?.timestamp, paper.data?.position?.trade_id]);

  const switchSource = (next: ReplaySource) => {
    if (next === source) return;
    setPlaying(false);
    setDate("");
    setSource(next);
  };

  if (sessions.isLoading) return <LoadingState />;
  if (sessions.error) return <ErrorState message={sessions.error.message} />;

  return <>
    <PageHead eyebrow="HISTORICAL SIMULATION" title="Intraday Replay"
      detail="Watch the model evaluate correctly aligned ATM CALL and PUT contracts candle by candle."
      side={<div className="replay-source"><i />{sourceLabel[source]}</div>} />

    <section className="replay-command-bar">
      <div className="replay-source-switch" aria-label="Replay data source">
        {(["rolling", "dhan", "frozen"] as ReplaySource[]).map(value =>
          <button key={value} className={source === value ? "active" : ""}
            onClick={() => switchSource(value)}>{sourceLabel[value]}</button>)}
      </div>
      <label className="replay-date"><CalendarDays /><span>Trade date</span>
        <select value={date} onChange={event => setDate(event.target.value)}>
          <option value="">Select date</option>
          {sessions.data?.sessions.map(value => <option value={value} key={value}>
            {new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", {
              day: "2-digit", month: "short", year: "numeric",
            })}
          </option>)}
        </select>
      </label>
      <div className="replay-controls">
        <button className="primary" onClick={() => setPlaying(value => !value)}
          disabled={replay.isLoading || complete || !candles.length}>
          {playing ? <Pause /> : <Play />}{playing ? "Pause" : "Play"}
        </button>
        <button title="Reveal next candle" onClick={() => setCursor(value => Math.min(value + 1, candles.length))}
          disabled={!candles.length || complete}><SkipForward /></button>
        <button title="Restart replay" onClick={() => { setCursor(Math.min(30, candles.length)); setPlaying(false); }}
          disabled={!candles.length}><RotateCcw /></button>
      </div>
      <label className="replay-speed"><span>Speed</span>
        <select value={speed} onChange={event => setSpeed(Number(event.target.value))}>
          {[1, 2, 5, 10, 20].map(value => <option key={value} value={value}>{value}x</option>)}
        </select>
      </label>
      <div className="replay-progress"><span><i style={{ width: `${progress}%` }} /></span>
        <b>{cursor}/{candles.length}</b><small>{progress}%</small></div>
    </section>

    {replay.isLoading ? <LoadingState /> : replay.error ? <ErrorState message={replay.error.message} /> :
      !candles.length ? <EmptyState title="No synchronized replay data"
        detail={source === "dhan" ? "Connect Dhan or choose Rolling ATM for historical validation." : "Choose another trade date."} /> : <>
      <section className={`replay-result-banner ${entries.length ? "has-entry" : ""} ${modelBlocked ? "model-blocked" : ""}`}>
        <div><span>{modelBlocked ? "MODEL SAFETY BLOCK" : complete ? "SESSION RESULT" : "REPLAY IN PROGRESS"}</span>
          <h2>{modelBlocked ? "CALL/PUT model is not validated" : entries.length ? `${entries.length} approved ${entries.length === 1 ? "entry" : "entries"}` : "No entry revealed yet"}</h2>
          {modelBlocked && <small>{replay.data?.summary.model_reason}</small>}</div>
        <div><Database /><span>Data contract<b>{source === "rolling" ? "Rolling ATM CE + PE" : sourceLabel[source]}</b></span></div>
        <div><Clock3 /><span>Replay time<b>{time(current?.timestamp)} IST</b></span></div>
        <div><ShieldCheck /><span>Data quality<b>{replay.data?.summary.data_valid === false ? "Incomplete" : "Aligned"}</b></span></div>
      </section>

      <div className="replay-terminal">
        <section className="chart-panel replay-chart">
          <header><div><b>NIFTY 50</b><span>{date} · Rolling ATM decision chart</span></div>
            <strong>{money(current?.close)}</strong></header>
          <MarketChart rows={visible} signal={chartSignal} storageKey="intraday-replay" />
        </section>
        <aside className="replay-decision-rail">
          <DecisionCard side="CALL" event={latestCall} premium={current?.option_close}
            strike={current?.call_strike} fallbackThreshold={replay.data?.threshold} />
          <DecisionCard side="PUT" event={latestPut} premium={current?.put_option_close}
            strike={current?.put_strike} fallbackThreshold={replay.data?.put_threshold} />
        </aside>
      </div>

      <section className="replay-ledger">
        <div className="replay-timeline">
          <header><div><span>CHRONOLOGICAL MODEL CHECKS</span><b>Entries and rejected setups</b></div>
            <strong>{entries.length} APPROVED</strong></header>
          <div className="replay-table-head"><span>Time</span><span>Side</span><span>Strategy / reason</span><span>Score</span><span>Decision</span></div>
          {[...callEvents, ...putEvents].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, 12).map((event, index) => {
            const isPut = event.action === "BUY PE" || event.source.includes("PUT");
            const approved = event.action === "BUY CE" || event.action === "BUY PE";
            const threshold = event.threshold ?? (isPut ? replay.data?.put_threshold : replay.data?.threshold);
            return <div className={`replay-event-row ${approved ? "approved" : "rejected"}`} key={`${event.timestamp}-${event.source}-${index}`}>
              <time>{time(event.timestamp)}</time><strong className={isPut ? "put" : "call"}>{isPut ? "PUT" : "CALL"}</strong>
              <span><b>{event.strike ? `${event.strike.toFixed(0)} ${isPut ? "PE" : "CE"} · ` : ""}{readable(event.strategy)}</b>
                <small>{approved ? `Premium INR ${money(event.entry_premium)} · NIFTY ${money(event.index_price)}` : readable(event.reason)}</small></span>
              <em>{percent(event.probability)}<small>Need {percent(threshold)}</small></em>
              <i>{approved ? "BUY" : "REJECTED"}</i>
            </div>;
          })}
          {!callEvents.length && !putEvents.length && <p>{modelBlocked
            ? "Replay data is available, but scoring is disabled until a model passes the corrected option-contract validation."
            : "No strategy setup has been evaluated at this replay time."}</p>}
        </div>
        <aside className="replay-session-summary">
          <header><span>SESSION SUMMARY</span><b>{complete ? "COMPLETE" : "LIVE REVEAL"}</b></header>
          <div><Target /><span>Approved CALL<b>{entries.filter(event => event.action === "BUY CE").length}</b></span></div>
          <div><Target /><span>Approved PUT<b>{entries.filter(event => event.action === "BUY PE").length}</b></span></div>
          <div><Activity /><span>Model checks<b>{callEvents.length + putEvents.length}</b></span></div>
          <footer>{modelBlocked ? <><ShieldCheck />Model scoring blocked; no synthetic signals are shown.</> : entries.length ? <><Check />Approved entries are marked on the NIFTY candle chart.</> :
            <><ChevronRight />Continue replay; future decisions remain hidden.</>}</footer>
        </aside>
      </section>
    </>}
  </>;
}

function DecisionCard({ side, event, premium, strike, fallbackThreshold }: {
  side: "CALL" | "PUT";
  event: any;
  premium?: number;
  strike?: number | null;
  fallbackThreshold?: number;
}) {
  const approved = event?.action === (side === "CALL" ? "BUY CE" : "BUY PE");
  return <article className={`replay-decision-card ${side.toLowerCase()} ${approved ? "approved" : ""}`}>
    <header><span>{side} DECISION</span><b>{approved ? "ENTRY APPROVED" : event ? "SETUP REJECTED" : "WAITING"}</b></header>
    <h3>{approved ? `BUY ${side}` : "NO ENTRY"}</h3>
    <p>{event ? readable(event.strategy) : `No ${side} setup has appeared yet.`}</p>
    <div><span>Contract<b>{strike ? `${strike.toFixed(0)} ${side === "CALL" ? "CE" : "PE"}` : `ATM ${side}`}</b></span>
      <span>Premium<b>INR {money(premium)}</b></span>
      <span>Probability<b>{percent(event?.probability)}</b></span>
      <span>Required<b>{percent(event?.threshold ?? fallbackThreshold)}</b></span></div>
    <footer><Activity />{event ? readable(event.reason) : "Waiting for a causal strategy event."}</footer>
  </article>;
}
