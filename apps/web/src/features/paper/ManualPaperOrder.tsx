import { useMemo, useState } from "react";
import { CircleDollarSign } from "lucide-react";
import { useManualPaperActions, useOptions } from "../../hooks/useTradingData";
import type { PaperReview } from "../../types/api";
import { money } from "../../utils/format";
import { OrderReviewModal } from "./OrderReviewModal";

export function ManualPaperOrder({positionOpen}:{positionOpen:boolean}) {
  const options = useOptions();
  const actions = useManualPaperActions();
  const [type, setType] = useState<"CE"|"PE">("CE");
  const [securityId, setSecurityId] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [review, setReview] = useState<PaperReview|null>(null);
  const rows = useMemo(() => (options.data?.rows || []).filter(
    row => row.option_type === type && row.security_id != null), [options.data, type]);
  const selectedId = securityId && rows.some(row => String(row.security_id) === securityId)
    ? securityId : String(rows[0]?.security_id || "");
  const selected = rows.find(row => String(row.security_id) === selectedId);
  const begin = () => selectedId && actions.review.mutate(
    {securityId:selectedId,quantity}, {onSuccess:setReview});
  const confirm = () => actions.open.mutate(
    {securityId:selectedId,quantity}, {onSuccess:()=>setReview(null)});

  return <section className="panel manual-paper-order">
    <header><div><span>MANUAL LIVE OPTION</span><h3>Select a CE or PE contract</h3></div><CircleDollarSign/></header>
    <div className="segments manual-type">
      <button className={type==="CE"?"active":""} onClick={()=>{setType("CE");setSecurityId("")}}>CALL (CE)</button>
      <button className={type==="PE"?"active":""} onClick={()=>{setType("PE");setSecurityId("")}}>PUT (PE)</button>
    </div>
    <label>Strike and expiry<select value={selectedId} onChange={event=>setSecurityId(event.target.value)}>
      {rows.map(row=><option key={String(row.security_id)} value={String(row.security_id)}>
        {money(row.strike)} {row.option_type} · {row.expiry} · LTP ₹{money(row.ltp)}
      </option>)}
    </select></label>
    <label>Virtual quantity<input type="number" min="1" max="10000" value={quantity}
      onChange={event=>setQuantity(Math.max(1,Number(event.target.value)||1))}/></label>
    <div className="manual-quote-grid">
      <span>Live bid<b>₹{money(selected?.bid)}</b></span>
      <span>Live ask<b>₹{money(selected?.ask)}</b></span>
      <span>Spread<b>₹{money(selected?.spread)}</b></span>
      <span>Open interest<b>{money(selected?.oi)}</b></span>
    </div>
    <p className="muted-copy">Entry uses live ask; P&amp;L and exit use live bid. No broker order is sent.</p>
    <button className="primary-action" disabled={positionOpen||!selected||actions.review.isPending} onClick={begin}>
      {actions.review.isPending?"Checking live quote...":"Review manual paper order"}
    </button>
    {actions.review.error&&<p className="form-message">{actions.review.error.message}</p>}
    {review&&<OrderReviewModal review={review} busy={actions.open.isPending}
      error={actions.open.error?.message} onConfirm={confirm} onClose={()=>setReview(null)}/>} 
  </section>;
}
