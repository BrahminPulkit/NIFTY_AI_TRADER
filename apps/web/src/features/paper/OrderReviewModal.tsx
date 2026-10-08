import { ShieldCheck, WalletCards, X } from "lucide-react";
import type { PaperReview } from "../../types/api";
import { money, percent, readable } from "../../utils/format";

export function OrderReviewModal({review,busy,error,onConfirm,onClose}:{
  review:PaperReview;busy:boolean;error?:string;
  onConfirm:()=>void;onClose:()=>void;
}) {
  const contract = review.option_type
    ? `NIFTY ${money(review.strike)} ${review.option_type}`
    : "NIFTY ATM CALL";
  return <div className="modal-backdrop" role="dialog" aria-modal="true"
    aria-label="Review paper order"><section className="order-modal">
    <header><div><span>PAPER ORDER REVIEW</span><h2>{contract}</h2></div>
      <button onClick={onClose} aria-label="Close review"><X/></button></header>
    <div className="paper-only-banner"><ShieldCheck/><span>Virtual order only
      <b>No broker order will be sent</b></span></div>
    <div className="review-grid">
      <span>Strategy<b>{readable(review.strategy)}</b></span>
      <span>{review.option_type?"Expiry":"Confidence"}<b>{review.option_type?review.expiry:percent(review.probability)}</b></span>
      <span>Entry premium<b>₹{money(review.entry_premium)}</b></span>
      <span>Quantity<b>{review.quantity}</b></span>
      <span>Stop loss<b>₹{money(review.stop_loss)}</b></span>
      <span>Target<b>₹{money(review.target)}</b></span>
      <span>Capital required<b>₹{money(review.capital_required)}</b></span>
      <span>Maximum loss<b>₹{money(review.maximum_loss)}</b></span>
    </div>
    <footer><button className="secondary" onClick={onClose}>Cancel</button>
      <button disabled={!review.eligible||busy} onClick={onConfirm}><WalletCards/>
        {busy?"Opening...":"Confirm paper entry"}</button></footer>
    {!review.eligible&&<p className="form-message">Blocked: {readable(review.reason)}</p>}
    {error&&<p className="form-message">{error}</p>}
  </section></div>;
}
