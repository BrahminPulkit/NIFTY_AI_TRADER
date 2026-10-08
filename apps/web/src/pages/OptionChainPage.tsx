import { Fragment, useMemo } from "react";
import { CircleDollarSign } from "lucide-react";
import { ErrorState, LoadingState } from "../components/ui/States";
import { PageHead } from "../components/ui/PageHead";
import { useMarket, useOptions } from "../hooks/useTradingData";
import type { OptionResponse } from "../types/api";
import { money, time } from "../utils/format";

type OptionRow = OptionResponse["rows"][number];
type ChainRow = {strike:number;ce?:OptionRow;pe?:OptionRow};

export function OptionChainPage() {
  const options = useOptions();
  const market = useMarket();
  const data = options.data;
  const ladder = useMemo<ChainRow[]>(() => {
    const grouped = new Map<number,ChainRow>();
    for (const row of data?.rows || []) {
      if (row.strike == null) continue;
      const item = grouped.get(row.strike) || {strike:row.strike};
      if (row.option_type === "CE") item.ce = row;
      if (row.option_type === "PE") item.pe = row;
      grouped.set(row.strike,item);
    }
    return [...grouped.values()].sort((a,b)=>a.strike-b.strike);
  },[data]);
  if (options.isLoading || market.isLoading) return <LoadingState/>;
  if (options.error) return <ErrorState message={options.error.message}/>;
  if (market.error) return <ErrorState message={market.error.message}/>;
  const spot = market.data?.market.quotes.NIFTY;
  const atm = ladder.length && spot != null
    ? ladder.reduce((best,row)=>Math.abs(row.strike-spot)<Math.abs(best.strike-spot)?row:best).strike
    : undefined;
  const strongestCall = ladder.reduce((best,row)=>(row.ce?.oi||0)>(best.ce?.oi||0)?row:best,ladder[0]);
  const strongestPut = ladder.reduce((best,row)=>(row.pe?.oi||0)>(best.pe?.oi||0)?row:best,ladder[0]);
  const callOi = ladder.reduce((sum,row)=>sum+(row.ce?.oi||0),0);
  const putOi = ladder.reduce((sum,row)=>sum+(row.pe?.oi||0),0);
  const pcr = callOi ? putOi/callOi : null;
  const expiry = ladder.find(row=>row.ce?.expiry||row.pe?.expiry)?.ce?.expiry
    || ladder.find(row=>row.pe?.expiry)?.pe?.expiry;

  return <>
    <PageHead eyebrow="DERIVATIVES" title="Option Chain"
      detail="Broker-style paired NIFTY option ladder from the shared live cache."
      side={<div className={`live-chip ${data?.connected?"online":""}`}><i/>{data?.status}</div>}/>
    {!ladder.length?<div className="empty"><CircleDollarSign/><h3>Option chain waiting</h3>
      <p>Connect Dhan and allow the background worker to resolve the current expiry.</p></div>
    :<section className="table-panel broker-chain">
      <div className="chain-market-bar">
        <div><span>CALL WALL</span><b>{money(strongestCall?.strike)} CE</b><small>Highest call OI</small></div>
        <div className="chain-spot"><span>NIFTY LIVE</span><strong>{money(spot)}</strong><small>ATM {money(atm)} · {time(data?.updated_at)} IST</small></div>
        <div><span>PUT WALL</span><b>{money(strongestPut?.strike)} PE</b><small>Highest put OI</small></div>
      </div>
      <div className="option-summary">
        <span>Expiry<b>{expiry||"--"}</b></span><span>Put/Call OI<b>{pcr?.toFixed(2)||"--"}</b></span>
        <span>Call OI<b>{money(callOi)}</b></span><span>Put OI<b>{money(putOi)}</b></span>
      </div>
      <div className="table-scroll chain-scroll"><table className="paired-chain">
        <thead><tr className="chain-groups"><th colSpan={5}>CALLS (CE)</th><th className="strike-head">STRIKE</th><th colSpan={5}>PUTS (PE)</th></tr>
          <tr><th>OI</th><th>Volume</th><th>Bid</th><th>Ask</th><th>LTP</th><th className="strike-head">Live strike</th><th>LTP</th><th>Bid</th><th>Ask</th><th>Volume</th><th>OI</th></tr></thead>
        <tbody>{ladder.map((row,index)=>{
          const next=ladder[index+1];
          const showSpot=spot!=null&&next!=null&&row.strike<=spot&&spot<next.strike;
          return <Fragment key={row.strike}><tr className={row.strike===atm?"atm-row":""}>
            <Cell value={row.ce?.oi} className={row.strike===strongestCall?.strike?"writer-call":""}/>
            <Cell value={row.ce?.volume}/><Cell value={row.ce?.bid}/><Cell value={row.ce?.ask}/>
            <Cell value={row.ce?.ltp} className="ce-ltp"/>
            <td className="strike-cell"><strong>{money(row.strike)}</strong>{row.strike===atm&&<small>ATM</small>}</td>
            <Cell value={row.pe?.ltp} className="pe-ltp"/><Cell value={row.pe?.bid}/><Cell value={row.pe?.ask}/>
            <Cell value={row.pe?.volume}/><Cell value={row.pe?.oi} className={row.strike===strongestPut?.strike?"writer-put":""}/>
          </tr>{showSpot&&<tr className="spot-marker-row">
            <td colSpan={5}><span/></td><td><b>LIVE NIFTY {money(spot)}</b></td><td colSpan={5}><span/></td>
          </tr>}</Fragment>})}</tbody>
      </table></div>
      <footer className="chain-legend"><span><i className="atm-key"/>ATM strike</span><span><i className="call-key"/>Highest Call OI</span><span><i className="put-key"/>Highest Put OI</span></footer>
    </section>}
  </>;
}

function Cell({value,className=""}:{value:number|null|undefined;className?:string}) {
  return <td className={className}>{money(value)}</td>;
}
