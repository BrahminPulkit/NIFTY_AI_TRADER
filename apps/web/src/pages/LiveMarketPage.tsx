import { Activity, CandlestickChart, Columns2, Eye, Grid2X2, Keyboard, Maximize2, Minimize2, PanelTop, Table2, WifiOff, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ErrorState, LoadingState } from "../components/ui/States";
import { PageHead } from "../components/ui/PageHead";
import { MarketChart } from "../features/charts/MarketChart";
import { OrderReviewModal } from "../features/paper/OrderReviewModal";
import { useManualPaperActions, useMarket, useOptions, usePaperAccount, usePaperActions } from "../hooks/useTradingData";
import { tradingApi } from "../services/api";
import type { MarketResponse, OptionResponse, PaperReview } from "../types/api";
import { money, readable, time } from "../utils/format";

type Instrument="NIFTY"|"CE"|"PE";
type Timeframe=1|3|5|15;
type ChartConfig={instrument:Instrument;timeframe:Timeframe;securityId?:string;label?:string};
type IndicatorState={ema9:boolean;ema20:boolean;vwap:boolean;volume:boolean;rsi:boolean;macd:boolean;levels:boolean};
const DEFAULT_CHARTS:ChartConfig[]=[{instrument:"NIFTY",timeframe:1},{instrument:"CE",timeframe:1},{instrument:"PE",timeframe:1}];

function aggregate(rows:MarketResponse["candles"],minutes:Timeframe){
  if(minutes===1)return rows;
  const buckets=new Map<number,MarketResponse["candles"]>();
  rows.forEach(row=>{const stamp=Date.parse(String(row.timestamp));const key=Math.floor(stamp/(minutes*60000));const group=buckets.get(key)||[];group.push(row);buckets.set(key,group)});
  return [...buckets.values()].map(group=>({timestamp:group[0].timestamp,open:group[0].open,high:Math.max(...group.map(row=>row.high)),low:Math.min(...group.map(row=>row.low)),close:group.at(-1)!.close,volume:group.reduce((total,row)=>total+(row.volume||0),0)}));
}

function instrumentRows(data:MarketResponse,config:ChartConfig){return config.securityId?data.watched_candles[config.securityId]||[]:config.instrument==="NIFTY"?data.candles:data.option_candles[config.instrument]}
function instrumentLabel(data:MarketResponse,config:ChartConfig){if(config.label)return config.label;const meta=data.instruments[config.instrument]||{};const strike=Number(meta.strike||0);return config.instrument==="NIFTY"?"NIFTY 50":`NIFTY ${strike?money(strike):"ATM"} ${config.instrument}`}

export function LiveMarketPage(){
  const market=useMarket();
  const options=useOptions();
  const manual=useManualPaperActions();
  const paperActions=usePaperActions("live");
  const account=usePaperAccount("live");
  const [layout,setLayout]=useState<1|2|3>(()=>{const value=Number(localStorage.getItem("live-chart-layout"));return value===2||value===3?value:1});
  const [charts,setCharts]=useState<ChartConfig[]>(()=>{try{const value=JSON.parse(localStorage.getItem("live-chart-configs")||"");return Array.isArray(value)&&value.length===3?value:DEFAULT_CHARTS}catch{return DEFAULT_CHARTS}});
  const [maximized,setMaximized]=useState<number|null>(null);
  const [activeSlot,setActiveSlot]=useState(0);
  const [chainOpen,setChainOpen]=useState(false);
  const [indicators,setIndicators]=useState<IndicatorState>(()=>{try{return JSON.parse(localStorage.getItem("live-chart-indicators-v3")||"") as IndicatorState}catch{return {ema9:false,ema20:false,vwap:false,volume:true,rsi:false,macd:false,levels:false}}});
  const [hotkeys,setHotkeys]=useState(false);
  const [review,setReview]=useState<PaperReview|null>(null);
  const [orderSecurityId,setOrderSecurityId]=useState("");
  const [chainMessage,setChainMessage]=useState("");
  useEffect(()=>localStorage.setItem("live-chart-indicators-v3",JSON.stringify(indicators)),[indicators]);
  useEffect(()=>localStorage.setItem("live-chart-layout",String(layout)),[layout]);
  useEffect(()=>localStorage.setItem("live-chart-configs",JSON.stringify(charts.map(({instrument,timeframe})=>({instrument,timeframe})))),[charts]);
  const activeIndex=maximized??Math.min(activeSlot,layout-1);
  const activeConfig=charts[activeIndex];
  const activeSecurityId=activeConfig.securityId||String(market.data?.instruments[activeConfig.instrument]?.security_id||"");
  useEffect(()=>{if(!hotkeys)return;const handler=(event:KeyboardEvent)=>{if(event.repeat||["INPUT","SELECT","TEXTAREA"].includes((event.target as HTMLElement)?.tagName))return;if(event.key.toLowerCase()==="o")setChainOpen(true);if(event.key.toLowerCase()==="b"&&activeSecurityId&&!account.data?.position){setOrderSecurityId(activeSecurityId);manual.review.mutate({securityId:activeSecurityId,quantity:1},{onSuccess:setReview})}if(["1","2","3"].includes(event.key)){setLayout(Number(event.key) as 1|2|3);setMaximized(null)}};window.addEventListener("keydown",handler);return()=>window.removeEventListener("keydown",handler)},[hotkeys,activeSecurityId,account.data?.position]);
  if(market.isLoading||options.isLoading)return <LoadingState/>;
  if(market.error||options.error)return <ErrorState message={(market.error||options.error)!.message}/>;
  const data=market.data!;
  const activeRows=instrumentRows(data,activeConfig);
  const activeQuote=options.data?.rows.find(row=>String(row.security_id||"")===activeSecurityId);
  const visible=maximized==null?charts.slice(0,layout):[charts[maximized]];
  const updateChart=(index:number,patch:Partial<ChartConfig>)=>setCharts(value=>value.map((chart,i)=>i===index?{...chart,...patch}:chart));
  const toggle=(key:keyof typeof indicators)=>setIndicators(value=>({...value,[key]:!value[key]}));
  const loadContract=async(row:OptionRow)=>{if(row.security_id==null||!row.option_type||row.strike==null)return;const securityId=String(row.security_id);setChainMessage(`Loading ${money(row.strike)} ${row.option_type} chart...`);try{await tradingApi.watchOption(securityId);updateChart(activeIndex,{instrument:row.option_type as Instrument,timeframe:1,securityId,label:`NIFTY ${money(row.strike)} ${row.option_type}`});setChainOpen(false);setChainMessage("");window.setTimeout(()=>market.refetch(),800)}catch(error){setChainMessage(error instanceof Error?error.message:"Unable to load this contract")}};
  const reviewOrder=(securityId:string)=>{setOrderSecurityId(securityId);manual.review.mutate({securityId,quantity:1},{onSuccess:setReview})};
  const confirmOrder=()=>manual.open.mutate({securityId:orderSecurityId,quantity:1},{onSuccess:()=>setReview(null)});
  return <div className="live-terminal-page">
    <PageHead eyebrow="MARKET TERMINAL" title="Live Market" detail="Configure one, two or three live charts from the shared Dhan session." side={<button className="open-chain" onClick={()=>setChainOpen(true)}><Table2/>Option Chain</button>}/>
    {!data.market.connected&&<div className="feed-alert"><WifiOff/><div><strong>LIVE FEED DISCONNECTED</strong><span>Connect Dhan from Settings. This workspace never authenticates independently.</span></div></div>}
    <section className="terminal-toolbar">
      <div className="layout-switch"><span>LAYOUT</span><button title="One chart" className={layout===1?"active":""} onClick={()=>{setLayout(1);setMaximized(null)}}><PanelTop/></button><button title="Two charts" className={layout===2?"active":""} onClick={()=>{setLayout(2);setMaximized(null)}}><Columns2/></button><button title="Three charts" className={layout===3?"active":""} onClick={()=>{setLayout(3);setMaximized(null)}}><Grid2X2/></button></div>
      <div className="terminal-signal"><i className={data.market.signal?.action&&data.market.signal.action!=="NO TRADE"?"ready":""}/><span>CALL MODEL<b>{data.market.signal?.action||"WAITING"}</b></span><span>PUT PAPER WATCH<b>{data.market.put_signal?.action||"WAITING"}</b></span><span>STATUS<b>{readable(data.market.put_signal_status)}</b></span></div>
      <div className="terminal-trade-command"><button className="sell" disabled={!account.data?.position} onClick={()=>paperActions.exit.mutate(undefined)}><small>EXIT</small><b>{money(account.data?.position?.current_premium)}</b></button><button className="buy" disabled={!activeSecurityId||Boolean(account.data?.position)} onClick={()=>activeSecurityId&&reviewOrder(activeSecurityId)}><small>PAPER BUY</small><b>{money(activeQuote?.ask||activeQuote?.ltp||activeRows.at(-1)?.close)}</b></button></div>
      <button className={`hotkey-toggle ${hotkeys?"active":""}`} title="Enable keyboard controls: B buy review, O option chain, 1-3 layouts" onClick={()=>setHotkeys(value=>!value)}><Keyboard/>HOTKEYS</button>
      <div className="global-indicators"><span>INDICATORS</span>{([["ema9","EMA 9"],["ema20","EMA 20"],["vwap","VWAP"],["rsi","RSI"],["macd","MACD"],["levels","LEVELS"],["volume","VOL"]] as [keyof typeof indicators,string][]).map(([key,name])=><button className={indicators[key]?"active":""} onClick={()=>toggle(key)} key={key}><Eye/>{name}</button>)}</div>
    </section>
    <section className="scalp-context"><span>ACTIVE CONTRACT<b>{instrumentLabel(data,activeConfig)}</b></span><span>BID<b>{money(activeQuote?.bid)}</b></span><span>ASK<b>{money(activeQuote?.ask)}</b></span><span>SPREAD<b>{money(activeQuote?.spread)}</b></span><span>OPEN INTEREST<b>{money(activeQuote?.oi)}</b></span><span>VOLUME<b>{money(activeQuote?.volume)}</b></span><span>DATA<b>{time(options.data?.updated_at)} IST</b></span></section>
    <section className={`multi-chart-grid layout-${maximized==null?layout:1}`}>{visible.map((config,visibleIndex)=>{const actualIndex=maximized??visibleIndex;const currentPosition=account.data?.position;const tradeSecurityId=config.securityId||String(data.instruments[config.instrument]?.security_id||"");const quote=options.data?.rows.find(row=>String(row.security_id||"")===tradeSecurityId);let chartPosition:undefined|{entry:number;stop:number;target:number};if(tradeSecurityId&&currentPosition&&String(currentPosition.security_id)===tradeSecurityId)chartPosition={entry:currentPosition.entry_premium,stop:currentPosition.stop_loss,target:currentPosition.target};return <ChartTile key={actualIndex} data={data} config={config} tradeSecurityId={tradeSecurityId} livePrice={quote?.ltp} indicators={indicators} signal={config.instrument==="NIFTY"?data.market.signal:null} position={chartPosition} onChange={patch=>updateChart(actualIndex,patch)} onMaximize={()=>{setActiveSlot(actualIndex);setMaximized(maximized==null?actualIndex:null)}} maximized={maximized!=null} onBuy={reviewOrder} active={actualIndex===activeIndex} onActivate={()=>setActiveSlot(actualIndex)}/>})}</section>
    <section className="account-manager"><header><b>ACCOUNT MANAGER</b><span>PAPER TRADING</span></header><div><span>Available capital<b>INR {money(account.data?.capital)}</b></span><span>Open position<b>{account.data?.position?`${account.data.position.option_type} ${money(account.data.position.strike)}`:"NONE"}</b></span><span>Live P&amp;L<b className={(account.data?.position?.current_pnl||0)>=0?"profit":"loss"}>INR {money(account.data?.position?.current_pnl||0)}</b></span><span>Trades today<b>{account.data?.trades.length||0}</b></span><span>Active chart<b>{instrumentLabel(data,activeConfig)}</b></span></div></section>
    {chainOpen&&<OptionChainDrawer data={options.data!} spot={data.market.quotes.NIFTY} close={()=>{setChainOpen(false);setChainMessage("")}} select={loadContract} message={chainMessage}/>} 
    {review&&<OrderReviewModal review={review} busy={manual.open.isPending} error={manual.open.error?.message} onConfirm={confirmOrder} onClose={()=>setReview(null)}/>} 
  </div>;
}

function ChartTile({data,config,tradeSecurityId,livePrice,indicators,signal,position,onChange,onMaximize,maximized,onBuy,active,onActivate}:{data:MarketResponse;config:ChartConfig;tradeSecurityId:string;livePrice?:number|null;indicators:IndicatorState;signal:MarketResponse["market"]["signal"];position?:{entry:number;stop:number;target:number};onChange:(patch:Partial<ChartConfig>)=>void;onMaximize:()=>void;maximized:boolean;onBuy:(securityId:string)=>void;active:boolean;onActivate:()=>void}){
  const rows=aggregate(instrumentRows(data,config),config.timeframe);
  const label=instrumentLabel(data,config);
  return <article className={`terminal-chart ${active?"active":""}`} onClick={onActivate}>
    <header><div className="chart-symbol"><select aria-label="Chart instrument" value={config.instrument} onChange={event=>onChange({instrument:event.target.value as Instrument,securityId:undefined,label:undefined})}><option value="NIFTY">NIFTY 50</option><option value="CE">{config.securityId&&config.instrument==="CE"?label:"ATM CALL"}</option><option value="PE">{config.securityId&&config.instrument==="PE"?label:"ATM PUT"}</option></select><span>{label} | NSE</span></div><strong>{money(livePrice||rows.at(-1)?.close)}</strong>{tradeSecurityId&&<button className="quick-buy" onClick={()=>onBuy(tradeSecurityId)}>PAPER BUY</button>}<div className="chart-timeframes">{([1,3,5,15] as Timeframe[]).map(value=><button className={config.timeframe===value?"active":""} onClick={()=>onChange({timeframe:value})} key={value}>{value}m</button>)}</div><button className="chart-maximize" title={maximized?"Restore layout":"Maximize chart"} onClick={onMaximize}>{maximized?<Minimize2/>:<Maximize2/>}</button></header>
    {rows.length?<MarketChart rows={rows} indicators={indicators} signal={signal} position={position} storageKey={config.securityId||config.instrument}/>:<div className="chart-empty"><CandlestickChart/><h3>{config.instrument} chart waiting</h3><p>Waiting for completed candles in the shared broker cache.</p></div>}
  </article>;
}

type OptionRow=OptionResponse["rows"][number];
function OptionChainDrawer({data,spot,close,select,message}:{data:OptionResponse;spot:number|undefined;close:()=>void;select:(row:OptionRow)=>void;message:string}){
  const ladder=useMemo(()=>{const grouped=new Map<number,{strike:number;ce?:OptionRow;pe?:OptionRow}>();for(const row of data.rows){if(row.strike==null)continue;const item=grouped.get(row.strike)||{strike:row.strike};if(row.option_type==="CE")item.ce=row;if(row.option_type==="PE")item.pe=row;grouped.set(row.strike,item)}const all=[...grouped.values()].sort((a,b)=>a.strike-b.strike);if(!spot)return all.slice(0,21);const nearest=all.reduce((best,row,index)=>Math.abs(row.strike-spot)<Math.abs(all[best].strike-spot)?index:best,0);return all.slice(Math.max(0,nearest-10),nearest+11)},[data.rows,spot]);
  return <div className="chain-drawer-backdrop" onClick={close}><aside className="chain-drawer" onClick={event=>event.stopPropagation()}><header><div><span>LIVE DERIVATIVES</span><h2>NIFTY Option Chain</h2><small>Spot {money(spot)} | {time(data.updated_at)} IST</small></div><button onClick={close}><X/></button></header>{message&&<div className="chain-message"><Activity/>{message}</div>}<div className="drawer-chain-head"><b>CALL LTP</b><span>STRIKE</span><b>PUT LTP</b></div><div className="drawer-chain-body">{ladder.map(row=>{const isAtm=Boolean(spot&&Math.abs(row.strike-spot)<=25);return <div className={isAtm?"atm":""} key={row.strike}><button disabled={!row.ce?.security_id||Boolean(message)} title="Open Call chart" onClick={()=>row.ce&&select(row.ce)}><strong>{money(row.ce?.ltp)}</strong><small>OI {money(row.ce?.oi)}</small></button><span>{money(row.strike)}{isAtm?<small>ATM</small>:null}</span><button disabled={!row.pe?.security_id||Boolean(message)} title="Open Put chart" onClick={()=>row.pe&&select(row.pe)}><strong>{money(row.pe?.ltp)}</strong><small>OI {money(row.pe?.oi)}</small></button></div>})}</div><footer>Click any available Call or Put quote to load its chart into the active slot.</footer></aside></div>;
}
