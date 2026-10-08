import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, HistogramSeries,
  LineSeries, type CandlestickData, type HistogramData, type IChartApi, type IPriceLine, type ISeriesApi, type LineData, type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { Eraser, Expand, Focus, Minus, ScanLine } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { MarketResponse } from "../../types/api";

const IST_OFFSET_SECONDS = 5.5 * 60 * 60;

function chartTime(value: unknown): UTCTimestamp {
  const parsed = typeof value === "number"
    ? value / 1000
    : Date.parse(String(value)) / 1000;
  return Math.floor(parsed + IST_OFFSET_SECONDS) as UTCTimestamp;
}

type Indicators = { ema9:boolean; ema20:boolean; vwap:boolean; volume:boolean;rsi:boolean;macd:boolean;levels:boolean };
type PositionLines={entry:number;stop:number;target:number};
type MarkerSignal={timestamp:string;action:string;strategy:string;strike?:number|null;entry_premium?:number;markers?:MarkerSignal[]};
type ReplayOptionRow={close:number;option_close?:number;put_option_close?:number;call_strike?:number|null;put_strike?:number|null};

function ema(values:number[], period:number){
  const alpha=2/(period+1);let current=values[0]||0;
  return values.map((value,index)=>current=index?value*alpha+current*(1-alpha):value);
}
function rsi(values:number[],period=14){let gain=0,loss=0;return values.map((value,index)=>{if(!index)return 50;const move=value-values[index-1];gain=(gain*(period-1)+Math.max(move,0))/period;loss=(loss*(period-1)+Math.max(-move,0))/period;return loss?100-100/(1+gain/loss):100})}

export function MarketChart({ rows, indicators, signal, signals, position, storageKey="default" }: { rows: MarketResponse["candles"]; indicators?:Indicators;signal?:MarkerSignal|null;signals?:MarkerSignal[];position?:PositionLines;storageKey?:string }) {
  const wrapper = useRef<HTMLDivElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const chartRef=useRef<IChartApi|null>(null);
  const candleRef=useRef<ISeriesApi<"Candlestick">|null>(null);
  const priceLines=useRef<IPriceLine[]>([]);
  const drawingRef=useRef(false);
  const [drawing,setDrawing]=useState(false);
  useEffect(()=>{drawingRef.current=drawing},[drawing]);

  useEffect(() => {
    if (!host.current || !rows.length) return;
    const chart: IChartApi = createChart(host.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#ffffff" },
        textColor: "#64748b",
        fontFamily: "Inter, system-ui",
      },
      grid: {
        vertLines: { color: "#eef2f7" },
        horzLines: { color: "#eef2f7" },
      },
      crosshair: {
        vertLine: { color: "#2563eb" },
        horzLine: { color: "#2563eb" },
      },
      rightPriceScale: { borderColor: "#dce4ee" },
      timeScale: {
        borderColor: "#dce4ee", timeVisible: true, secondsVisible: false,
      },
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#168a52", downColor: "#dc3545",
      wickUpColor: "#168a52", wickDownColor: "#dc3545",
      borderVisible: false,
    });
    chartRef.current=chart;
    candleRef.current=candles;
    const unique = new Map<number, MarketResponse["candles"][number]>();
    rows.forEach((row) => unique.set(chartTime(row.timestamp), row));
    const ordered = [...unique.entries()].sort(([a], [b]) => a - b);
    candles.setData(ordered.map(([timestamp, row]) => ({
      time: timestamp as Time, open: row.open, high: row.high,
      low: row.low, close: row.close,
    } as CandlestickData)));
    const markerSignals=signals?.length?signals:(signal?.markers?.length?signal.markers:(signal?[signal]:[]));
    if(markerSignals.length&&ordered.length){
      const markers=markerSignals.map(item=>{
        const wanted=chartTime(item.timestamp);
        const nearest=ordered.reduce((best,row)=>Math.abs(row[0]-wanted)<Math.abs(best[0]-wanted)?row:best,ordered[0]);
        const trade=item.action!=="NO TRADE";const isPut=item.action==="BUY PE";
        const contract=item.strike?`${item.strike.toFixed(0)} ${isPut?"PE":"CE"}`:item.action;
        const premium=item.entry_premium!=null?` @ INR ${item.entry_premium.toFixed(1)}`:"";
        return {time:nearest[0] as Time,position:trade?(isPut?"aboveBar":"belowBar"):"aboveBar",shape:trade?(isPut?"arrowDown":"arrowUp"):"circle",color:trade?(isPut?"#7c3aed":"#168a52"):"#d97706",text:trade?`${contract}${premium}`:"AI: NO TRADE"} as const;
      }).sort((a,b)=>(a.time as number)-(b.time as number));
      createSeriesMarkers(candles,markers);
    }
    if(position){
      candles.createPriceLine({price:position.entry,color:"#2563eb",lineWidth:2,lineStyle:2,axisLabelVisible:true,title:"ENTRY"});
      candles.createPriceLine({price:position.stop,color:"#dc3545",lineWidth:2,lineStyle:2,axisLabelVisible:true,title:"STOP"});
      candles.createPriceLine({price:position.target,color:"#168a52",lineWidth:2,lineStyle:2,axisLabelVisible:true,title:"TARGET"});
    }
    if(indicators?.volume!==false){
      const volume = chart.addSeries(HistogramSeries, {priceFormat:{type:"volume"},priceScaleId:"volume",color:"rgba(37,99,235,.25)"});
      chart.priceScale("volume").applyOptions({scaleMargins:{top:.78,bottom:0}});
      volume.setData(ordered.map(([timestamp,row])=>({time:timestamp as Time,value:row.volume||0,color:row.close>=row.open?"rgba(22,138,82,.24)":"rgba(220,53,69,.22)"} as HistogramData)));
    }
    const closes=ordered.map(([,row])=>row.close);
    const addLine=(values:number[],color:string,title:string)=>{
      const series=chart.addSeries(LineSeries,{color,lineWidth:2,title,priceLineVisible:false,lastValueVisible:false});
      series.setData(ordered.map(([timestamp],index)=>({time:timestamp as Time,value:values[index]} as LineData)));
    };
    if(indicators?.ema9) addLine(ema(closes,9),"#2563eb","EMA 9");
    if(indicators?.ema20) addLine(ema(closes,20),"#9b7bff","EMA 20");
    if(indicators?.vwap){
      let cumulativeValue=0,cumulativeVolume=0;
      const values=ordered.map(([,row])=>{const volume=row.volume||0;cumulativeValue+=((row.high+row.low+row.close)/3)*volume;cumulativeVolume+=volume;return cumulativeVolume?cumulativeValue/cumulativeVolume:row.close});
      addLine(values,"#d97706","VWAP");
    }
    if(indicators?.levels){
      const dates=ordered.map(([,row])=>new Date(String(row.timestamp)).toLocaleDateString("en-CA",{timeZone:"Asia/Kolkata"}));
      const latest=dates.at(-1);const previous=[...new Set(dates.filter(value=>value!==latest))].at(-1);
      const prior=ordered.filter((_,index)=>dates[index]===previous).map(([,row])=>row);
      if(prior.length){const high=Math.max(...prior.map(row=>row.high)),low=Math.min(...prior.map(row=>row.low)),close=prior.at(-1)!.close,pivot=(high+low+close)/3;[[high,"#16a34a","PDH"],[low,"#dc3545","PDL"],[pivot,"#d97706","PIVOT"]].forEach(([price,color,title])=>candles.createPriceLine({price:price as number,color:color as string,lineWidth:1,lineStyle:2,axisLabelVisible:true,title:title as string}))}
    }
    let pane=1;
    if(indicators?.rsi){const values=rsi(closes);const series=chart.addSeries(LineSeries,{color:"#7c3aed",lineWidth:2,title:"RSI 14",priceLineVisible:false,lastValueVisible:true},pane++);series.setData(ordered.map(([timestamp],index)=>({time:timestamp as Time,value:values[index]})));series.createPriceLine({price:70,color:"#dc3545",lineWidth:1,lineStyle:2,axisLabelVisible:false,title:"70"});series.createPriceLine({price:30,color:"#168a52",lineWidth:1,lineStyle:2,axisLabelVisible:false,title:"30"})}
    if(indicators?.macd){const fast=ema(closes,12),slow=ema(closes,26),macd=fast.map((value,index)=>value-slow[index]),signalLine=ema(macd,9),hist=macd.map((value,index)=>value-signalLine[index]);const macdSeries=chart.addSeries(LineSeries,{color:"#2563eb",lineWidth:2,title:"MACD",priceLineVisible:false,lastValueVisible:false},pane);const signalSeries=chart.addSeries(LineSeries,{color:"#d97706",lineWidth:1,title:"Signal",priceLineVisible:false,lastValueVisible:false},pane);const histogram=chart.addSeries(HistogramSeries,{priceLineVisible:false,lastValueVisible:false},pane);macdSeries.setData(ordered.map(([timestamp],index)=>({time:timestamp as Time,value:macd[index]})));signalSeries.setData(ordered.map(([timestamp],index)=>({time:timestamp as Time,value:signalLine[index]})));histogram.setData(ordered.map(([timestamp],index)=>({time:timestamp as Time,value:hist[index],color:hist[index]>=0?"rgba(22,138,82,.35)":"rgba(220,53,69,.3)"})))}
    if(pane>1){const panes=chart.panes();panes[0]?.setStretchFactor(4);for(let index=1;index<panes.length;index++)panes[index].setStretchFactor(1)}
    const saved=JSON.parse(localStorage.getItem(`chart-lines:${storageKey}`)||"[]") as number[];
    priceLines.current=saved.map(price=>candles.createPriceLine({price,color:"#16a34a",lineWidth:2,lineStyle:0,axisLabelVisible:true,title:"LEVEL"}));
    chart.subscribeClick(param=>{if(!drawingRef.current||!param.point)return;const price=candles.coordinateToPrice(param.point.y);if(price==null)return;priceLines.current.push(candles.createPriceLine({price,color:"#16a34a",lineWidth:2,lineStyle:0,axisLabelVisible:true,title:"LEVEL"}));const values=priceLines.current.map(line=>line.options().price);localStorage.setItem(`chart-lines:${storageKey}`,JSON.stringify(values));setDrawing(false)});
    chart.timeScale().fitContent();
    return () => {chartRef.current=null;candleRef.current=null;priceLines.current=[];chart.remove()};
  }, [rows, indicators, signal, signals, position, storageKey]);

  const fit=()=>chartRef.current?.timeScale().fitContent();
  const reset=()=>{chartRef.current?.timeScale().resetTimeScale();chartRef.current?.priceScale("right").applyOptions({autoScale:true})};
  const clear=()=>{const series=candleRef.current;if(series)priceLines.current.forEach(line=>series.removePriceLine(line));priceLines.current=[];localStorage.removeItem(`chart-lines:${storageKey}`)};
  const fullscreen=()=>wrapper.current?.requestFullscreen();
  const latest=rows.at(-1) as ReplayOptionRow|undefined;const replayOptions=latest?.option_close!=null;const entries=signals?.length?signals:(signal?.markers||[]);const latestCall=[...entries].reverse().find(item=>item.action==="BUY CE");const latestPut=[...entries].reverse().find(item=>item.action==="BUY PE");
  return <div className={`chart-stage ${drawing?"drawing":""} ${replayOptions?"replay-options":""}`} ref={wrapper}><div className="chart-actions"><button title="Fit all candles" onClick={fit}><Focus/></button><button title="Reset chart scale" onClick={reset}><ScanLine/></button><button title="Add horizontal level, then click chart" className={drawing?"active":""} onClick={()=>setDrawing(value=>!value)}><Minus/></button><button title="Clear saved levels" onClick={clear}><Eraser/></button><button title="Fullscreen chart" onClick={fullscreen}><Expand/></button></div><div className="lightweight-chart" ref={host}/>{replayOptions&&<div className="replay-option-strip"><div className={latestCall?"approved":""}><span>ATM CALL</span><b>{latest?.call_strike?`${latest.call_strike.toFixed(0)} CE`:"ATM CE"}</b><strong>INR {latest?.option_close?.toFixed(2)}</strong><small>{latestCall?`BUY CALL | ${latestCall.strategy}`:"NO CALL ENTRY"}</small></div><div className="index"><span>NIFTY</span><b>{latest?.close.toFixed(2)}</b><small>Replay candle</small></div><div className={`put ${latestPut?"approved":""}`}><span>ATM PUT</span><b>{latest?.put_strike?`${latest.put_strike.toFixed(0)} PE`:"ATM PE"}</b><strong>INR {latest?.put_option_close?.toFixed(2)}</strong><small>{latestPut?`BUY PUT | ${latestPut.strategy}`:"NO PUT ENTRY"}</small></div></div>}</div>;
}
