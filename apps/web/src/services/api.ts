import { brokerSchema, marketSchema, optionSchema, paperAccountSchema, paperReviewSchema, replaySessionSchema, replaySessionsSchema, signalAuditSchema, type BrokerState, type DeskResponse, type MarketResponse, type OptionResponse, type PaperAccount, type PaperResponse, type PaperReview, type ReplaySession, type ReplaySessions, type SignalAudit } from "../types/api";

const request = async <T>(path:string, init?:RequestInit):Promise<T> => {
  const response=await fetch(`/api/v1${path}`,init);
  if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(body.message||body.detail||`Request failed (${response.status})`)}
  return response.json() as Promise<T>;
};
const post=(path:string,body?:unknown)=>request<unknown>(path,{method:"POST",headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});

export const tradingApi={
  desk:()=>request<DeskResponse>("/desk"),
  market:async():Promise<MarketResponse>=>marketSchema.parse(await request<unknown>("/market")),
  options:async():Promise<OptionResponse>=>optionSchema.parse(await request<unknown>("/options")),
  watchOption:(security_id:string)=>post("/market/watch-option",{security_id}),
  paper:()=>request<PaperResponse>("/paper"),
  replaySessions:async(source:"frozen"|"dhan"|"rolling"="rolling"):Promise<ReplaySessions>=>replaySessionsSchema.parse(await request<unknown>(`/replay/${source==="dhan"?"broker/":source==="rolling"?"rolling/":""}sessions`)),
  replaySession:async(date:string,source:"frozen"|"dhan"|"rolling"="rolling"):Promise<ReplaySession>=>replaySessionSchema.parse(await request<unknown>(`/replay/${source==="dhan"?"broker/":source==="rolling"?"rolling/":""}session?date=${encodeURIComponent(date)}`)),
  signalAudit:async(date?:string):Promise<SignalAudit>=>signalAuditSchema.parse(await request<unknown>(`/audit/signals${date?`?date=${encodeURIComponent(date)}`:""}`)),
};
export const brokerApi={
  session:async():Promise<BrokerState>=>brokerSchema.parse(await request<unknown>("/broker/session")),
  connect:async(client_id:string,access_token:string)=>brokerSchema.parse(await post("/broker/connect",{client_id,access_token})),
  disconnect:async()=>brokerSchema.parse(await post("/broker/disconnect")),
  refresh:async()=>brokerSchema.parse(await post("/broker/refresh")),
};
export interface PaperContext{mode:"live"|"replay";session_date?:string;timestamp?:string}
export const paperApi={
  account:async(context:PaperContext):Promise<PaperAccount>=>paperAccountSchema.parse(await request<unknown>(`/paper/account?mode=${context.mode}${context.session_date?`&session_date=${encodeURIComponent(context.session_date)}`:""}`)),
  review:async(context:PaperContext):Promise<PaperReview>=>paperReviewSchema.parse(await post("/paper/review",context)),
  open:(context:PaperContext)=>post("/paper/orders",context),
  mark:(context:PaperContext)=>post("/paper/mark",context),
  exit:(context:PaperContext)=>post("/paper/exit",context),
  manualReview:async(security_id:string,quantity:number):Promise<PaperReview>=>paperReviewSchema.parse(await post("/paper/manual/review",{security_id,quantity})),
  manualOpen:(security_id:string,quantity:number)=>post("/paper/manual/orders",{security_id,quantity}),
};
