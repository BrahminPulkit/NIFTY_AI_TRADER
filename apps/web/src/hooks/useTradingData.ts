import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { brokerApi, paperApi, tradingApi, type PaperContext } from "../services/api";

export const queryKeys={broker:["broker"] as const,desk:["desk"] as const,market:["market"] as const,options:["options"] as const,paper:["paper"] as const};
export const useBroker=()=>useQuery({queryKey:queryKeys.broker,queryFn:brokerApi.session,refetchInterval:8000});
export const useDesk=()=>useQuery({queryKey:queryKeys.desk,queryFn:tradingApi.desk,refetchInterval:10000});
export const useMarket=()=>useQuery({queryKey:queryKeys.market,queryFn:tradingApi.market,refetchInterval:5000});
export const useOptions=()=>useQuery({queryKey:queryKeys.options,queryFn:tradingApi.options,refetchInterval:8000});
export const usePaper=()=>useQuery({queryKey:queryKeys.paper,queryFn:tradingApi.paper,refetchInterval:15000});
export const useReplaySessions=(source:"frozen"|"dhan"|"rolling"="rolling")=>useQuery({queryKey:["replay-sessions",source],queryFn:()=>tradingApi.replaySessions(source),staleTime:source==="dhan"?15000:300000});
export const useReplaySession=(date:string,source:"frozen"|"dhan"|"rolling"="rolling")=>useQuery({queryKey:["replay-session",source,date],queryFn:()=>tradingApi.replaySession(date,source),enabled:Boolean(date),staleTime:source==="dhan"?15000:300000});
export const useSignalAudit=(date?:string)=>useQuery({queryKey:["signal-audit",date],queryFn:()=>tradingApi.signalAudit(date),refetchInterval:15000});
export const usePaperAccount=(mode:"live"|"replay"="live",sessionDate?:string)=>useQuery({queryKey:["paper-account",mode,sessionDate],queryFn:()=>paperApi.account({mode,session_date:sessionDate}),enabled:mode==="live"||Boolean(sessionDate),refetchInterval:mode==="live"?3000:false});
export function usePaperActions(mode:"live"|"replay",sessionDate?:string){const client=useQueryClient();const refresh=()=>client.invalidateQueries({queryKey:["paper-account",mode,sessionDate]});return{
  review:useMutation({mutationFn:(timestamp?:string)=>paperApi.review({mode,session_date:sessionDate,timestamp})}),
  open:useMutation({mutationFn:(timestamp?:string)=>paperApi.open({mode,session_date:sessionDate,timestamp}),onSuccess:refresh}),
  mark:useMutation({mutationFn:(timestamp:string)=>paperApi.mark({mode,session_date:sessionDate,timestamp}),onSuccess:refresh}),
  exit:useMutation({mutationFn:(timestamp?:string)=>paperApi.exit({mode,session_date:sessionDate,timestamp}),onSuccess:refresh}),
}}
export function useManualPaperActions(){const client=useQueryClient();const refresh=()=>client.invalidateQueries({queryKey:["paper-account","live",undefined]});return{
  review:useMutation({mutationFn:({securityId,quantity}:{securityId:string;quantity:number})=>paperApi.manualReview(securityId,quantity)}),
  open:useMutation({mutationFn:({securityId,quantity}:{securityId:string;quantity:number})=>paperApi.manualOpen(securityId,quantity),onSuccess:refresh}),
}}
export function useBrokerActions(){const client=useQueryClient();const done=()=>{client.invalidateQueries({queryKey:queryKeys.broker});client.invalidateQueries({queryKey:queryKeys.market});client.invalidateQueries({queryKey:queryKeys.desk})};return{
  connect:useMutation({mutationFn:({clientId,token}:{clientId:string;token:string})=>brokerApi.connect(clientId,token),onSuccess:done}),
  disconnect:useMutation({mutationFn:brokerApi.disconnect,onSuccess:done}),refresh:useMutation({mutationFn:brokerApi.refresh,onSuccess:done}),
}}
