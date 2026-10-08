import { z } from "zod";

export const signalSchema = z.object({
  timestamp: z.string(), probability: z.number().nullable(), confidence: z.number().nullable(),
  decision: z.string(), action: z.string(), reason: z.string(), strategy: z.string(),
  regime: z.string(), risk_status: z.string(), latency_ms: z.number(),
  model_version: z.string(), instrument_scope: z.string(),
});
export const brokerSchema = z.object({
  connected: z.boolean(), status: z.string(), market_status: z.string(),
  last_data_update: z.string().nullable(), last_error: z.string().nullable(), worker_alive: z.boolean(),
  health: z.object({status:z.string(),detail:z.string(),latency_ms:z.number().nullable(),token_valid:z.boolean()}).passthrough(),
});
const candleSchema=z.object({timestamp:z.unknown(),open:z.number(),high:z.number(),low:z.number(),close:z.number(),volume:z.number().nullable().optional()});
export const marketSchema = z.object({
  market: z.object({
    connected:z.boolean(),status:z.string(),manager_status:z.string(),quotes:z.record(z.string(),z.number()),
    updated_at:z.string().nullable(),last_error:z.string().nullable(),signal:signalSchema.nullable(),
    signal_status:z.string(),signal_error:z.string().nullable(),
    put_signal:z.object({timestamp:z.string(),action:z.string(),decision:z.string(),probability:z.number().nullable(),threshold:z.number(),reason:z.string(),strategy:z.string(),model_version:z.string(),instrument_scope:z.string(),paper_only:z.boolean()}).nullable().default(null),
    put_signal_status:z.string().default("UNAVAILABLE"),put_signal_error:z.string().nullable().default(null),
    multistrategy_signal:z.object({timestamp:z.string(),status:z.string(),paper_only:z.boolean(),best_signal:z.object({timestamp:z.string(),side:z.string(),action:z.string(),decision:z.string(),strategy:z.string(),probability:z.number(),threshold:z.number(),reason:z.string(),paper_only:z.boolean()}).nullable(),checks:z.array(z.object({timestamp:z.string(),side:z.string(),action:z.string(),decision:z.string(),strategy:z.string(),probability:z.number(),threshold:z.number(),reason:z.string(),paper_only:z.boolean()}))}).nullable().default(null),
    multistrategy_status:z.string().default("UNAVAILABLE"),multistrategy_error:z.string().nullable().default(null),
  }),
  candles:z.array(candleSchema),
  option_candles:z.object({CE:z.array(candleSchema),PE:z.array(candleSchema)}).default({CE:[],PE:[]}),
  instruments:z.record(z.string(),z.record(z.string(),z.unknown()).nullable()).default({}),
  watched_candles:z.record(z.string(),z.array(candleSchema)).default({}),
});
export const optionSchema = z.object({
  connected:z.boolean(),status:z.string(),updated_at:z.string().nullable(),
  rows:z.array(z.object({
    security_id:z.union([z.string(),z.number()]).nullable().optional(),instrument_name:z.string().nullable().optional(),
    option_type:z.string().nullable().optional(),strike:z.number().nullable().optional(),expiry:z.string().nullable().optional(),
    ltp:z.number().nullable().optional(),oi:z.number().nullable().optional(),volume:z.number().nullable().optional(),
    bid:z.number().nullable().optional(),ask:z.number().nullable().optional(),spread:z.number().nullable().optional(),
  }).passthrough()),
});

export type Signal = z.infer<typeof signalSchema>;
export type BrokerState = z.infer<typeof brokerSchema>;
export type MarketResponse = z.infer<typeof marketSchema>;
export type OptionResponse = z.infer<typeof optionSchema>;

export interface StrategyGuide { name:string; summary:string; watch:string }
export interface DecisionCheck { label:string; ready:boolean }
export interface DeskState {
  available:boolean; action:string; tone:string; reason:string; probability:number; confidence:number;
  strategy:StrategyGuide; direction:string; regime:string; holding_minutes:number|null;
  expected_drawdown:number|null; expected_expansion:number|null; timestamp:string; source:string;
  freshness:{label:string;age:string;stale:boolean}; contract:Record<string,unknown>|null; checks:DecisionCheck[];
  writer_activity:{method:string;dominant:string;call_total_oi:number;put_total_oi:number;strongest_call:{option_type:string;strike:number;oi:number;ltp:number|null;expiry:string|null}|null;strongest_put:{option_type:string;strike:number;oi:number;ltp:number|null;expiry:string|null}|null};
}
export interface DeskResponse { desk:DeskState; market:MarketResponse["market"] }
export interface PaperResponse { statistics:Record<string,unknown>; journal:Record<string,unknown>[]; updated_at:string }
export const replaySessionsSchema=z.object({sessions:z.array(z.string()),count:z.number(),latest:z.string().nullable()});
export const replayEventSchema=z.object({timestamp:z.string(),action:z.string(),decision:z.string(),reason:z.string(),probability:z.number(),confidence:z.number(),strategy:z.string(),regime:z.string().default("MULTISTRATEGY"),source:z.string(),option_type:z.string().optional(),strike:z.number().nullable().optional(),expiry:z.string().nullable().optional(),entry_premium:z.number().optional(),index_price:z.number().optional(),threshold:z.number().optional()});
export const replaySessionSchema=z.object({
  session_date:z.string(),source:z.enum(["FROZEN_OOF_REPLAY","NATIVE_MODEL_AUDIT","ROLLING_ATM_STRATEGY_REPLAY"]),threshold:z.number(),
  candles:z.array(z.object({timestamp:z.string(),open:z.number(),high:z.number(),low:z.number(),close:z.number(),volume:z.number().nullable(),option_open:z.number(),option_high:z.number(),option_low:z.number(),option_close:z.number(),option_volume:z.number().nullable(),put_option_open:z.number().optional(),put_option_high:z.number().optional(),put_option_low:z.number().optional(),put_option_close:z.number().optional(),put_option_volume:z.number().nullable().optional(),call_strike:z.number().nullable().optional(),put_strike:z.number().nullable().optional()})),
  events:z.array(replayEventSchema),put_events:z.array(replayEventSchema).default([]),strategy_events:z.array(replayEventSchema.extend({side:z.string(),threshold:z.number(),paper_only:z.boolean()})).default([]),put_threshold:z.number().optional(),summary:z.object({candle_count:z.number(),setup_count:z.number(),approved_signal_count:z.number(),put_setup_count:z.number().optional(),put_approved_signal_count:z.number().optional(),strategy_setup_count:z.number().optional(),strategy_approved_signal_count:z.number().optional(),session_start:z.string(),session_end:z.string(),missing_candles:z.number().optional(),data_valid:z.boolean().optional(),model_status:z.string().optional(),model_reason:z.string().nullable().optional()}),
});
export type ReplaySessions=z.infer<typeof replaySessionsSchema>;
export type ReplaySession=z.infer<typeof replaySessionSchema>;
export type ReplayEvent=z.infer<typeof replayEventSchema>;
export const paperPositionSchema=z.object({trade_id:z.string(),entry_timestamp:z.string(),strategy:z.string(),regime:z.string(),probability:z.number(),entry_premium:z.number(),quantity:z.number(),stop_loss:z.number(),target:z.number(),current_premium:z.number(),current_pnl:z.number(),maximum_favourable_excursion:z.number(),maximum_adverse_excursion:z.number(),entry_mode:z.string().optional(),market:z.string().optional(),security_id:z.string().nullable().optional(),option_type:z.string().optional(),strike:z.number().nullable().optional(),expiry:z.string().nullable().optional()});
export const paperAccountSchema=z.object({mode:z.string(),session_date:z.string().nullable(),capital:z.number(),initial_capital:z.number(),position:paperPositionSchema.nullable(),statistics:z.record(z.string(),z.unknown()),trades:z.array(z.record(z.string(),z.unknown())),rules:z.object({quantity:z.number(),stop_loss_pct:z.number(),target_pct:z.number(),maximum_holding_minutes:z.number(),maximum_daily_trades:z.number(),daily_loss_limit:z.number(),daily_profit_target:z.number(),broker_orders_enabled:z.literal(false)})});
export const paperReviewSchema=z.object({eligible:z.boolean(),reason:z.string(),timestamp:z.string(),strategy:z.string(),probability:z.number(),entry_premium:z.number(),quantity:z.number(),stop_loss:z.number(),target:z.number(),capital_required:z.number(),maximum_loss:z.number(),broker_orders_enabled:z.literal(false),security_id:z.string().optional(),option_type:z.string().optional(),strike:z.number().optional(),expiry:z.string().optional()});
export type PaperAccount=z.infer<typeof paperAccountSchema>;
export type PaperReview=z.infer<typeof paperReviewSchema>;
export const signalAuditSchema=z.object({
  session_date:z.string(),scope:z.string(),threshold:z.number(),
  methodology:z.object({forward_minutes:z.number(),review_move_pct:z.number(),note:z.string()}),
  summary:z.object({market_candles:z.number(),audited_candles:z.number(),coverage_pct:z.number(),model_evaluations:z.number(),approved_signals:z.number(),review_candidates:z.number(),data_gaps:z.number()}),
  reason_breakdown:z.array(z.object({reason:z.string(),count:z.number()})),
  events:z.array(z.object({timestamp:z.string(),setup:z.string(),stage1:z.string(),model:z.string(),probability:z.number().nullable(),threshold:z.number(),decision:z.string(),action:z.string(),reason:z.string(),forward_up_pct:z.number().nullable(),forward_down_pct:z.number().nullable(),review_candidate:z.boolean(),outcome_window_complete:z.boolean()})),
});
export type SignalAudit=z.infer<typeof signalAuditSchema>;
