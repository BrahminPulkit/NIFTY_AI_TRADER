import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

cell7 = '''import time

LIVE_SIGNAL_COOLDOWN_MINUTES = 15
LIVE_MAX_SIGNALS_PER_DAY = 3

def _live_bias(features, chain):
    """Intrabar watch direction; this is preparation, not an entry signal."""
    if features.empty:
        return 'WAIT', 'no candles'
    row = features.iloc[-1]
    long_context = (int(row.get('m15_trend',0)) == 1
                    and int(row.get('m5_trend',0)) == 1
                    and row.close > row.vwap)
    short_context = (int(row.get('m15_trend',0)) == -1
                     and int(row.get('m5_trend',0)) == -1
                     and row.close < row.vwap)
    if long_context:
        approved, detail = option_chain_confirmation(chain,1)
        return ('WATCH BUY CE' if approved else 'WAIT'), (
            f"bullish price context | PCR {detail.get('pcr',float('nan')):.2f} | "
            f"chain {'aligned' if approved else 'not aligned'}")
    if short_context:
        approved, detail = option_chain_confirmation(chain,-1)
        return ('WATCH BUY PE' if approved else 'WAIT'), (
            f"bearish price context | PCR {detail.get('pcr',float('nan')):.2f} | "
            f"chain {'aligned' if approved else 'not aligned'}")
    return 'WAIT', '15m/5m/VWAP context not aligned'

def _live_snapshot():
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    bars, chain = fetch_live_index_bars(), fetch_live_chain()
    features = build_features(bars,CFG)
    closed = features[features.timestamp < now.floor('min')].reset_index(drop=True)
    bias, reason = _live_bias(features,chain)
    return now,features,closed,chain,bias,reason

def check_live_signal(verbose=True):
    """Return only a fully confirmed best-trade setup from the last closed candle."""
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    if now.weekday() >= 5 or not ('09:30' <= now.strftime('%H:%M') <= '14:45'):
        print(f'DHAN STATUS: IDLE | {now:%Y-%m-%d %H:%M:%S %Z} | outside entry window')
        return None
    try:
        now,features,closed,chain,bias,reason = _live_snapshot()
        if closed.empty:
            print('DHAN STATUS: CONNECTED | waiting for first closed candle')
            return None
        row = closed.iloc[-1]
        if verbose:
            print(f'DHAN CONNECTED | {now:%H:%M:%S} | NIFTY {features.iloc[-1].close:.2f} | '
                  f'{bias} | {reason}')
        setup = signal_at(closed,len(closed)-1,chain=chain,
                          require_option_confirmation=True,cfg=CFG)
        if setup is None:
            return None
        print('\\n'+'='*72)
        print(f"BEST TRADE SIGNAL: {setup['signal']} | confirmed candle {row.timestamp:%H:%M}")
        print(f"Spot {setup['spot_signal_entry']:.2f} | SL {setup['spot_stop']:.2f} | "
              f"Target {setup['spot_target']:.2f} | RR 1:{setup['rr']:.1f} | "
              f"PCR {setup['option_chain']['pcr']:.2f}")
        print('='*72)
        return setup
    except Exception as error:
        print(f'DHAN STATUS: DISCONNECTED/ERROR | {type(error).__name__}: {error}')
        return None

def run_live_monitor(poll_seconds=10):
    """Intrabar bias every change; closed-candle entry only for the best setup."""
    print('Dhan live monitor started — WATCH bias is not an entry.')
    print('A trade is valid only when BEST TRADE SIGNAL is printed.')
    print('Use Kernel Interrupt to stop.\\n')
    last_bias = None
    last_heartbeat = None
    last_evaluated_candle = None
    last_signal_time = None
    signalled_events = set()
    signal_count = 0
    while True:
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        if now.weekday() >= 5 or now.strftime('%H:%M') > '15:30':
            print(f'Monitor stopped: market closed at {now:%H:%M:%S}')
            break
        try:
            now,features,closed,chain,bias,reason = _live_snapshot()
            heartbeat_due = (last_heartbeat is None or
                (now-last_heartbeat).total_seconds() >= 60)
            if bias != last_bias or heartbeat_due:
                price = float(features.iloc[-1].close) if not features.empty else float('nan')
                print(f'[{now:%H:%M:%S}] DHAN CONNECTED | NIFTY {price:.2f} | {bias} | {reason}')
                last_bias,last_heartbeat = bias,now
            if closed.empty:
                time.sleep(max(5,int(poll_seconds))); continue
            candle_time = closed.iloc[-1].timestamp
            if candle_time != last_evaluated_candle:
                last_evaluated_candle = candle_time
                setup = signal_at(closed,len(closed)-1,chain=chain,
                                  require_option_confirmation=True,cfg=CFG)
                cooldown_ok = (last_signal_time is None or
                    (candle_time-last_signal_time).total_seconds() >=
                    LIVE_SIGNAL_COOLDOWN_MINUTES*60)
                if (setup is not None and setup['signal_time'] not in signalled_events
                        and cooldown_ok and signal_count < LIVE_MAX_SIGNALS_PER_DAY):
                    print('\\n'+'='*72)
                    print(f"BEST TRADE SIGNAL #{signal_count+1}: {setup['signal']} | "
                          f"confirmed {candle_time:%H:%M}")
                    print(f"Spot {setup['spot_signal_entry']:.2f} | "
                          f"SL {setup['spot_stop']:.2f} | Target {setup['spot_target']:.2f} | "
                          f"RR 1:{setup['rr']:.1f} | PCR {setup['option_chain']['pcr']:.2f}")
                    print('='*72+'\\n')
                    signalled_events.add(setup['signal_time'])
                    last_signal_time = candle_time
                    signal_count += 1
            if signal_count >= LIVE_MAX_SIGNALS_PER_DAY:
                print('Daily best-trade limit reached; monitoring stopped.')
                break
        except Exception as error:
            print(f'[{now:%H:%M:%S}] DHAN ERROR | {type(error).__name__}: {error}')
        time.sleep(max(5,int(poll_seconds)))

RUN_CONTINUOUS_MONITOR = True
if RUN_CONTINUOUS_MONITOR:
    run_live_monitor(poll_seconds=10)
else:
    live_signal = check_live_signal()
'''

nb["cells"][7]["source"] = cell7.splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(path)
