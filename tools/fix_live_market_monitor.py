"""Fix live OI deltas and make Dhan connectivity explicit in the existing notebook."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

cell6 = '''def fetch_live_index_bars():
    """Fetch today's live NIFTY 1-minute candles from Dhan."""
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    payload = {'securityId':'13', 'exchangeSegment':'IDX_I', 'instrument':'INDEX',
               'interval':'1', 'oi':False,
               'fromDate':f"{now:%Y-%m-%d} 09:15:00",
               'toDate':f"{now:%Y-%m-%d %H:%M:%S}"}
    raw = dhan_post('https://api.dhan.co/v2/charts/intraday', payload)
    required = ['timestamp','open','high','low','close','volume']
    lengths = {key:len(raw.get(key,[])) for key in required}
    if not lengths['timestamp'] or len(set(lengths.values())) != 1:
        raise RuntimeError(f'Invalid live candle response from Dhan: {lengths}')
    return pd.DataFrame({key:raw[key] for key in required})

def fetch_live_chain():
    """Fetch nearest-expiry chain and derive OI change from Dhan's previous_oi."""
    expiries = dhan_post('https://api.dhan.co/v2/optionchain/expirylist',
                         {'UnderlyingScrip':13, 'UnderlyingSeg':'IDX_I'})
    today = pd.Timestamp.now(tz='Asia/Kolkata').date()
    valid = [pd.Timestamp(x) for x in expiries if pd.Timestamp(x).date() >= today]
    if not valid:
        raise RuntimeError('Dhan returned no active NIFTY option expiry')
    expiry = min(valid)
    raw = dhan_post('https://api.dhan.co/v2/optionchain',
                    {'UnderlyingScrip':13, 'UnderlyingSeg':'IDX_I',
                     'Expiry':expiry.strftime('%Y-%m-%d')})
    rows = []
    for strike, legs in raw.get('oc', {}).items():
        ce, pe = legs.get('ce') or {}, legs.get('pe') or {}
        ce_oi, pe_oi = float(ce.get('oi') or 0), float(pe.get('oi') or 0)
        ce_previous = float(ce.get('previous_oi') or ce_oi)
        pe_previous = float(pe.get('previous_oi') or pe_oi)
        rows.append({'strike':float(strike), 'expiry':expiry.strftime('%Y-%m-%d'),
                     'ce_oi':ce_oi, 'pe_oi':pe_oi,
                     'ce_oi_change':ce_oi-ce_previous,
                     'pe_oi_change':pe_oi-pe_previous})
    chain = pd.DataFrame(rows)
    if chain.empty:
        raise RuntimeError('Dhan returned an empty NIFTY option chain')
    return chain
'''

cell7 = '''import time

def check_live_signal(verbose=True):
    """One strict decision from the latest fully closed candle."""
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    if now.weekday() >= 5 or not ('09:30' <= now.strftime('%H:%M') <= '14:45'):
        print(f'DHAN STATUS: IDLE | {now:%Y-%m-%d %H:%M:%S %Z} | outside entry window')
        return None
    try:
        bars = fetch_live_index_bars()
        chain = fetch_live_chain()
        features = build_features(bars, CFG)
        closed = features[features.timestamp < now.floor('min')].reset_index(drop=True)
        if closed.empty:
            print('DHAN STATUS: CONNECTED | waiting for first closed candle')
            return None
        row = closed.iloc[-1]
        bullish_trigger = bool(row.get('entry_long',False))
        bearish_trigger = bool(row.get('entry_short',False))
        pcr = float(chain.pe_oi.sum()/max(chain.ce_oi.sum(),1.0))
        if verbose:
            print(f'DHAN STATUS: CONNECTED | checked {now:%H:%M:%S} | '
                  f'closed candle {row.timestamp:%H:%M} | NIFTY {row.close:.2f} | '
                  f'bars {len(closed)} | chain strikes {len(chain)} | PCR {pcr:.2f}')
        if not bullish_trigger and not bearish_trigger:
            print('NO TRADE — live connection is healthy; no 1-minute entry trigger')
            return None
        direction = 1 if bullish_trigger else -1
        chain_ok, chain_detail = option_chain_confirmation(chain,direction)
        setup = signal_at(closed,len(closed)-1,chain=chain,
                          require_option_confirmation=True,cfg=CFG)
        if setup is None:
            print('NO TRADE — entry trigger found but MTF/VWAP/option-chain gates rejected it',
                  chain_detail)
            return None
        print(f"{setup['signal']} | spot {setup['spot_signal_entry']:.2f} | "
              f"SL {setup['spot_stop']:.2f} | target {setup['spot_target']:.2f} | "
              f"RR 1:{setup['rr']:.1f} | PCR {setup['option_chain']['pcr']:.2f}")
        return setup
    except Exception as error:
        print(f'DHAN STATUS: DISCONNECTED/ERROR | {type(error).__name__}: {error}')
        return None

def run_live_monitor(poll_seconds=10):
    """Continuously check once per newly closed minute; paper signals only."""
    print('Starting Dhan live monitor — press Kernel Interrupt to stop.')
    checked_minute = None
    while True:
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        if now.weekday() >= 5 or now.strftime('%H:%M') > '15:30':
            print(f'Monitor stopped: market closed at {now:%H:%M:%S}')
            break
        current_minute = now.floor('min')
        if current_minute != checked_minute and now.second >= 3:
            check_live_signal(verbose=True)
            checked_minute = current_minute
        time.sleep(max(5,int(poll_seconds)))

# One immediate connection and signal check:
live_signal = check_live_signal()

# For continuous monitoring, run this manually in a separate execution:
# run_live_monitor(poll_seconds=10)
'''

for index, source in [(6,cell6),(7,cell7)]:
    nb["cells"][index]["source"] = source.splitlines(keepends=True)
    nb["cells"][index]["outputs"] = []
    nb["cells"][index]["execution_count"] = None
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
