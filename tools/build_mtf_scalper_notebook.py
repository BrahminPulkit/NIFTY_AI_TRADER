from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}

cells = [
md("""# NIFTY 15M Trend + 5M EMA Pullback + 1M Entry Scalper

Paper signals only — no order-placement endpoint. Rules: closed 15m trend, closed 5m EMA pullback, closed 1m EMA reclaim, session VWAP, option-chain OI confirmation, ATR/swing SL and hard minimum 1:2 RR.

Backtest fills at the next 1m open, charges slippage/cost, gives SL priority if SL and target occur in the same candle, and squares off intraday. Historical option confirmation is used only when timestamped CE/PE OI snapshots are supplied. Without them, the backtest is explicitly a **spot diagnostic**, not an option-chain-confirmed result.
"""),
code("""from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.mtf_ema_pullback_scalper import (
    ScalperConfig, backtest, build_features, normalize_ohlcv,
    option_chain_confirmation, signal_at,
)

CFG = ScalperConfig(rr=2.0, atr_stop_multiplier=1.0, max_holding_bars=30)
import getpass
import requests

CLIENT_ID = getpass.getpass('Dhan Client ID: ')
ACCESS_TOKEN = getpass.getpass('Dhan Access Token: ')
HEADERS = {'Accept':'application/json', 'Content-Type':'application/json',
           'client-id':CLIENT_ID, 'access-token':ACCESS_TOKEN}

def dhan_post(url, payload):
    response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Dhan HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    if body.get('status') == 'failure':
        raise RuntimeError(f"Dhan API failure: {body}")
    return body.get('data', body)

CFG
"""),
md("""## Dhan-connected historical intraday backtest

Choose a date range. The downloader fetches NIFTY 1-minute index candles and ATM weekly CALL/PUT rolling-option OI directly from Dhan. Requests are split into 30-day non-inclusive batches, cached locally, and deduplicated. CE/PE OI change is computed causally within each session.
"""),
code("""def _date_batches(start_date, end_date, days=30):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start >= end:
        raise ValueError('START_DATE must be earlier than END_DATE (END_DATE is non-inclusive)')
    while start < end:
        stop = min(start + pd.Timedelta(days=days), end)
        yield start, stop
        start = stop

def fetch_dhan_index_range(start_date, end_date):
    parts = []
    for start, stop in _date_batches(start_date, end_date):
        raw = dhan_post('https://api.dhan.co/v2/charts/intraday', {
            'securityId':'13', 'exchangeSegment':'IDX_I', 'instrument':'INDEX',
            'interval':'1', 'oi':False,
            'fromDate':f"{start:%Y-%m-%d} 09:15:00",
            'toDate':f"{stop:%Y-%m-%d} 00:00:00"})
        expected = ['timestamp','open','high','low','close','volume']
        lengths = {key: len(raw.get(key, [])) for key in expected}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f'Misaligned Dhan index arrays: {lengths}')
        parts.append(pd.DataFrame({key:raw[key] for key in expected}))
    if not parts:
        raise RuntimeError('No Dhan index batches requested')
    return normalize_ohlcv(pd.concat(parts, ignore_index=True))

def fetch_dhan_rolling_side(start_date, end_date, option_type):
    parts = []
    response_key = 'ce' if option_type == 'CALL' else 'pe'
    for start, stop in _date_batches(start_date, end_date):
        raw = dhan_post('https://api.dhan.co/v2/charts/rollingoption', {
            'exchangeSegment':'NSE_FNO', 'interval':'1', 'securityId':13,
            'instrument':'OPTIDX', 'expiryFlag':'WEEK', 'expiryCode':1,
            'strike':'ATM', 'drvOptionType':option_type,
            'requiredData':['oi','spot'],
            'fromDate':f"{start:%Y-%m-%d}", 'toDate':f"{stop:%Y-%m-%d}"})
        side = raw.get(response_key) or {}
        if side.get('timestamp'):
            parts.append(pd.DataFrame({'timestamp':side['timestamp'],
                                       f'{response_key}_oi':side['oi']}))
    if not parts:
        raise RuntimeError(f'No rolling {option_type} data returned by Dhan')
    result = pd.concat(parts, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result.timestamp, unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
    return result.sort_values('timestamp').drop_duplicates('timestamp')

def fetch_dhan_historical_chain(start_date, end_date):
    ce = fetch_dhan_rolling_side(start_date, end_date, 'CALL')
    pe = fetch_dhan_rolling_side(start_date, end_date, 'PUT')
    chain = pd.merge(ce, pe, on='timestamp', how='inner').sort_values('timestamp')
    session = chain.timestamp.dt.date
    chain['ce_oi_change'] = chain.groupby(session).ce_oi.diff().fillna(0)
    chain['pe_oi_change'] = chain.groupby(session).pe_oi.diff().fillna(0)
    return chain

START_DATE = '2026-06-01'
END_DATE = '2026-07-01'  # non-inclusive
CACHE_DIR = ROOT / 'data/cache/dhan_mtf_scalper'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_tag = f'{START_DATE}_{END_DATE}'.replace('-', '')
spot_cache = CACHE_DIR / f'nifty_index_{cache_tag}.csv'
chain_cache = CACHE_DIR / f'nifty_atm_chain_{cache_tag}.csv'

if spot_cache.exists() and chain_cache.exists():
    spot = pd.read_csv(spot_cache)
    historical_chain = pd.read_csv(chain_cache)
    print('Loaded cached Dhan data')
else:
    spot = fetch_dhan_index_range(START_DATE, END_DATE)
    historical_chain = fetch_dhan_historical_chain(START_DATE, END_DATE)
    spot.to_csv(spot_cache, index=False)
    historical_chain.to_csv(chain_cache, index=False)
    print(f'Cached {len(spot):,} index bars and {len(historical_chain):,} chain rows')

trades, metrics = backtest(
    spot,
    historical_chain=historical_chain,
    require_option_confirmation=True,
    cfg=CFG,
)
print(metrics)
display(trades.tail(20))
"""),
code("""# Daily audit and equity curve
if not trades.empty:
    audit = trades.assign(day=pd.to_datetime(trades.entry_time).dt.date)
    display(audit.groupby('day').agg(
        trades=('result_r', 'size'), net_r=('result_r', 'sum'),
        win_rate=('result_r', lambda x: (x > 0).mean())
    ).tail(20))
    trades.assign(equity_points=trades.net_points.cumsum()).plot(
        x='exit_time', y='equity_points', figsize=(12, 4), grid=True,
        title=f"Equity curve — {metrics['mode']}")
else:
    print('No trades for the selected rules/data.')
"""),
md("""## Live closed-candle signal (Dhan, strict option-chain confirmation)

Credentials are requested with `getpass` and remain only in kernel memory. This cell fetches data and prints `BUY CE`, `BUY PE`, or `NO TRADE`; it never places an order. Dhan response shapes can change, so inspect the normalized chain table if your account returns `UNAVAILABLE`.
"""),
code("""def fetch_live_index_bars():
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    payload = {'securityId':'13', 'exchangeSegment':'IDX_I', 'instrument':'INDEX',
               'interval':'1', 'oi':False,
               'fromDate':f"{now:%Y-%m-%d} 09:15:00",
               'toDate':f"{now:%Y-%m-%d %H:%M:%S}"}
    raw = dhan_post('https://api.dhan.co/v2/charts/intraday', payload)
    return pd.DataFrame({k: raw[k] for k in ['timestamp','open','high','low','close','volume']})

def fetch_live_chain():
    expiries = dhan_post('https://api.dhan.co/v2/optionchain/expirylist',
                         {'UnderlyingScrip':13, 'UnderlyingSeg':'IDX_I'})
    today = pd.Timestamp.now(tz='Asia/Kolkata').date()
    expiry = min(pd.Timestamp(x) for x in expiries if pd.Timestamp(x).date() >= today)
    raw = dhan_post('https://api.dhan.co/v2/optionchain',
                    {'UnderlyingScrip':13, 'UnderlyingSeg':'IDX_I',
                     'Expiry':expiry.strftime('%Y-%m-%d')})
    rows = []
    for strike, legs in raw.get('oc', {}).items():
        ce, pe = legs.get('ce') or {}, legs.get('pe') or {}
        rows.append({'strike':float(strike), 'ce_oi':ce.get('oi',0), 'pe_oi':pe.get('oi',0),
                     'ce_oi_change':ce.get('oi_change',ce.get('change_in_oi',0)),
                     'pe_oi_change':pe.get('oi_change',pe.get('change_in_oi',0))})
    return pd.DataFrame(rows)
"""),
code("""def check_live_signal():
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    if now.weekday() >= 5 or not ('09:30' <= now.strftime('%H:%M') <= '14:45'):
        print('NO TRADE — outside entry window')
        return None
    bars, chain = fetch_live_index_bars(), fetch_live_chain()
    features = build_features(bars, CFG)
    closed = features[features.timestamp < now.floor('min')].reset_index(drop=True)
    setup = signal_at(closed, len(closed)-1, chain=chain,
                      require_option_confirmation=True, cfg=CFG)
    if setup is None:
        direction = 1 if len(closed) and bool(closed.iloc[-1].entry_long) else -1
        ok, detail = option_chain_confirmation(chain, direction)
        print('NO TRADE — rules/option chain not aligned', detail)
        return None
    print(f"{setup['signal']} | spot {setup['spot_signal_entry']:.2f} | "
          f"SL {setup['spot_stop']:.2f} | target {setup['spot_target']:.2f} | "
          f"RR 1:{setup['rr']:.1f} | PCR {setup['option_chain']['pcr']:.2f}")
    return setup

live_signal = check_live_signal()
"""),
md("""### Safety and interpretation

- Signal prices/SL/target are NIFTY spot levels; option premium execution needs a separately selected liquid contract and its own bid/ask-aware risk sizing.
- Backtest performance is not a promise of live returns. Inspect walk-forward periods, costs and data quality before paper trading.
- Live mode is strict: missing/rejected option-chain data always means `NO TRADE`.
"""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
    "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
