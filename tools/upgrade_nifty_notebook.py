"""Surgical in-place upgrade for NIFTY_Scalping_Paper_Trading.ipynb.

The transformation preserves cell count, order, ids, outputs and metadata.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
BACKUP = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.pre_institutional.ipynb"


def source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
if len(nb["cells"]) != 12:
    raise RuntimeError("Unexpected notebook shape; refusing to edit")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

config_marker = "    option_target_pct: float = 0.40\n"
config_upgrade = """    option_target_pct: float = 0.40
    adx_period: int = 14
    minimum_adx: float = 18.0
    atr_percentile_lookback: int = 60
    minimum_atr_percentile: float = 0.20
    maximum_atr_percentile: float = 0.95
    swing_lookback: int = 3
    structure_lookback: int = 20
    opening_range_end: str = '09:30'
    minimum_orb_buffer_atr: float = 0.05
    institutional_body_atr: float = 0.80
    institutional_volume_ratio: float = 1.50
    value_area_pct: float = 0.70
    minimum_pcr: float = 0.70
    maximum_pcr: float = 1.40
    minimum_confidence_score: float = 70.0
    daily_max_drawdown_r: float = 2.0
    breakeven_at_r: float = 1.0
    trail_after_r: float = 2.0
    trail_atr: float = 1.0
    ml_enabled: bool = True
    ml_probability_threshold: float = 0.90
"""
cell3 = "".join(nb["cells"][3]["source"])
if "minimum_confidence_score" not in cell3:
    if config_marker not in cell3:
        raise RuntimeError("Config marker missing")
    nb["cells"][3]["source"] = source(cell3.replace(config_marker, config_upgrade))

institutional = r'''

# --- Institutional feature and decision layer (causal; closed candles only) ---
_build_features_base = build_features
_select_affordable_nifty_option_base = select_affordable_nifty_option

def _wilder(series, period):
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _adx(frame, period=14):
    up, down = frame['high'].diff(), -frame['low'].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = frame['close'].shift()
    tr = pd.concat([(frame['high']-frame['low']), (frame['high']-prev).abs(),
                    (frame['low']-prev).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, period).replace(0, np.nan)
    plus_di, minus_di = 100*_wilder(plus_dm, period)/atr, 100*_wilder(minus_dm, period)/atr
    dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0, np.nan)
    return _wilder(dx, period), plus_di, minus_di

def _closed_mtf(frame, rule, prefix):
    bars = (frame.set_index('timestamp')[['open','high','low','close','futures_volume']]
            .resample(rule, label='right', closed='left')
            .agg({'open':'first','high':'max','low':'min','close':'last','futures_volume':'sum'})
            .dropna().reset_index())
    bars[f'{prefix}_fast'] = bars['close'].ewm(span=3 if prefix == 'trend15' else 5, adjust=False).mean()
    bars[f'{prefix}_slow'] = bars['close'].ewm(span=8 if prefix == 'trend15' else 13, adjust=False).mean()
    bars[f'{prefix}_slope'] = bars[f'{prefix}_slow'].diff()
    bars[f'{prefix}_direction'] = np.select(
        [(bars[f'{prefix}_fast'] > bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] > 0),
         (bars[f'{prefix}_fast'] < bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] < 0)],
        [1, -1], default=0)
    return bars[['timestamp',f'{prefix}_fast',f'{prefix}_slow',f'{prefix}_slope',f'{prefix}_direction']]

def _volume_profile_levels(group):
    result = pd.DataFrame(index=group.index, columns=['poc','vah','val'], dtype=float)
    for end in range(len(group)):
        sample = group.iloc[:end+1]
        lo, hi = float(sample['low'].min()), float(sample['high'].max())
        if hi <= lo:
            result.iloc[end] = [sample['close'].iloc[-1]]*3
            continue
        bins = np.linspace(lo, hi, 31)
        bucket = np.clip(np.digitize(((sample.high+sample.low+sample.close)/3), bins)-1, 0, 29)
        profile = pd.Series(sample.futures_volume.to_numpy()).groupby(bucket).sum()
        poc_bin = int(profile.idxmax())
        ordered = profile.sort_values(ascending=False)
        chosen, total = [], max(float(profile.sum()), 1.0)
        running = 0.0
        for b, vol in ordered.items():
            chosen.append(int(b)); running += float(vol)
            if running/total >= CFG.value_area_pct: break
        centers = (bins[:-1]+bins[1:])/2
        result.iloc[end] = [centers[poc_bin], centers[max(chosen)], centers[min(chosen)]]
    return result

def build_features(index_bars, futures_bars):
    data = _build_features_base(index_bars, futures_bars)
    adx, plus_di, minus_di = _adx(data, CFG.adx_period)
    data['adx'], data['plus_di'], data['minus_di'] = adx, plus_di, minus_di
    atr_min = data['atr'].shift(1).rolling(CFG.atr_percentile_lookback, min_periods=20).min()
    atr_max = data['atr'].shift(1).rolling(CFG.atr_percentile_lookback, min_periods=20).max()
    data['atr_percentile'] = (data['atr']-atr_min)/(atr_max-atr_min).replace(0, np.nan)
    for rule, prefix in [('5min','trend5'), ('15min','trend15')]:
        data = pd.merge_asof(data.sort_values('timestamp'), _closed_mtf(data, rule, prefix),
                             on='timestamp', direction='backward')
    data['trend1_direction'] = np.select(
        [(data.ema_fast > data.ema_slow) & (data.trend_slope > 0),
         (data.ema_fast < data.ema_slow) & (data.trend_slope < 0)], [1,-1], default=0)
    left_high = data.high.shift(1).rolling(CFG.swing_lookback).max()
    right_high = data.high.shift(-1).rolling(CFG.swing_lookback).max().shift(-(CFG.swing_lookback-1))
    left_low = data.low.shift(1).rolling(CFG.swing_lookback).min()
    right_low = data.low.shift(-1).rolling(CFG.swing_lookback).min().shift(-(CFG.swing_lookback-1))
    # Confirm pivots only after the right-hand candles close, then forward-fill.
    confirmed_high = data.high.where((data.high > left_high) & (data.high >= right_high)).shift(CFG.swing_lookback)
    confirmed_low = data.low.where((data.low < left_low) & (data.low <= right_low)).shift(CFG.swing_lookback)
    data['swing_high'] = confirmed_high.ffill()
    data['swing_low'] = confirmed_low.ffill()
    data['previous_swing_high'] = confirmed_high.ffill().shift(1)
    data['previous_swing_low'] = confirmed_low.ffill().shift(1)
    data['market_structure'] = np.select(
        [(data.swing_high > data.previous_swing_high) & (data.swing_low >= data.previous_swing_low),
         (data.swing_low < data.previous_swing_low) & (data.swing_high <= data.previous_swing_high)],
        ['HH_HL','LH_LL'], default='TRANSITION')
    data['swing_break_up'] = (data.close > data.swing_high.shift(1)) & (data.close.shift(1) <= data.swing_high.shift(1))
    data['swing_break_down'] = (data.close < data.swing_low.shift(1)) & (data.close.shift(1) >= data.swing_low.shift(1))
    session_minute = data.timestamp.dt.hour*60+data.timestamp.dt.minute
    opening = data[(session_minute >= 555) & (session_minute < 570)]
    levels = opening.groupby('session').agg(orb_high=('high','max'), orb_low=('low','min'))
    data = data.join(levels, on='session')
    data['orb_break_up'] = (session_minute >= 570) & (data.close > data.orb_high + CFG.minimum_orb_buffer_atr*data.atr)
    data['orb_break_down'] = (session_minute >= 570) & (data.close < data.orb_low - CFG.minimum_orb_buffer_atr*data.atr)
    candle_range = (data.high-data.low).replace(0, np.nan)
    body = (data.close-data.open).abs()
    data['institutional_candle'] = ((body >= CFG.institutional_body_atr*data.atr) &
        (data.volume_ratio >= CFG.institutional_volume_ratio) & (body/candle_range >= .65))
    data['liquidity_sweep_up'] = (data.low < data.liquidity_low) & (data.close > data.liquidity_low) & (data.close > data.open)
    data['liquidity_sweep_down'] = (data.high > data.liquidity_high) & (data.close < data.liquidity_high) & (data.close < data.open)
    data['fake_break_up'] = (data.high > data.liquidity_high) & (data.close <= data.liquidity_high)
    data['fake_break_down'] = (data.low < data.liquidity_low) & (data.close >= data.liquidity_low)
    data['dynamic_resistance'] = pd.concat([data.swing_high, data.liquidity_high, data.orb_high], axis=1).min(axis=1)
    data['dynamic_support'] = pd.concat([data.swing_low, data.liquidity_low, data.orb_low], axis=1).max(axis=1)
    profiles = data.groupby('session', group_keys=False).apply(_volume_profile_levels, include_groups=False)
    data[['poc','vah','val']] = profiles[['poc','vah','val']].to_numpy()
    data['market_regime'] = np.select(
        [data.atr_percentile >= .80,
         (data.adx >= CFG.minimum_adx) & (data.trend5_direction != 0) & (data.trend15_direction != 0)],
        ['HIGH_VOLATILITY','TRENDING'], default='RANGE')
    return data.reset_index(drop=True)

def fetch_option_chain_confirmation(direction):
    expiry_data = _post_dhan('https://api.dhan.co/v2/optionchain/expirylist',
                             {'UnderlyingScrip':13,'UnderlyingSeg':'IDX_I'})
    today = pd.Timestamp.now(tz='Asia/Kolkata').date()
    expiry = min(pd.Timestamp(x).strftime('%Y-%m-%d') for x in expiry_data
                 if pd.Timestamp(x).date() >= today)
    chain = _post_dhan('https://api.dhan.co/v2/optionchain',
                       {'UnderlyingScrip':13,'UnderlyingSeg':'IDX_I','Expiry':expiry})
    ce_oi = pe_oi = ce_change = pe_change = 0.0
    for legs in chain.get('oc', {}).values():
        ce, pe = legs.get('ce') or {}, legs.get('pe') or {}
        ce_oi += float(ce.get('oi') or 0); pe_oi += float(pe.get('oi') or 0)
        ce_change += float(ce.get('oi_change') or ce.get('change_in_oi') or 0)
        pe_change += float(pe.get('oi_change') or pe.get('change_in_oi') or 0)
    pcr = pe_oi/max(ce_oi, 1.0)
    call_writing, put_writing = ce_change > 0, pe_change > 0
    aligned = ((direction == 1 and put_writing and pcr >= CFG.minimum_pcr) or
               (direction == -1 and call_writing and pcr <= CFG.maximum_pcr))
    return {'pcr':pcr, 'call_writing':call_writing, 'put_writing':put_writing,
            'ce_oi_change':ce_change, 'pe_oi_change':pe_change, 'aligned':aligned}

def select_affordable_nifty_option(direction):
    option = _select_affordable_nifty_option_base(direction)
    if option is None: return None
    confirmation = fetch_option_chain_confirmation(direction)
    if not confirmation['aligned']: return None
    option.update(confirmation)
    # Prefer 0.45-0.60 delta; reject low-delta lottery contracts.
    if not .35 <= option['delta'] <= .65: return None
    return option

def _ml_confirmation(index_history, option, direction):
    """Use the frozen production model when its full live contract is available."""
    if not CFG.ml_enabled:
        return {'approved':True, 'probability':None, 'reason':'ML_DISABLED'}
    try:
        from src.live_feature_generator import build_live_contract_row
        from src.production_predictor import ProductionPredictor
        observations_path = Path(r'E:\NIFTY_AI_TRADER\data\runtime\decision_observations.parquet')
        if not observations_path.exists():
            return {'approved':False, 'probability':None, 'reason':'ML_DECISION_CONTRACT_UNAVAILABLE'}
        trading_date = index_history.timestamp.iloc[-1].strftime('%Y-%m-%d')
        option_history = fetch_intraday(option['security_id'], 'NSE_FNO', 'OPTIDX', trading_date)
        predictor = ProductionPredictor(Path(r'E:\NIFTY_AI_TRADER\production_model'))
        row, audit = build_live_contract_row(index_history, option_history,
            pd.read_parquet(observations_path), predictor.contract['feature_order'])
        if row is None:
            return {'approved':False, 'probability':None, 'reason':audit.get('reason','ML_CONTRACT_BLOCK')}
        output = predictor.predict(row)
        probability = float(output['probability'])
        return {'approved': output['decision']=='TRADE' and probability >= CFG.ml_probability_threshold,
                'probability':probability, 'reason':output['reason']}
    except Exception as error:
        return {'approved':False, 'probability':None, 'reason':f'ML_SAFE_BLOCK:{error}'}

def evaluate_baseline_setup(data, i):
    """Institutional 15m/5m/1m alignment with structure, volatility and confidence."""
    if i < 55: return None
    row, previous = data.iloc[i], data.iloc[i-1]
    if any(pd.isna(row.get(k)) for k in ['atr','adx','vwap','trend5_direction','trend15_direction']):
        return None
    direction = int(row['trend1_direction'])
    if direction == 0 or direction != int(row['trend5_direction']) or direction != int(row['trend15_direction']):
        return None
    structure_ok = row.market_structure == ('HH_HL' if direction == 1 else 'LH_LL')
    swing_break = bool(row.swing_break_up if direction == 1 else row.swing_break_down)
    orb_break = bool(row.orb_break_up if direction == 1 else row.orb_break_down)
    sweep = bool(row.liquidity_sweep_up if direction == 1 else row.liquidity_sweep_down)
    fake_against = bool(row.fake_break_up if direction == 1 else row.fake_break_down)
    vwap_ok = row.close > row.vwap and row.vwap >= previous.vwap if direction == 1 else row.close < row.vwap and row.vwap <= previous.vwap
    adx_ok = row.adx >= CFG.minimum_adx and (row.plus_di > row.minus_di if direction == 1 else row.minus_di > row.plus_di)
    atr_ok = CFG.minimum_atr_percentile <= row.atr_percentile <= CFG.maximum_atr_percentile
    trigger = swing_break or orb_break or sweep
    if not (structure_ok and trigger and vwap_ok and adx_ok and atr_ok and not fake_against):
        return None
    atr, entry = float(row.atr), float(row.close)
    support = float(row.dynamic_support if direction == 1 else row.dynamic_resistance)
    raw_stop = support-direction*CFG.swing_buffer_atr*atr
    risk = direction*(entry-raw_stop)
    if risk <= 0 or risk > CFG.maximum_stop_atr*atr:
        risk = CFG.minimum_stop_atr*atr
    risk = max(risk, CFG.minimum_stop_atr*atr)
    opposing = float(row.dynamic_resistance if direction == 1 else row.dynamic_support)
    room_r = direction*(opposing-entry)/max(risk,.01)
    planned_r = min(CFG.target_r, room_r) if room_r >= CFG.minimum_room_r else CFG.target_r
    score_parts = {
        'mtf':25.0, 'structure':15.0, 'swing_or_orb':15.0,
        'vwap':10.0, 'adx':10.0, 'atr':5.0,
        'volume':10.0*float(np.clip(row.volume_ratio/CFG.institutional_volume_ratio,0,1)),
        'institutional_candle':5.0 if row.institutional_candle else 0.0,
        'volume_profile':5.0 if (row.close >= row.poc if direction == 1 else row.close <= row.poc) else 0.0}
    confidence = round(min(100.0, sum(score_parts.values())), 1)
    if confidence < CFG.minimum_confidence_score: return None
    setup_type = 'LIQUIDITY_SWEEP' if sweep else ('ORB' if orb_break else 'SWING_BREAK')
    return {'event_time':row.timestamp, 'direction':direction,
        'side':'BUY CE' if direction == 1 else 'BUY PE', 'setup':setup_type,
        'entry':entry, 'stop':entry-direction*risk, 'target':entry+direction*risk*planned_r,
        'risk':risk, 'planned_r':planned_r, 'entry_time':row.timestamp,
        'rsi':float(row.rsi), 'gap_atr':float(row.gap_atr), 'volume_ratio':float(row.volume_ratio),
        'confidence_score':confidence, 'score_components':score_parts,
        'market_structure':row.market_structure, 'market_regime':row.market_regime,
        'adx':float(row.adx), 'atr_percentile':float(row.atr_percentile),
        'poc':float(row.poc), 'vah':float(row.vah), 'val':float(row.val), 'room_r':room_r}

def simulate_managed_exit(data, entry_index, setup, max_bars=60):
    """Conservative OHLC simulation: SL-to-cost at 1R and ATR trail after 2R."""
    direction, entry, initial_risk = setup['direction'], setup['entry'], setup['risk']
    stop, target = setup['stop'], setup['target']
    highest, lowest = entry, entry
    end = min(entry_index+max_bars, len(data)-1)
    for j in range(entry_index+1, end+1):
        bar = data.iloc[j]
        if bar.session != data.iloc[entry_index].session: break
        highest, lowest = max(highest,float(bar.high)), min(lowest,float(bar.low))
        favorable_r = direction*((highest if direction == 1 else lowest)-entry)/initial_risk
        if favorable_r >= CFG.breakeven_at_r:
            stop = max(stop,entry) if direction == 1 else min(stop,entry)
        if favorable_r >= CFG.trail_after_r:
            candidate = (highest-CFG.trail_atr*bar.atr if direction == 1
                         else lowest+CFG.trail_atr*bar.atr)
            stop = max(stop,candidate) if direction == 1 else min(stop,candidate)
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        target_hit = bar.high >= target if direction == 1 else bar.low <= target
        if stop_hit:
            exit_price = stop
            return j, ('LOSS' if direction*(exit_price-entry) < 0 else 'BREAKEVEN'), direction*(exit_price-entry)/initial_risk
        if target_hit:
            return j, 'WIN', setup['planned_r']
    exit_price = float(data.iloc[end].close)
    return end, 'TIME', direction*(exit_price-entry)/initial_risk

print('Institutional feature, confirmation and risk layer loaded.')
'''
cell4 = "".join(nb["cells"][4]["source"])
if "Institutional feature and decision layer" not in cell4:
    nb["cells"][4]["source"] = source(cell4 + institutional)

cell2 = '''def check_current_signal():
    trading_date = datetime.now().strftime('%Y-%m-%d')
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    if not (9*60 + 30 <= now.hour*60 + now.minute <= 14*60 + 30):
        print('NO TRADE — institutional entries are allowed only during 09:30–14:30.')
        return None
    future_id, future_name = find_nearest_nifty_future(trading_date)
    index_bars = fetch_intraday('13','IDX_I','INDEX',trading_date)
    futures_bars = fetch_intraday(future_id,'NSE_FNO','FUTIDX',trading_date)
    if index_bars.empty or futures_bars.empty:
        print('NO TRADE — matching index/futures candles unavailable.'); return None
    data = build_features(index_bars, futures_bars)
    closed = data[data.timestamp < now.floor('min')].reset_index(drop=True)
    setup = evaluate_baseline_setup(closed, len(closed)-1) if len(closed) >= 56 else None
    if setup is None:
        print('NO TRADE — MTF/structure/VWAP/ORB/ADX/ATR gates are not aligned.'); return None
    option = select_affordable_nifty_option(setup['direction'])
    if option is None:
        print('NO TRADE — option-chain direction or contract-quality checks failed.'); return None
    ml = _ml_confirmation(index_bars, option, setup['direction'])
    if not ml['approved']:
        print(f"NO TRADE — {ml['reason']}"); return None
    premium_entry = float(option['entry_price'])
    premium_risk_pct = float(np.clip(setup['risk']/setup['entry']*option['delta']*4, .15, .30))
    premium_stop = premium_entry*(1-premium_risk_pct)
    premium_target = premium_entry*(1+premium_risk_pct*setup['planned_r'])
    result = dict(setup)
    result.update(option=option, ml=ml, option_entry=premium_entry,
                  option_stop=premium_stop, option_target=premium_target)
    print(f"APPROVED {setup['side']} | {setup['setup']} | confidence {setup['confidence_score']:.1f}/100")
    print(f"Regime {setup['market_regime']} | structure {setup['market_structure']} | ADX {setup['adx']:.1f}")
    print(f"PCR {option['pcr']:.2f} | ML {ml['probability']:.3f} | premium {premium_entry:.2f} | SL {premium_stop:.2f} | target {premium_target:.2f}")
    return result
'''
nb["cells"][2]["source"] = source(cell2)

cell8 = '''# Backtest the prior session by default; change TEST_DATE as needed.
TEST_DATE = (pd.Timestamp.now(tz='Asia/Kolkata')-pd.Timedelta(days=1)).strftime('%Y-%m-%d')

def backtest_specific_date(test_date):
    future_id, future_name = find_nearest_nifty_future(test_date)
    index_bars = fetch_intraday('13','IDX_I','INDEX',test_date)
    futures_bars = fetch_intraday(future_id,'NSE_FNO','FUTIDX',test_date)
    if index_bars.empty or futures_bars.empty:
        raise RuntimeError(f'No matching NIFTY index/futures candles for {test_date}.')
    data = build_features(index_bars, futures_bars)
    trades, used_events = [], set()
    consecutive_losses, daily_r, blocked_until = 0, 0.0, -1
    last_entry_time = None
    for i in range(55,len(data)):
        row = data.iloc[i]
        minute = row.timestamp.hour*60+row.timestamp.minute
        if row.session != str(test_date) or not 570 <= minute <= 870 or i <= blocked_until: continue
        if (len(trades) >= CFG.maximum_trades or consecutive_losses >= CFG.maximum_daily_losses
                or daily_r <= -CFG.daily_max_drawdown_r): break
        if last_entry_time is not None and (row.timestamp-last_entry_time).total_seconds() < CFG.cooldown_minutes*60: continue
        setup = evaluate_baseline_setup(data,i)
        if setup is None or setup['event_time'] in used_events: continue
        exit_index, outcome, result_r = simulate_managed_exit(data,i,setup)
        result_r -= CFG.cost_r_per_trade
        trades.append({'time':setup['entry_time'].strftime('%H:%M'),'setup':setup['setup'],
            'side':setup['side'],'entry':round(setup['entry'],2),'stop':round(setup['stop'],2),
            'target':round(setup['target'],2),'confidence':setup['confidence_score'],
            'structure':setup['market_structure'],'regime':setup['market_regime'],
            'adx':round(setup['adx'],1),'atr_pct':round(setup['atr_percentile'],2),
            'poc':round(setup['poc'],2),'outcome':outcome,'result_r':round(result_r,2)})
        used_events.add(setup['event_time']); last_entry_time = row.timestamp
        blocked_until = exit_index; daily_r += result_r
        consecutive_losses = consecutive_losses+1 if result_r < 0 else 0
    result = pd.DataFrame(trades)
    print(f'INSTITUTIONAL NIFTY backtest | {future_name} | {test_date}')
    display(result)
    print(f"Trades {len(result)} | Net R {result.result_r.sum() if not result.empty else 0:.2f} | "
          f"Max consecutive losses {CFG.maximum_daily_losses} | DD stop {-CFG.daily_max_drawdown_r:.1f}R")
    return result

def backtest_mtf_specific_date(test_date):
    return backtest_specific_date(test_date)

backtest_trades = backtest_mtf_specific_date(TEST_DATE)
'''
nb["cells"][8]["source"] = source(cell8)

# Patch the live MTF loop to use managed risk limits and the institutional evaluator.
cell10 = "".join(nb["cells"][10]["source"])
cell10 = cell10.replace(
    "return len(trades) >= CFG.maximum_trades or losses >= CFG.maximum_daily_losses",
    "return (len(trades) >= CFG.maximum_trades or losses >= CFG.maximum_daily_losses or "
    "sum(float(t.get('result_r', 0)) for t in trades) <= -CFG.daily_max_drawdown_r)")
cell10 = cell10.replace("9 * 60 + 25 <= minute_of_day <= 14 * 60 + 30",
                        "9 * 60 + 30 <= minute_of_day <= 14 * 60 + 30")
cell10 = cell10.replace(
"""                stop_hit = row['low'] <= position['stop'] if position['direction']==1 else row['high'] >= position['stop']
                target_hit = row['high'] >= position['target'] if position['direction']==1 else row['low'] <= position['target']
                if stop_hit or target_hit:
                    outcome = 'LOSS' if stop_hit else 'WIN'
                    position.update(outcome=outcome, exit_time=row['timestamp'])
                    trades.append(position)
                    losses += int(outcome=='LOSS')""",
"""                direction, entry, risk = position['direction'], position['entry'], position['risk']
                position['best_price'] = (max(position.get('best_price', entry), float(row['high']))
                                          if direction == 1 else
                                          min(position.get('best_price', entry), float(row['low'])))
                favorable_r = direction*(position['best_price']-entry)/max(risk, .01)
                if favorable_r >= CFG.breakeven_at_r:
                    position['stop'] = max(position['stop'],entry) if direction == 1 else min(position['stop'],entry)
                if favorable_r >= CFG.trail_after_r:
                    trail = (position['best_price']-CFG.trail_atr*row['atr'] if direction == 1
                             else position['best_price']+CFG.trail_atr*row['atr'])
                    position['stop'] = max(position['stop'],trail) if direction == 1 else min(position['stop'],trail)
                stop_hit = row['low'] <= position['stop'] if direction==1 else row['high'] >= position['stop']
                target_hit = row['high'] >= position['target'] if direction==1 else row['low'] <= position['target']
                if stop_hit or target_hit:
                    exit_price = position['stop'] if stop_hit else position['target']
                    result_r = direction*(exit_price-entry)/max(risk,.01)-CFG.cost_r_per_trade
                    outcome = 'WIN' if result_r > 0 else ('BREAKEVEN' if result_r > -0.1 else 'LOSS')
                    position.update(outcome=outcome, exit_time=row['timestamp'],
                                    exit_price=exit_price, result_r=result_r)
                    trades.append(position)
                    losses = losses+1 if outcome=='LOSS' else 0""")
cell10 = cell10.replace("paper_trades = run_live_paper_trading()",
                        "paper_trades = run_live_mtf_paper_trading()")
live_institutional = r'''

def run_live_institutional_paper_trading() -> pd.DataFrame:
    """Unified closed-candle signal, option selection, ML and premium management."""
    trading_date = datetime.now().strftime('%Y-%m-%d')
    future_id, future_name = find_nearest_nifty_future(trading_date)
    print(f'INSTITUTIONAL PAPER MODE | NIFTY + {future_name} | no real orders')
    position, last_processed, last_entry, last_exit = None, None, None, None
    trades, used_events, consecutive_losses, daily_r = [], set(), 0, 0.0
    while True:
        try:
            now = pd.Timestamp.now(tz='Asia/Kolkata')
            if _paper_session_finished(now):
                print('\nMarket session finished.'); break
            if (len(trades) >= CFG.maximum_trades or
                    consecutive_losses >= CFG.maximum_daily_losses or
                    daily_r <= -CFG.daily_max_drawdown_r):
                print('\nDaily risk lock engaged.'); break
            index_bars = fetch_intraday('13','IDX_I','INDEX',trading_date)
            futures_bars = fetch_intraday(future_id,'NSE_FNO','FUTIDX',trading_date)
            if index_bars.empty or futures_bars.empty:
                print('Waiting for Dhan candles...',end='\r'); time.sleep(CFG.poll_seconds); continue
            closed = _closed_candles(build_features(index_bars,futures_bars),now)
            if len(closed) < 56:
                print('Waiting for warm-up...',end='\r'); time.sleep(CFG.poll_seconds); continue
            row, i = closed.iloc[-1], len(closed)-1
            if row.timestamp == last_processed:
                time.sleep(CFG.poll_seconds); continue
            last_processed, exited = row.timestamp, False
            if position is not None:
                mark = fetch_option_mark(position['security_id'])
                if mark <= 0:
                    print('Waiting for option quote...',end='\r'); time.sleep(CFG.poll_seconds); continue
                position['best_price'] = max(position.get('best_price',position['entry']),mark)
                favorable_r = (position['best_price']-position['entry'])/position['initial_risk']
                if favorable_r >= CFG.breakeven_at_r:
                    position['stop'] = max(position['stop'],position['entry'])
                if favorable_r >= CFG.trail_after_r:
                    # Premium trail scales with the initial option risk; no index/option unit mixing.
                    position['stop'] = max(position['stop'],
                        position['best_price']-CFG.trail_atr*position['initial_risk'])
                stop_hit, target_hit = mark <= position['stop'], mark >= position['target']
                if stop_hit or target_hit:
                    exit_price = position['stop'] if stop_hit else position['target']
                    result_r = (exit_price-position['entry'])/position['initial_risk']-CFG.cost_r_per_trade
                    outcome = 'WIN' if result_r > 0 else ('BREAKEVEN' if result_r > -.1 else 'LOSS')
                    position.update(exit_time=row.timestamp,exit_price=exit_price,
                                    outcome=outcome,result_r=result_r)
                    trades.append(position); daily_r += result_r
                    consecutive_losses = consecutive_losses+1 if outcome == 'LOSS' else 0
                    print(f"\nEXIT {outcome} {position['symbol']} | {result_r:.2f}R")
                    position, last_exit, exited = None, row.timestamp, True
            if (position is None and not exited and _inside_entry_window(row.timestamp)
                    and _cooldown_complete(row.timestamp,last_exit,CFG.exit_cooldown_minutes)
                    and _cooldown_complete(row.timestamp,last_entry,CFG.cooldown_minutes)):
                setup = evaluate_baseline_setup(closed,i)
                if setup is not None and setup['event_time'] not in used_events:
                    option = select_affordable_nifty_option(setup['direction'])
                    if option is not None:
                        ml = _ml_confirmation(index_bars,option,setup['direction'])
                        if ml['approved']:
                            entry = float(option['entry_price'])
                            risk_pct = float(np.clip(setup['risk']/setup['entry']*option['delta']*4,.15,.30))
                            initial_risk = entry*risk_pct
                            symbol = f"NIFTY {option['expiry']} {option['strike']:.0f} {option['option_type']}"
                            position = {**setup, **option, 'symbol':symbol, 'entry':entry,
                                'stop':entry-initial_risk,
                                'target':entry+initial_risk*setup['planned_r'],
                                'initial_risk':initial_risk, 'best_price':entry,
                                'underlying_entry':setup['entry'], 'ml_probability':ml['probability']}
                            used_events.add(setup['event_time']); last_entry = row.timestamp
                            print(f"\nPAPER ENTRY {symbol} | {setup['setup']} | "
                                  f"confidence {setup['confidence_score']:.1f} | ML {ml['probability']:.3f}")
                        else:
                            print(f"\nML SAFE BLOCK: {ml['reason']}")
            status = f"OPEN {position['symbol']}" if position else 'SCANNING'
            print(f"[{now:%H:%M:%S}] {status} | daily {daily_r:.2f}R",end='\r')
            time.sleep(CFG.poll_seconds)
        except KeyboardInterrupt:
            print('\nPaper trading stopped by user.'); break
        except Exception as error:
            print(f'\nError: {error}'); time.sleep(CFG.poll_seconds)
    return pd.DataFrame(trades)
'''
if "def run_live_institutional_paper_trading" not in cell10:
    cell10 = cell10.replace(
        "# Start CURRENT NIFTY OPTION paper trading (premium-based, no real orders):",
        live_institutional + "\n# Start CURRENT NIFTY OPTION paper trading (premium-based, no real orders):")
cell10 = cell10.replace("paper_trades = run_live_mtf_paper_trading()",
                        "paper_trades = run_live_institutional_paper_trading()")
nb["cells"][10]["source"] = source(cell10)

# Correct older generated copies too; harmless on a first run.
nb["cells"][4]["source"] = source("".join(nb["cells"][4]["source"]).replace(
    r"E:\NIFTY_AI_TRADER\data\processed\live_decision_state.parquet",
    r"E:\NIFTY_AI_TRADER\data\runtime\decision_observations.parquet"))

nb["cells"][0]["source"] = source("""# NIFTY 50 Institutional Intraday Options Scalping — Paper Trading Only

15m/5m/1m alignment, market structure, swing/ORB/VWAP/ADX/ATR, liquidity/fake-break,
volume profile, option-chain confirmation, frozen-ML gate and portfolio risk controls.
**No order-placement endpoint is used.**
""")

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK}")
print(f"Backup  {BACKUP}")
