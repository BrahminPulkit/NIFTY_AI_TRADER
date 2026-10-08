"""Reduce entry lag in the rule-based institutional NIFTY notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
BACKUP = ROOT / "backup" / "notebooks" / "NIFTY_Scalping_Paper_Trading.pre_early_entry.ipynb"
nb = json.loads(PATH.read_text(encoding="utf-8"))
if len(nb["cells"]) != 12:
    raise RuntimeError("Unexpected notebook shape; refusing to edit")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
# Always transform the known-good pre-change snapshot. This also makes retries atomic
# after an interrupted or failed validation run.
nb = json.loads(BACKUP.read_text(encoding="utf-8"))


def set_source(index: int, text: str) -> None:
    nb["cells"][index]["source"] = text.splitlines(keepends=True)


# Cell 3: retain all names and add explicit early-entry controls.
cell3 = "".join(nb["cells"][3]["source"])
marker = "    minimum_confidence_score: float = 70.0\n"
addition = """    minimum_confidence_score: float = 70.0
    minimum_warmup_bars: int = 30
    early_entry_confidence: float = 74.0
    reclaim_volume_ratio: float = 0.90
    breakout_close_buffer_atr: float = 0.03
    orb_retest_tolerance_atr: float = 0.12
    supertrend_period: int = 10
    supertrend_multiplier: float = 2.5
    supertrend_retest_tolerance_atr: float = 0.10
    swing_retest_valid_bars: int = 8
    swing_retest_tolerance_atr: float = 0.15
    swing_retest_min_body_ratio: float = 0.35
    first_hour_end_minute: int = 630
    first_hour_min_confidence: float = 75.0
    maximum_first_hour_trades: int = 1
    opportunity_cluster_minutes: int = 5
    maximum_holding_bars: int = 35
"""
if "minimum_warmup_bars" not in cell3:
    if marker not in cell3:
        raise RuntimeError("Configuration insertion point not found")
    cell3 = cell3.replace(marker, addition)
cell3 = cell3.replace("    cooldown_minutes: int = 12\n",
                      "    cooldown_minutes: int = 6\n")
set_source(3, cell3)

# Cell 4: replace the simplistic pivot comparison with distinct confirmed swings.
cell4 = "".join(nb["cells"][4]["source"])
supertrend_helper = '''def _supertrend(frame, period=10, multiplier=2.5):
    """Return causal Supertrend line and direction using closed 1-minute bars."""
    previous_close = frame['close'].shift(1)
    true_range = pd.concat([
        frame['high']-frame['low'], (frame['high']-previous_close).abs(),
        (frame['low']-previous_close).abs()], axis=1).max(axis=1)
    atr = _wilder(true_range, period)
    midpoint = (frame['high']+frame['low'])/2
    basic_upper, basic_lower = midpoint+multiplier*atr, midpoint-multiplier*atr
    final_upper, final_lower = basic_upper.copy(), basic_lower.copy()
    trend = pd.Series(1, index=frame.index, dtype=int)
    line = pd.Series(np.nan, index=frame.index, dtype=float)
    started = False
    for i in range(len(frame)):
        if pd.isna(atr.iloc[i]):
            continue
        if not started:
            final_upper.iloc[i], final_lower.iloc[i] = basic_upper.iloc[i], basic_lower.iloc[i]
            trend.iloc[i] = 1 if frame.close.iloc[i] >= midpoint.iloc[i] else -1
            line.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
            started = True
            continue
        final_upper.iloc[i] = (basic_upper.iloc[i] if
            basic_upper.iloc[i] < final_upper.iloc[i-1] or frame.close.iloc[i-1] > final_upper.iloc[i-1]
            else final_upper.iloc[i-1])
        final_lower.iloc[i] = (basic_lower.iloc[i] if
            basic_lower.iloc[i] > final_lower.iloc[i-1] or frame.close.iloc[i-1] < final_lower.iloc[i-1]
            else final_lower.iloc[i-1])
        prior_trend = trend.iloc[i-1]
        if frame.close.iloc[i] > final_upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif frame.close.iloc[i] < final_lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prior_trend
        line.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
    return line, trend

'''
if "def _supertrend(" not in cell4:
    active_builder = cell4.rfind("def build_features(index_bars, futures_bars):")
    if active_builder < 0:
        raise RuntimeError("Active feature builder not found")
    cell4 = cell4[:active_builder] + supertrend_helper + cell4[active_builder:]

adx_marker = """    data['adx'], data['plus_di'], data['minus_di'] = adx, plus_di, minus_di
"""
adx_upgrade = """    data['adx'], data['plus_di'], data['minus_di'] = adx, plus_di, minus_di
    data['supertrend'], data['supertrend_direction'] = _supertrend(
        data, CFG.supertrend_period, CFG.supertrend_multiplier)
    data['supertrend_flip_up'] = ((data.supertrend_direction == 1) &
                                  (data.supertrend_direction.shift(1) == -1))
    data['supertrend_flip_down'] = ((data.supertrend_direction == -1) &
                                    (data.supertrend_direction.shift(1) == 1))
    data['supertrend_retest_up'] = ((data.supertrend_direction == 1) &
        (data.low <= data.supertrend+CFG.supertrend_retest_tolerance_atr*data.atr) &
        (data.close > data.supertrend) & (data.close > data.open))
    data['supertrend_retest_down'] = ((data.supertrend_direction == -1) &
        (data.high >= data.supertrend-CFG.supertrend_retest_tolerance_atr*data.atr) &
        (data.close < data.supertrend) & (data.close < data.open))
"""
if "supertrend_flip_up" not in cell4:
    if adx_marker not in cell4:
        raise RuntimeError("ADX feature insertion point not found")
    cell4 = cell4.replace(adx_marker, adx_upgrade, 1)
old_structure = """    data['swing_high'] = confirmed_high.ffill()
    data['swing_low'] = confirmed_low.ffill()
    data['previous_swing_high'] = confirmed_high.ffill().shift(1)
    data['previous_swing_low'] = confirmed_low.ffill().shift(1)
    data['market_structure'] = np.select(
        [(data.swing_high > data.previous_swing_high) & (data.swing_low >= data.previous_swing_low),
         (data.swing_low < data.previous_swing_low) & (data.swing_high <= data.previous_swing_high)],
        ['HH_HL','LH_LL'], default='TRANSITION')
    data['swing_break_up'] = (data.close > data.swing_high.shift(1)) & (data.close.shift(1) <= data.swing_high.shift(1))
    data['swing_break_down'] = (data.close < data.swing_low.shift(1)) & (data.close.shift(1) >= data.swing_low.shift(1))
"""
new_structure = """    # Compare distinct confirmed pivots, not a forward-filled level with its prior row.
    # This keeps HH/HL/LH/LL state stable between swing events and remains causal.
    high_state, low_state, last_high, last_low = [], [], None, None
    current_high_state, current_low_state = 'UNKNOWN', 'UNKNOWN'
    for high_value, low_value in zip(confirmed_high, confirmed_low):
        if pd.notna(high_value):
            current_high_state = ('HH' if last_high is not None and high_value > last_high
                                  else ('LH' if last_high is not None else 'UNKNOWN'))
            last_high = float(high_value)
        if pd.notna(low_value):
            current_low_state = ('HL' if last_low is not None and low_value > last_low
                                 else ('LL' if last_low is not None else 'UNKNOWN'))
            last_low = float(low_value)
        high_state.append(current_high_state); low_state.append(current_low_state)
    data['swing_high'] = confirmed_high.ffill()
    data['swing_low'] = confirmed_low.ffill()
    data['previous_swing_high'] = data['swing_high'].shift(1)
    data['previous_swing_low'] = data['swing_low'].shift(1)
    data['high_structure'], data['low_structure'] = high_state, low_state
    data['structure_direction'] = np.select(
        [(data.high_structure == 'HH') & (data.low_structure == 'HL'),
         (data.high_structure == 'LH') & (data.low_structure == 'LL')],
        [1, -1], default=0)
    data['market_structure'] = np.select(
        [data.structure_direction == 1, data.structure_direction == -1],
        ['HH_HL','LH_LL'], default=(data.high_structure + '_' + data.low_structure))
    prior_high, prior_low = data.swing_high.shift(1), data.swing_low.shift(1)
    data['swing_break_up'] = ((data.high > prior_high) &
        (data.close >= prior_high-CFG.breakout_close_buffer_atr*data.atr) &
        (data.close.shift(1) <= prior_high))
    data['swing_break_down'] = ((data.low < prior_low) &
        (data.close <= prior_low+CFG.breakout_close_buffer_atr*data.atr) &
        (data.close.shift(1) >= prior_low))
"""
if old_structure not in cell4:
    raise RuntimeError("Expected structure block not found")
cell4 = cell4.replace(old_structure, new_structure)

retest_marker = """    data['swing_break_down'] = ((data.low < prior_low) &
        (data.close <= prior_low+CFG.breakout_close_buffer_atr*data.atr) &
        (data.close.shift(1) >= prior_low))
"""
retest_upgrade = """    data['swing_break_down'] = ((data.low < prior_low) &
        (data.close <= prior_low+CFG.breakout_close_buffer_atr*data.atr) &
        (data.close.shift(1) >= prior_low))
    # Remember each broken swing for a limited number of subsequent closed candles.
    # Fire once on the first controlled rejection, then retire the level.
    retest_up, retest_down = np.zeros(len(data),dtype=bool), np.zeros(len(data),dtype=bool)
    broken_up, broken_down = np.full(len(data),np.nan), np.full(len(data),np.nan)
    active_up, active_down, age_up, age_down = None, None, 0, 0
    for position in range(len(data)):
        if bool(data.swing_break_up.iloc[position]) and pd.notna(prior_high.iloc[position]):
            active_up, age_up = float(prior_high.iloc[position]), 0
        elif active_up is not None:
            age_up += 1
            if age_up > CFG.swing_retest_valid_bars:
                active_up = None
        if bool(data.swing_break_down.iloc[position]) and pd.notna(prior_low.iloc[position]):
            active_down, age_down = float(prior_low.iloc[position]), 0
        elif active_down is not None:
            age_down += 1
            if age_down > CFG.swing_retest_valid_bars:
                active_down = None
        candle_range = max(float(data.high.iloc[position]-data.low.iloc[position]),.01)
        body_ratio = abs(float(data.close.iloc[position]-data.open.iloc[position]))/candle_range
        if active_up is not None and age_up > 0:
            tolerance = CFG.swing_retest_tolerance_atr*float(data.atr.iloc[position])
            retest_up[position] = bool(
                data.low.iloc[position] <= active_up+tolerance and
                data.close.iloc[position] > active_up and
                data.close.iloc[position] > data.open.iloc[position] and
                body_ratio >= CFG.swing_retest_min_body_ratio)
            broken_up[position] = active_up
            if retest_up[position]:
                active_up = None
        if active_down is not None and age_down > 0:
            tolerance = CFG.swing_retest_tolerance_atr*float(data.atr.iloc[position])
            retest_down[position] = bool(
                data.high.iloc[position] >= active_down-tolerance and
                data.close.iloc[position] < active_down and
                data.close.iloc[position] < data.open.iloc[position] and
                body_ratio >= CFG.swing_retest_min_body_ratio)
            broken_down[position] = active_down
            if retest_down[position]:
                active_down = None
    data['swing_break_retest_up'], data['swing_break_retest_down'] = retest_up, retest_down
    data['broken_swing_high'], data['broken_swing_low'] = broken_up, broken_down
"""
if "swing_break_retest_up" not in cell4:
    if retest_marker not in cell4:
        raise RuntimeError("Swing-retest insertion point not found")
    cell4 = cell4.replace(retest_marker, retest_upgrade, 1)

# Add causal retest/reclaim flags after ORB construction.
orb_marker = """    data['orb_break_down'] = (session_minute >= 570) & (data.close < data.orb_low - CFG.minimum_orb_buffer_atr*data.atr)
"""
orb_upgrade = """    data['orb_break_down'] = (session_minute >= 570) & (data.close < data.orb_low - CFG.minimum_orb_buffer_atr*data.atr)
    prior_orb_up = data.groupby('session')['orb_break_up'].transform(
        lambda values: values.cummax().shift(1, fill_value=False))
    prior_orb_down = data.groupby('session')['orb_break_down'].transform(
        lambda values: values.cummax().shift(1, fill_value=False))
    data['orb_retest_up'] = (prior_orb_up & (data.low <= data.orb_high+CFG.orb_retest_tolerance_atr*data.atr)
        & (data.close > data.orb_high) & (data.close > data.open))
    data['orb_retest_down'] = (prior_orb_down & (data.high >= data.orb_low-CFG.orb_retest_tolerance_atr*data.atr)
        & (data.close < data.orb_low) & (data.close < data.open))
    data['ema_reclaim_up'] = ((data.low <= data.ema_fast) & (data.close > data.ema_fast)
        & (data.close > data.open) & (data.volume_ratio >= CFG.reclaim_volume_ratio))
    data['ema_reclaim_down'] = ((data.high >= data.ema_fast) & (data.close < data.ema_fast)
        & (data.close < data.open) & (data.volume_ratio >= CFG.reclaim_volume_ratio))
    data['vwap_reclaim_up'] = ((data.close.shift(1) <= data.vwap.shift(1)) & (data.close > data.vwap)
        & (data.close > data.open) & (data.volume_ratio >= CFG.reclaim_volume_ratio))
    data['vwap_reclaim_down'] = ((data.close.shift(1) >= data.vwap.shift(1)) & (data.close < data.vwap)
        & (data.close < data.open) & (data.volume_ratio >= CFG.reclaim_volume_ratio))
"""
if "orb_retest_up" not in cell4:
    if orb_marker not in cell4:
        raise RuntimeError("ORB insertion point not found")
    cell4 = cell4.replace(orb_marker, orb_upgrade)

start = cell4.rfind("def evaluate_baseline_setup(data, i):")
end = cell4.find("def simulate_managed_exit(", start)
if start < 0 or end < 0:
    raise RuntimeError("Evaluator boundaries not found")
evaluator = '''def evaluate_baseline_setup(data, i):
    """Earlier causal entries with institutional alignment and tiered confirmation."""
    if i < CFG.minimum_warmup_bars:
        return None
    row, previous = data.iloc[i], data.iloc[i-1]
    required = ['atr','adx','vwap','trend5_direction','trend15_direction','atr_percentile']
    if any(pd.isna(row.get(name)) for name in required):
        return None
    direction = int(row['trend1_direction'])
    mtf_ok = (direction != 0 and direction == int(row['trend5_direction'])
              and direction == int(row['trend15_direction']))
    supertrend_ok = int(row['supertrend_direction']) == direction
    if not mtf_ok or not supertrend_ok:
        return None
    structure_direction = int(row.get('structure_direction', 0))
    # Mature trends require complete HH/HL or LH/LL. During a new trend transition,
    # accept the first supportive distinct pivot but demand a higher confidence score.
    supportive_pivot = ((row.high_structure == 'HH' or row.low_structure == 'HL')
                        if direction == 1 else
                        (row.high_structure == 'LH' or row.low_structure == 'LL'))
    opposing_pivot = ((row.high_structure == 'LH' and row.low_structure == 'LL')
                      if direction == 1 else
                      (row.high_structure == 'HH' and row.low_structure == 'HL'))
    structure_ok = structure_direction == direction or (supportive_pivot and not opposing_pivot)
    raw_swing_break = bool(row.swing_break_up if direction == 1 else row.swing_break_down)
    # A direct breakout is allowed only when the candle itself is institutional;
    # normal breakouts must prove acceptance through a subsequent retest.
    swing_break = bool(raw_swing_break and row.institutional_candle and
                       row.volume_ratio >= CFG.institutional_volume_ratio)
    swing_retest = bool(row.swing_break_retest_up if direction == 1
                        else row.swing_break_retest_down)
    orb_break = bool(row.orb_break_up if direction == 1 else row.orb_break_down)
    orb_retest = bool(row.orb_retest_up if direction == 1 else row.orb_retest_down)
    sweep = bool(row.liquidity_sweep_up if direction == 1 else row.liquidity_sweep_down)
    ema_reclaim = bool(row.ema_reclaim_up if direction == 1 else row.ema_reclaim_down)
    vwap_reclaim = bool(row.vwap_reclaim_up if direction == 1 else row.vwap_reclaim_down)
    institutional_momentum = bool(row.institutional_candle and
        (row.close > row.open if direction == 1 else row.close < row.open))
    supertrend_flip = bool(row.supertrend_flip_up if direction == 1 else row.supertrend_flip_down)
    supertrend_retest = bool(row.supertrend_retest_up if direction == 1 else row.supertrend_retest_down)
    fake_against = bool(row.fake_break_up if direction == 1 else row.fake_break_down)
    vwap_ok = ((row.close > row.vwap and row.vwap >= previous.vwap) if direction == 1
               else (row.close < row.vwap and row.vwap <= previous.vwap))
    adx_ok = (row.adx >= CFG.minimum_adx and
              (row.plus_di > row.minus_di if direction == 1 else row.minus_di > row.plus_di))
    atr_ok = CFG.minimum_atr_percentile <= row.atr_percentile <= CFG.maximum_atr_percentile
    trigger = (swing_break or swing_retest or orb_break or orb_retest or sweep or ema_reclaim or
               vwap_reclaim or institutional_momentum or supertrend_flip or supertrend_retest)
    if not (structure_ok and trigger and vwap_ok and adx_ok and atr_ok and not fake_against):
        return None
    atr, entry = float(row.atr), float(row.close)
    support = float(row.dynamic_support if direction == 1 else row.dynamic_resistance)
    raw_stop = support-direction*CFG.swing_buffer_atr*atr
    risk = direction*(entry-raw_stop)
    if swing_retest:
        broken_level = float(row.broken_swing_high if direction == 1 else row.broken_swing_low)
        retest_invalidation = (min(float(row.low),broken_level)-CFG.swing_buffer_atr*atr
                               if direction == 1 else
                               max(float(row.high),broken_level)+CFG.swing_buffer_atr*atr)
        raw_stop = retest_invalidation
        risk = direction*(entry-raw_stop)
    if risk <= 0 or risk > CFG.maximum_stop_atr*atr:
        risk = CFG.minimum_stop_atr*atr
    risk = max(risk, CFG.minimum_stop_atr*atr)
    opposing = float(row.dynamic_resistance if direction == 1 else row.dynamic_support)
    room_r = direction*(opposing-entry)/max(risk,.01)
    planned_r = min(CFG.target_r,room_r) if room_r >= CFG.minimum_room_r else CFG.target_r
    early_trigger = (ema_reclaim or vwap_reclaim or institutional_momentum or
                     supertrend_flip or supertrend_retest)
    score_parts = {
        'mtf':20.0, 'supertrend':10.0, 'structure':15.0 if structure_direction == direction else 10.0,
        'trigger':15.0 if (swing_break or swing_retest or orb_retest or sweep) else 11.0,
        'vwap':10.0, 'adx':10.0, 'atr':5.0,
        'volume':10.0*float(np.clip(row.volume_ratio/CFG.institutional_volume_ratio,0,1)),
        'institutional_candle':5.0 if row.institutional_candle else 0.0,
        'volume_profile':5.0 if (row.close >= row.poc if direction == 1 else row.close <= row.poc) else 0.0}
    confidence = round(min(100.0,sum(score_parts.values())),1)
    setup_type = ('SWING_BREAK_RETEST' if swing_retest else 'LIQUIDITY_SWEEP' if sweep else
                  'ORB_RETEST' if orb_retest else 'SUPERTREND_RETEST' if supertrend_retest else
                  'SUPERTREND_FLIP' if supertrend_flip else
                  'SWING_BREAK' if swing_break else 'ORB' if orb_break else
                  'VWAP_RECLAIM' if vwap_reclaim else 'EMA_RECLAIM' if ema_reclaim
                  else 'INSTITUTIONAL_MOMENTUM')
    priority = {'SWING_BREAK_RETEST':1,'LIQUIDITY_SWEEP':2,'ORB_RETEST':3,
                'SUPERTREND_RETEST':4,'SWING_BREAK':5,'SUPERTREND_FLIP':6,
                'ORB':7,'VWAP_RECLAIM':8,'EMA_RECLAIM':9,
                'INSTITUTIONAL_MOMENTUM':10}[setup_type]
    minute = row.timestamp.hour*60+row.timestamp.minute
    first_hour = minute < CFG.first_hour_end_minute
    weak_early_reclaim = first_hour and setup_type in {'EMA_RECLAIM','VWAP_RECLAIM'}
    threshold = (CFG.first_hour_min_confidence if first_hour else
                 (CFG.early_entry_confidence if early_trigger and
                  not (swing_break or swing_retest or orb_retest or sweep)
                  else CFG.minimum_confidence_score))
    if confidence < threshold or weak_early_reclaim:
        return None
    return {'event_time':row.timestamp,'direction':direction,
        'side':'BUY CE' if direction == 1 else 'BUY PE','setup':setup_type,
        'entry':entry,'stop':entry-direction*risk,'target':entry+direction*risk*planned_r,
        'risk':risk,'planned_r':planned_r,'entry_time':row.timestamp,
        'rsi':float(row.rsi),'gap_atr':float(row.gap_atr),'volume_ratio':float(row.volume_ratio),
        'confidence_score':confidence,'score_components':score_parts,'setup_priority':priority,
        'market_structure':row.market_structure,'market_regime':row.market_regime,
        'adx':float(row.adx),'atr_percentile':float(row.atr_percentile),
        'poc':float(row.poc),'vah':float(row.vah),'val':float(row.val),'room_r':room_r}

'''
cell4 = cell4[:start] + evaluator + cell4[end:]
set_source(4, cell4)

# Cells 2, 8 and 10 must use the same reduced but indicator-safe warm-up.
cell2 = "".join(nb["cells"][2]["source"])
cell2 = cell2.replace("if len(closed) >= 56 else None",
                      "if len(closed) > CFG.minimum_warmup_bars else None")
set_source(2, cell2)

cell8 = '''# Backtest the prior session by default; change TEST_DATE as needed.
TEST_DATE = (pd.Timestamp.now(tz='Asia/Kolkata')-pd.Timedelta(days=1)).strftime('%Y-%m-%d')

def backtest_specific_date(test_date):
    """Replay all qualified opportunities while enforcing execution risk limits."""
    global backtest_opportunities
    future_id, future_name = find_nearest_nifty_future(test_date)
    index_bars = fetch_intraday('13','IDX_I','INDEX',test_date)
    futures_bars = fetch_intraday(future_id,'NSE_FNO','FUTIDX',test_date)
    if index_bars.empty or futures_bars.empty:
        raise RuntimeError(f'No matching NIFTY index/futures candles for {test_date}.')
    data = build_features(index_bars,futures_bars)
    trades, opportunities, used_events = [], [], set()
    consecutive_losses, daily_r, blocked_until = 0, 0.0, -1
    last_entry_time, last_candidate_time, last_candidate_direction = None, None, None
    early_trades = 0
    for i in range(CFG.minimum_warmup_bars,len(data)):
        row = data.iloc[i]
        minute = row.timestamp.hour*60+row.timestamp.minute
        if row.session != str(test_date) or not 570 <= minute <= 870:
            continue
        setup = evaluate_baseline_setup(data,i)
        if setup is None or setup['event_time'] in used_events:
            continue
        clustered = (last_candidate_time is not None and
            setup['direction'] == last_candidate_direction and
            (row.timestamp-last_candidate_time).total_seconds() <
            CFG.opportunity_cluster_minutes*60)
        if clustered:
            continue
        last_candidate_time, last_candidate_direction = row.timestamp, setup['direction']
        used_events.add(setup['event_time'])
        risk_locked = (len(trades) >= CFG.maximum_trades or
                       consecutive_losses >= CFG.maximum_daily_losses or
                       daily_r <= -CFG.daily_max_drawdown_r)
        early_locked = (minute < CFG.first_hour_end_minute and
                        early_trades >= CFG.maximum_first_hour_trades)
        position_locked = i <= blocked_until
        cooldown_locked = (last_entry_time is not None and
            (row.timestamp-last_entry_time).total_seconds() < CFG.cooldown_minutes*60)
        status = ('SKIPPED_RISK_LOCK' if risk_locked else
                  'SKIPPED_POSITION_OPEN' if position_locked else
                  'SKIPPED_FIRST_HOUR_CAP' if early_locked else
                  'SKIPPED_COOLDOWN' if cooldown_locked else 'EXECUTED')
        opportunity = {'time':row.timestamp.strftime('%H:%M'),'setup':setup['setup'],
            'priority':setup['setup_priority'],'side':setup['side'],
            'confidence':setup['confidence_score'],'status':status}
        if status != 'EXECUTED':
            opportunities.append(opportunity)
            continue
        exit_index, outcome, result_r = simulate_managed_exit(
            data,i,setup,max_bars=CFG.maximum_holding_bars)
        result_r -= CFG.cost_r_per_trade
        trade = {'time':setup['entry_time'].strftime('%H:%M'),'setup':setup['setup'],
            'priority':setup['setup_priority'],'side':setup['side'],
            'entry':round(setup['entry'],2),'stop':round(setup['stop'],2),
            'target':round(setup['target'],2),'confidence':setup['confidence_score'],
            'structure':setup['market_structure'],'regime':setup['market_regime'],
            'adx':round(setup['adx'],1),'atr_pct':round(setup['atr_percentile'],2),
            'poc':round(setup['poc'],2),'outcome':outcome,'result_r':round(result_r,2)}
        trades.append(trade)
        opportunity.update(outcome=outcome,result_r=round(result_r,2))
        opportunities.append(opportunity)
        last_entry_time, blocked_until = row.timestamp, exit_index
        daily_r += result_r
        consecutive_losses = consecutive_losses+1 if result_r < 0 else 0
        early_trades += int(minute < CFG.first_hour_end_minute)
    result, backtest_opportunities = pd.DataFrame(trades), pd.DataFrame(opportunities)
    print(f'INSTITUTIONAL NIFTY backtest | {future_name} | {test_date}')
    display(result)
    print(f"Qualified opportunities {len(backtest_opportunities)} | Executed {len(result)} | "
          f"Net R {result.result_r.sum() if not result.empty else 0:.2f}")
    if not backtest_opportunities.empty:
        print('Opportunity status:',backtest_opportunities.status.value_counts().to_dict())
        display(backtest_opportunities)
    return result

def backtest_mtf_specific_date(test_date):
    return backtest_specific_date(test_date)

backtest_trades = backtest_mtf_specific_date(TEST_DATE)
'''
set_source(8, cell8)

cell10 = "".join(nb["cells"][10]["source"])
cell10 = cell10.replace("if len(closed) < 56:", "if len(closed) <= CFG.minimum_warmup_bars:")
cell10 = cell10.replace("if len(closed) < 56:", "if len(closed) <= CFG.minimum_warmup_bars:")
cell10 = cell10.replace(
    "    trades, used_events, consecutive_losses, daily_r = [], set(), 0, 0.0\n",
    "    trades, used_events, consecutive_losses, daily_r = [], set(), 0, 0.0\n"
    "    early_trades = 0\n")
cell10 = cell10.replace(
"""                position['best_price'] = max(position.get('best_price',position['entry']),mark)
                favorable_r = (position['best_price']-position['entry'])/position['initial_risk']
""",
"""                position['best_price'] = max(position.get('best_price',position['entry']),mark)
                position['bars_held'] = position.get('bars_held',0)+1
                favorable_r = (position['best_price']-position['entry'])/position['initial_risk']
""")
cell10 = cell10.replace(
"""                stop_hit, target_hit = mark <= position['stop'], mark >= position['target']
                if stop_hit or target_hit:
                    exit_price = position['stop'] if stop_hit else position['target']
""",
"""                stop_hit, target_hit = mark <= position['stop'], mark >= position['target']
                time_hit = position['bars_held'] >= CFG.maximum_holding_bars
                if stop_hit or target_hit or time_hit:
                    exit_price = (position['stop'] if stop_hit else
                                  position['target'] if target_hit else mark)
""")
cell10 = cell10.replace(
"""            if (position is None and not exited and _inside_entry_window(row.timestamp)
                    and _cooldown_complete(row.timestamp,last_exit,CFG.exit_cooldown_minutes)
                    and _cooldown_complete(row.timestamp,last_entry,CFG.cooldown_minutes)):
""",
"""            minute = row.timestamp.hour*60+row.timestamp.minute
            first_hour_available = (minute >= CFG.first_hour_end_minute or
                                    early_trades < CFG.maximum_first_hour_trades)
            if (position is None and not exited and first_hour_available
                    and _inside_entry_window(row.timestamp)
                    and _cooldown_complete(row.timestamp,last_exit,CFG.exit_cooldown_minutes)
                    and _cooldown_complete(row.timestamp,last_entry,CFG.cooldown_minutes)):
""")
cell10 = cell10.replace(
"""                            'initial_risk':initial_risk, 'best_price':entry,
                            'underlying_entry':setup['entry']}
                        used_events.add(setup['event_time']); last_entry = row.timestamp
""",
"""                            'initial_risk':initial_risk, 'best_price':entry,
                            'bars_held':0, 'underlying_entry':setup['entry']}
                        used_events.add(setup['event_time']); last_entry = row.timestamp
                        early_trades += int(minute < CFG.first_hour_end_minute)
""")
set_source(10, cell10)

PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Improved early-entry logic in {PATH}")
print(f"Backup saved to {BACKUP}")
