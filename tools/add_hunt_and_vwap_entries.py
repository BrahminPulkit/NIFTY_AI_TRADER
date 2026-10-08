"""Add causal reversal-hunt and completed-5m VWAP-volume entries in place."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
backup = path.with_name("NIFTY_Scalping_Paper_Trading.pre_hunt_entries.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
shape = [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs")) for c in nb["cells"]]
if not backup.exists():
    backup.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

def replace(cell, old, new, label):
    text = "".join(nb["cells"][cell]["source"])
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, found {text.count(old)}")
    nb["cells"][cell]["source"] = text.replace(old, new, 1).splitlines(keepends=True)

replace(3,
"""    vwap_slope_bars: int = 3
""",
"""    vwap_slope_bars: int = 3
    five_min_vwap_volume_ratio: float = 1.30
    reversal_hunt_volume_ratio: float = 1.50
    bottom_hunt_max_rsi: float = 45.0
    top_hunt_min_rsi: float = 55.0
""", "hunt configuration")

# The higher-timeframe builder retains the final session VWAP in each completed 5m
# candle and compares its volume only with previously completed 5m candles.
replace(4,
"""    bars = (frame.set_index('timestamp')[['open','high','low','close','futures_volume']]
            .resample(rule, label='right', closed='left')
            .agg({'open':'first','high':'max','low':'min','close':'last','futures_volume':'sum'})
            .dropna().reset_index())
""",
"""    bars = (frame.set_index('timestamp')[['open','high','low','close','futures_volume','vwap']]
            .resample(rule, label='right', closed='left')
            .agg({'open':'first','high':'max','low':'min','close':'last',
                  'futures_volume':'sum','vwap':'last'})
            .dropna().reset_index())
""", "closed MTF VWAP")

replace(4,
"""    else:
        bars[f'{prefix}_direction'] = np.select(
            [(bars[f'{prefix}_fast'] > bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] > 0),
             (bars[f'{prefix}_fast'] < bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] < 0)],
            [1, -1], default=0)
    keep = ['timestamp',f'{prefix}_fast',f'{prefix}_slow',f'{prefix}_slope',f'{prefix}_direction']
    if prefix == 'trend15':
        keep.append(f'{prefix}_structure')
""",
"""    else:
        bars[f'{prefix}_direction'] = np.select(
            [(bars[f'{prefix}_fast'] > bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] > 0),
             (bars[f'{prefix}_fast'] < bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] < 0)],
            [1, -1], default=0)
        prior_volume = bars['futures_volume'].shift(1).rolling(CFG.volume_period,min_periods=5).mean()
        bars['trend5_volume_ratio'] = bars['futures_volume']/prior_volume.replace(0,np.nan)
        bars['trend5_vwap_cross_up'] = (
            (bars.close.shift(1) <= bars.vwap.shift(1)) & (bars.close > bars.vwap)
            & (bars.close > bars.open)
            & (bars.trend5_volume_ratio >= CFG.five_min_vwap_volume_ratio))
        bars['trend5_vwap_cross_down'] = (
            (bars.close.shift(1) >= bars.vwap.shift(1)) & (bars.close < bars.vwap)
            & (bars.close < bars.open)
            & (bars.trend5_volume_ratio >= CFG.five_min_vwap_volume_ratio))
    keep = ['timestamp',f'{prefix}_fast',f'{prefix}_slow',f'{prefix}_slope',f'{prefix}_direction']
    if prefix == 'trend15':
        keep.append(f'{prefix}_structure')
    else:
        keep.extend(['trend5_volume_ratio','trend5_vwap_cross_up','trend5_vwap_cross_down'])
""", "5m VWAP breakout")

# Define confirmed reversal hunts before the normal MTF veto. They still require a
# 1m EMA direction change, rejection candle, exhaustion, volume and later option-chain approval.
replace(4,
"""    mtf_ok = (direction != 0 and direction == int(row['trend5_direction'])
              and direction == int(row['trend15_direction']))
    supertrend_ok = int(row['supertrend_direction']) == direction
    if not mtf_ok:
        return None
""",
"""    mtf_ok = (direction != 0 and direction == int(row['trend5_direction'])
              and direction == int(row['trend15_direction']))
    supertrend_ok = int(row['supertrend_direction']) == direction
    bottom_hunt = bool(direction == 1 and row.liquidity_sweep_up
        and row.close > row.open and row.close > row.ema_fast
        and row.rsi <= CFG.bottom_hunt_max_rsi
        and row.volume_ratio >= CFG.reversal_hunt_volume_ratio)
    top_hunt = bool(direction == -1 and row.liquidity_sweep_down
        and row.close < row.open and row.close < row.ema_fast
        and row.rsi >= CFG.top_hunt_min_rsi
        and row.volume_ratio >= CFG.reversal_hunt_volume_ratio)
    reversal_hunt = bottom_hunt or top_hunt
    vwap5_breakout = bool(row.trend5_vwap_cross_up if direction == 1
                          else row.trend5_vwap_cross_down)
    # A 5m VWAP breakout may lead the slower 5m EMA turn, but must agree with 15m.
    vwap5_alignment = (direction != 0 and direction == int(row.trend15_direction)
                       and vwap5_breakout)
    if not (mtf_ok or reversal_hunt or vwap5_alignment):
        return None
""", "early reversal alignment")

replace(4,
"""    trending_trigger = (swing_break or swing_retest or orb_break or orb_retest or sweep
        or ema_reclaim or vwap_reclaim or institutional_momentum
        or supertrend_flip or supertrend_retest)
    sideways_trigger = sweep or orb_break or orb_retest
    regime_ok = (trending_trigger if row.market_regime == 'TRENDING' else sideways_trigger)
    if not (regime_ok and vwap_ok and adx_ok and atr_ok
            and one_minute_confirmation and not fake_against):
""",
"""    trending_trigger = (swing_break or swing_retest or orb_break or orb_retest or sweep
        or ema_reclaim or vwap_reclaim or institutional_momentum
        or supertrend_flip or supertrend_retest or vwap5_breakout or reversal_hunt)
    sideways_trigger = sweep or orb_break or orb_retest or reversal_hunt or vwap5_breakout
    regime_ok = (trending_trigger if row.market_regime == 'TRENDING' else sideways_trigger)
    # Hunts start below/above VWAP by design; their strict sweep/rejection/volume
    # confirmation replaces the normal VWAP-location gate.
    directional_adx = (row.plus_di > row.minus_di if direction == 1
                       else row.minus_di > row.plus_di)
    adx_ok = row.adx > CFG.minimum_adx and (directional_adx or reversal_hunt)
    if not (regime_ok and (vwap_ok or reversal_hunt) and adx_ok and atr_ok
            and one_minute_confirmation and not fake_against):
""", "hunt entry gates")

replace(4,
"""    setup_type = ('SWING_BREAK_RETEST' if swing_retest else 'LIQUIDITY_SWEEP' if sweep else
                  'ORB_RETEST' if orb_retest else 'SUPERTREND_RETEST' if supertrend_retest else
""",
"""    setup_type = ('BOTTOM_HUNT' if bottom_hunt else 'TOP_HUNT' if top_hunt else
                  '5M_VWAP_VOLUME_BREAKOUT' if vwap5_breakout else
                  'SWING_BREAK_RETEST' if swing_retest else 'LIQUIDITY_SWEEP' if sweep else
                  'ORB_RETEST' if orb_retest else 'SUPERTREND_RETEST' if supertrend_retest else
""", "setup labels")

replace(4,
"""    priority = {'SWING_BREAK_RETEST':1,'LIQUIDITY_SWEEP':2,'ORB_RETEST':3,
""",
"""    priority = {'BOTTOM_HUNT':1,'TOP_HUNT':1,'5M_VWAP_VOLUME_BREAKOUT':2,
                'SWING_BREAK_RETEST':3,'LIQUIDITY_SWEEP':4,'ORB_RETEST':5,
""", "setup priorities")
replace(4,
"""                'SUPERTREND_RETEST':4,'SWING_BREAK':5,'SUPERTREND_FLIP':6,
                'ORB':7,'VWAP_RECLAIM':8,'EMA_RECLAIM':9,
                'INSTITUTIONAL_MOMENTUM':10}[setup_type]
""",
"""                'SUPERTREND_RETEST':6,'SWING_BREAK':7,'SUPERTREND_FLIP':8,
                'ORB':9,'VWAP_RECLAIM':10,'EMA_RECLAIM':11,
                'INSTITUTIONAL_MOMENTUM':12}[setup_type]
""", "remaining priorities")

if shape != [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs"))
             for c in nb["cells"]]:
    raise RuntimeError("Notebook structure changed")
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
print(backup)
