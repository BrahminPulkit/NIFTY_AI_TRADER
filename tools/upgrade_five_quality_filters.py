"""Surgical quality-filter upgrade for the existing paper-trading notebook.

Preserves cell count/order/ids/metadata/outputs and edits only cells 3, 4, 8, 10.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
BACKUP = ROOT / "notebooks" / "NIFTY_Scalping_Paper_Trading.pre_five_filters.ipynb"


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, found {text.count(old)}")
    return text.replace(old, new, 1)


nb = json.loads(PATH.read_text(encoding="utf-8"))
if len(nb["cells"]) != 12:
    raise RuntimeError("Unexpected notebook shape; refusing to edit")
before_shape = [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs"))
                for c in nb["cells"]]
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Cell 3: only configuration thresholds required by the requested filters.
c3 = src(nb["cells"][3])
c3 = replace_once(c3, "    maximum_daily_losses: int = 2\n",
                  "    maximum_daily_losses: int = 3\n", "loss limit")
c3 = replace_once(c3, "    minimum_adx: float = 18.0\n",
                  "    minimum_adx: float = 22.0\n", "ADX threshold")
c3 = replace_once(c3, "    minimum_pcr: float = 0.70\n    maximum_pcr: float = 1.40\n",
                  "    minimum_pcr: float = 1.00\n    maximum_pcr: float = 1.00\n",
                  "PCR thresholds")
c3 = replace_once(
    c3,
    "    daily_max_drawdown_r: float = 2.0\n",
    "    daily_max_drawdown_r: float = 2.0\n"
    "    daily_max_drawdown_pct: float = 0.02\n"
    "    bb_period: int = 20\n"
    "    bb_width_multiplier: float = 2.0\n"
    "    minimum_bb_width: float = 0.0015\n"
    "    minimum_ema_distance_atr: float = 0.20\n"
    "    minimum_regime_atr_percentile: float = 0.25\n"
    "    vwap_slope_bars: int = 3\n",
    "regime configuration",
)
set_src(nb["cells"][3], c3)

# Cell 4: closed 15m structure, VWAP slope, four-factor regime, strict entry gates,
# and corrected option-chain OI changes.
c4 = src(nb["cells"][4])
old_mtf = """def _closed_mtf(frame, rule, prefix):
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
"""
new_mtf = """def _closed_mtf(frame, rule, prefix):
    \"\"\"Build completed higher-timeframe bars; 15m direction is pure HH/HL or LH/LL.\"\"\"
    bars = (frame.set_index('timestamp')[['open','high','low','close','futures_volume']]
            .resample(rule, label='right', closed='left')
            .agg({'open':'first','high':'max','low':'min','close':'last','futures_volume':'sum'})
            .dropna().reset_index())
    bars[f'{prefix}_fast'] = bars['close'].ewm(span=3 if prefix == 'trend15' else 5, adjust=False).mean()
    bars[f'{prefix}_slow'] = bars['close'].ewm(span=8 if prefix == 'trend15' else 13, adjust=False).mean()
    bars[f'{prefix}_slope'] = bars[f'{prefix}_slow'].diff()
    if prefix == 'trend15':
        bars[f'{prefix}_hh'] = bars['high'] > bars['high'].shift(1)
        bars[f'{prefix}_hl'] = bars['low'] > bars['low'].shift(1)
        bars[f'{prefix}_lh'] = bars['high'] < bars['high'].shift(1)
        bars[f'{prefix}_ll'] = bars['low'] < bars['low'].shift(1)
        bars[f'{prefix}_direction'] = np.select(
            [bars[f'{prefix}_hh'] & bars[f'{prefix}_hl'],
             bars[f'{prefix}_lh'] & bars[f'{prefix}_ll']], [1, -1], default=0)
        bars[f'{prefix}_structure'] = np.select(
            [bars[f'{prefix}_direction'] == 1, bars[f'{prefix}_direction'] == -1],
            ['HH_HL', 'LH_LL'], default='TRANSITION')
    else:
        bars[f'{prefix}_direction'] = np.select(
            [(bars[f'{prefix}_fast'] > bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] > 0),
             (bars[f'{prefix}_fast'] < bars[f'{prefix}_slow']) & (bars[f'{prefix}_slope'] < 0)],
            [1, -1], default=0)
    keep = ['timestamp',f'{prefix}_fast',f'{prefix}_slow',f'{prefix}_slope',f'{prefix}_direction']
    if prefix == 'trend15':
        keep.append(f'{prefix}_structure')
    return bars[keep]
"""
c4 = replace_once(c4, old_mtf, new_mtf, "closed MTF")

c4 = replace_once(
    c4,
    "    data['atr_percentile'] = (data['atr']-atr_min)/(atr_max-atr_min).replace(0, np.nan)\n",
    "    data['atr_percentile'] = (data['atr']-atr_min)/(atr_max-atr_min).replace(0, np.nan)\n"
    "    data['vwap_slope'] = data.groupby('session')['vwap'].diff(CFG.vwap_slope_bars)\n"
    "    bb_mid = data['close'].rolling(CFG.bb_period).mean()\n"
    "    bb_std = data['close'].rolling(CFG.bb_period).std(ddof=0)\n"
    "    data['bb_width'] = (2*CFG.bb_width_multiplier*bb_std)/bb_mid.replace(0,np.nan)\n"
    "    data['ema_distance_atr'] = (data.ema_fast-data.ema_slow).abs()/data.atr.replace(0,np.nan)\n",
    "VWAP and regime features",
)
old_regime = """    data['market_regime'] = np.select(
        [data.atr_percentile >= .80,
         (data.adx >= CFG.minimum_adx) & (data.trend5_direction != 0) & (data.trend15_direction != 0)],
        ['HIGH_VOLATILITY','TRENDING'], default='RANGE')
"""
new_regime = """    # Regime requires agreement across strength, volatility, expansion and EMA separation.
    regime_votes = (
        (data.adx > CFG.minimum_adx).astype(int)
        + (data.atr_percentile >= CFG.minimum_regime_atr_percentile).astype(int)
        + (data.bb_width >= CFG.minimum_bb_width).astype(int)
        + (data.ema_distance_atr >= CFG.minimum_ema_distance_atr).astype(int))
    data['regime_score'] = regime_votes
    data['market_regime'] = np.where(regime_votes >= 3, 'TRENDING', 'SIDEWAYS')
"""
c4 = replace_once(c4, old_regime, new_regime, "market regime")

old_chain = """    ce_oi = pe_oi = ce_change = pe_change = 0.0
    for legs in chain.get('oc', {}).values():
        ce, pe = legs.get('ce') or {}, legs.get('pe') or {}
        ce_oi += float(ce.get('oi') or 0); pe_oi += float(pe.get('oi') or 0)
        ce_change += float(ce.get('oi_change') or ce.get('change_in_oi') or 0)
        pe_change += float(pe.get('oi_change') or pe.get('change_in_oi') or 0)
    pcr = pe_oi/max(ce_oi, 1.0)
    call_writing, put_writing = ce_change > 0, pe_change > 0
    aligned = ((direction == 1 and put_writing and pcr >= CFG.minimum_pcr) or
               (direction == -1 and call_writing and pcr <= CFG.maximum_pcr))
"""
new_chain = """    ce_oi = pe_oi = ce_change = pe_change = 0.0
    for legs in chain.get('oc', {}).values():
        ce, pe = legs.get('ce') or {}, legs.get('pe') or {}
        current_ce, current_pe = float(ce.get('oi') or 0), float(pe.get('oi') or 0)
        previous_ce = float(ce.get('previous_oi') or current_ce)
        previous_pe = float(pe.get('previous_oi') or current_pe)
        ce_oi += current_ce; pe_oi += current_pe
        ce_change += current_ce-previous_ce
        pe_change += current_pe-previous_pe
    pcr = pe_oi/max(ce_oi, 1.0)
    call_writing, put_writing = ce_change > 0, pe_change > 0
    aligned = ((direction == 1 and pcr >= CFG.minimum_pcr
                and not call_writing and put_writing)
               or (direction == -1 and pcr <= CFG.maximum_pcr
                   and not put_writing and call_writing))
"""
c4 = replace_once(c4, old_chain, new_chain, "option chain")

old_gates = """    vwap_ok = ((row.close > row.vwap and row.vwap >= previous.vwap) if direction == 1
               else (row.close < row.vwap and row.vwap <= previous.vwap))
    adx_ok = (row.adx >= CFG.minimum_adx and
              (row.plus_di > row.minus_di if direction == 1 else row.minus_di > row.plus_di))
    atr_ok = CFG.minimum_atr_percentile <= row.atr_percentile <= CFG.maximum_atr_percentile
    trigger = (swing_break or swing_retest or orb_break or orb_retest or sweep or ema_reclaim or
               vwap_reclaim or institutional_momentum or supertrend_flip or supertrend_retest)
    if not (structure_ok and trigger and vwap_ok and adx_ok and atr_ok and not fake_against):
        return None
"""
new_gates = """    vwap_ok = ((row.close > row.vwap and row.vwap_slope > 0) if direction == 1
               else (row.close < row.vwap and row.vwap_slope < 0))
    adx_ok = (row.adx > CFG.minimum_adx and
              (row.plus_di > row.minus_di if direction == 1 else row.minus_di > row.plus_di))
    atr_ok = CFG.minimum_atr_percentile <= row.atr_percentile <= CFG.maximum_atr_percentile
    one_minute_confirmation = bool(
        row.close > row.open if direction == 1 else row.close < row.open)
    trending_trigger = (swing_break or swing_retest or orb_break or orb_retest or sweep
        or ema_reclaim or vwap_reclaim or institutional_momentum
        or supertrend_flip or supertrend_retest)
    sideways_trigger = sweep or orb_break or orb_retest
    regime_ok = (trending_trigger if row.market_regime == 'TRENDING' else sideways_trigger)
    if not (structure_ok and regime_ok and vwap_ok and adx_ok and atr_ok
            and one_minute_confirmation and not fake_against):
        return None
"""
c4 = replace_once(c4, old_gates, new_gates, "entry gates")
set_src(nb["cells"][4], c4)

# Cell 8: the existing replay keeps its R lock and now also audits the requested 2% capital lock.
c8 = src(nb["cells"][8])
c8 = replace_once(
    c8,
    "        risk_locked = (len(trades) >= CFG.maximum_trades or\n"
    "                       consecutive_losses >= CFG.maximum_daily_losses or\n"
    "                       daily_r <= -CFG.daily_max_drawdown_r)\n",
    "        drawdown_pct = max(0.0,-daily_r*CFG.cost_r_per_trade)\n"
    "        risk_locked = (len(trades) >= CFG.maximum_trades or\n"
    "                       consecutive_losses >= CFG.maximum_daily_losses or\n"
    "                       daily_r <= -CFG.daily_max_drawdown_r or\n"
    "                       drawdown_pct >= CFG.daily_max_drawdown_pct)\n",
    "backtest risk lock",
)
set_src(nb["cells"][8], c8)

# Cell 10: percentage-based paper-capital daily lock, retaining all existing exits/trailing.
c10 = src(nb["cells"][10])
c10 = replace_once(
    c10,
    "def _paper_risk_limit_reached(trades: list[dict], losses: int) -> bool:\n"
    "    \"\"\"Apply the existing trade-count and loss-count limits.\"\"\"\n"
    "    return (len(trades) >= CFG.maximum_trades or losses >= CFG.maximum_daily_losses or sum(float(t.get('result_r', 0)) for t in trades) <= -CFG.daily_max_drawdown_r)\n",
    "def _paper_risk_limit_reached(trades: list[dict], losses: int) -> bool:\n"
    "    \"\"\"Lock after trade/loss limits or a 2% realized paper-capital drawdown.\"\"\"\n"
    "    realized_pnl = sum(float(t.get('pnl', 0.0)) for t in trades)\n"
    "    return (len(trades) >= CFG.maximum_trades\n"
    "            or losses >= CFG.maximum_daily_losses\n"
    "            or realized_pnl <= -CFG.paper_capital*CFG.daily_max_drawdown_pct)\n",
    "paper risk helper",
)
c10 = replace_once(
    c10,
    "    trades, used_events, consecutive_losses, daily_r = [], set(), 0, 0.0\n"
    "    early_trades = 0\n",
    "    trades, used_events, consecutive_losses, daily_r = [], set(), 0, 0.0\n"
    "    daily_pnl = 0.0\n"
    "    early_trades = 0\n",
    "institutional pnl state",
)
c10 = replace_once(
    c10,
    "            if (len(trades) >= CFG.maximum_trades or\n"
    "                    consecutive_losses >= CFG.maximum_daily_losses or\n"
    "                    daily_r <= -CFG.daily_max_drawdown_r):\n",
    "            if (len(trades) >= CFG.maximum_trades or\n"
    "                    consecutive_losses >= CFG.maximum_daily_losses or\n"
    "                    daily_r <= -CFG.daily_max_drawdown_r or\n"
    "                    daily_pnl <= -CFG.paper_capital*CFG.daily_max_drawdown_pct):\n",
    "institutional daily lock",
)
c10 = replace_once(
    c10,
    "                    position.update(exit_time=row.timestamp,exit_price=exit_price,\n"
    "                                    outcome=outcome,result_r=result_r)\n"
    "                    trades.append(position); daily_r += result_r\n",
    "                    pnl = (exit_price-position['entry'])*position['lot_size']\n"
    "                    position.update(exit_time=row.timestamp,exit_price=exit_price,\n"
    "                                    outcome=outcome,result_r=result_r,pnl=pnl)\n"
    "                    trades.append(position); daily_r += result_r; daily_pnl += pnl\n",
    "institutional realized pnl",
)
set_src(nb["cells"][10], c10)

after_shape = [(c.get("id"), c["cell_type"], c.get("metadata"), c.get("outputs"))
               for c in nb["cells"]]
if before_shape != after_shape:
    raise RuntimeError("Cell structure/metadata/output mutation detected")
PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {PATH}")
print(f"Backup {BACKUP}")
