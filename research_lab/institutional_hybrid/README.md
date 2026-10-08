# Institutional Hybrid Research Lab R2

An independent research framework for studying explosive short-horizon moves in
NIFTY futures and ATM option premiums. It is not a directional prediction or
execution system and has no broker integration.

## Isolation contract

- All code and generated artifacts live below this directory.
- No production module is imported. Market files supplied to the runner are
  read-only inputs; outputs are guarded to remain inside this lab.
- The lab never trains or loads the production CatBoost model and never calls
  Stage-1, Stage-2, the Decision Engine, paper/live trading, dashboards, or the
  production backtester.
- Put copied/sanitised research inputs in `datasets/`; never point output paths
  at production locations.

## Modules

`indicators/engines.py` implements causal regime, EMA pullback, liquidity sweep,
momentum, breakout and futures-volume features. `strategies/option_engines.py`
scores ATM tradability and 5–15 minute expansion capacity. The composite strategy
produces a 0–100 Institutional Confidence Score with the requested weights.
`backtests/research_backtester.py` is a standalone premium backtester with target,
stop, time exit, costs and cooldown. Reports are emitted as CSV, Markdown and
HTML; charts include EMAs, entries, rejected setups, ATR/range expansion,
institutional score and ATM premium.

## Input contracts

Futures CSV: `timestamp,open,high,low,close,volume`.

ATM options CSV (aligned at the same bar frequency):
`timestamp,bid,ask,volume,open_interest,delta,gamma,theta,iv,premium`.

Timestamps must identify the same exchange time convention. Choose the ATM
contract without future knowledge and preserve expired-contract boundaries.

## Run

From this directory:

```powershell
python run_experiment.py --futures datasets/futures.csv --options datasets/options.csv --experiment-id baseline_001
pytest -q tests
```

Each run creates a dedicated subfolder in `reports/` and `charts/`, plus a
manifest recording configuration and input paths. Never reuse an experiment ID
when preserving an earlier result matters.

## Research protocol

Use chronological train/calibration/test windows and purge overlapping 5–15
minute labels. Include spreads, brokerage, taxes, slippage, strike selection and
contract-roll rules. Compare against simple baselines, report confidence
intervals, and segment results by regime, time of day, expiry proximity and gap
days. Treat 65 as a hypothesis—not an optimized production threshold. Promotion
outside this lab requires a separate review and explicit authorization.

