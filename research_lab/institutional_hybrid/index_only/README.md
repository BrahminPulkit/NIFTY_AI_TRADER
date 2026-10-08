# Index-only signal research

Uses only the pinned five-year one-minute NIFTY 50 spot CSV. Options, rolling
premiums, OI, IV and strike data are completely excluded.

Methodology:

- causal close-of-bar signals;
- entry at the next contiguous minute's open;
- OHLC stop/target simulation with stop-first ambiguity handling;
- fixed costs and two-sided slippage;
- no synthetic candles and no overnight positions;
- corrupt source rows quarantined with evidence;
- gaps handled explicitly rather than forward-filled.

Because NIFTY spot is not directly tradable, results measure signal quality and
are not claims of executable futures or ETF performance.
