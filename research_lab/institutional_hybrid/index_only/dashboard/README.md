# Index Strategy Dashboard

Run locally:

```powershell
streamlit run index_only/dashboard/app.py
```

The dashboard reads only completed index-only journals. Starting capital may be
as low as ₹10. This is mathematical capital stress testing: NIFTY spot is not
directly tradable, and percentage returns scale with capital unless a fixed
rupee cost is supplied.

The Daily Multi-Strategy Replay tab provides a date picker, minute-by-minute
historical replay, NIFTY candlesticks, multi-strategy entries/exits, active
capital, running P&L, active positions, execution logs and trade-level details.
It is historical replay only, not a real-time market feed.
