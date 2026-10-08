# Dhan Option Data Collection

## Runtime path

`DhanConnectionManager` remains the only owner of the authenticated Dhan
session. Its background worker obtains completed NIFTY ATM CE/PE candles and
option-chain quotes. `OptionMarketDataCollector` receives only a defensive
cache snapshot and cannot authenticate, reconnect, place orders, or call Dhan.

Each latest completed candle is stored under:

`data/collected/dhan_options/YYYY-MM-DD/nifty_atm_options.parquet`

Rows are keyed by candle timestamp and Dhan security ID. Repeated heartbeats
therefore update/deduplicate a row rather than growing duplicate data. Contract
identity includes security ID, expiry, strike, CE/PE type and exchange segment.
The daily manifest reports CE/PE counts and time coverage.

Quote fields (`ltp`, `open_interest`, `bid`, `ask`, `spread`) are nullable. The
collector never applies a current quote to older historical candles. Historical
OHLCV and OI contain only values returned by Dhan for that request.

## Historical backfill

Set `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN`, retain the required current or
archived Dhan security-master CSV files, then run:

```powershell
python -m commands.data.main_download_dhan_options --start 2026-07-01 --end 2026-08-01
```

The end date is exclusive. Downloads are validated, chunked, resumable and
written to `data/raw/dhan_option_chunks`. Failed requests retain their exact API
error in the manifest without exposing credentials.

Expired option history can only be resolved when archived security masters
contain those expired security IDs. A current master cannot reconstruct missing
expired contracts. Forward collection should therefore remain enabled every
market day to build reliable CE and PE coverage.

## Scope

This collection layer does not change the frozen CatBoost model, thresholds,
feature pipeline, Decision Engine, live signal direction or paper-trading
rules. PE model training and PE signal generation require a separate approved
model-development step after sufficient PE history has accumulated.
