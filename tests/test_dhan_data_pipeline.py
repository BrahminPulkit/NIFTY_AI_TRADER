import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.continuous_futures_builder import trading_day_roll_date
from src.dhan_data_downloader import (
    DhanAPIError, DhanHistoricalDownloader, date_chunks, discover_nifty_futures,
    discover_nifty_options, plan_contract_requests, response_to_frame,
    validate_chunk,
)


class DhanDataPipelineTests(unittest.TestCase):
    def test_discovers_contract_aware_ce_and_pe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.csv"
            base = {
                "EXCH_ID": "NSE", "SEGMENT": "D", "INSTRUMENT": "OPTIDX",
                "UNDERLYING_SYMBOL": "NIFTY", "SM_EXPIRY_DATE": "2026-07-30",
                "STRIKE_PRICE": 25000, "DISPLAY_NAME": "NIFTY OPTION",
            }
            pd.DataFrame([
                {**base, "SECURITY_ID": 11, "OPTION_TYPE": "CE"},
                {**base, "SECURITY_ID": 12, "OPTION_TYPE": "PE"},
                {**base, "SECURITY_ID": 13, "OPTION_TYPE": "XX"},
            ]).to_csv(path, index=False)
            config = {"exchange_id": "NSE", "master_segment": "D",
                      "underlying_symbol": "NIFTY"}
            found = discover_nifty_options([str(path)], config)
            self.assertEqual(set(found.option_type), {"CE", "PE"})
            self.assertEqual(set(found.security_id), {"11", "12"})
            self.assertEqual(found.strike.unique().tolist(), [25000])

    def test_discovers_only_nifty_futidx(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.csv"
            pd.DataFrame([
                {"EXCH_ID": "NSE", "SEGMENT": "D", "SECURITY_ID": 1, "INSTRUMENT": "FUTIDX", "UNDERLYING_SYMBOL": "NIFTY", "SM_EXPIRY_DATE": "2026-07-28"},
                {"EXCH_ID": "NSE", "SEGMENT": "D", "SECURITY_ID": 2, "INSTRUMENT": "OPTIDX", "UNDERLYING_SYMBOL": "NIFTY", "SM_EXPIRY_DATE": "2026-07-28"},
            ]).to_csv(path, index=False)
            config = {"exchange_id": "NSE", "master_segment": "D", "instrument": "FUTIDX", "underlying_symbol": "NIFTY"}
            self.assertEqual(discover_nifty_futures([str(path)], config).security_id.tolist(), ["1"])

    def test_chunks_are_non_overlapping(self):
        chunks = list(date_chunks(date(2026, 1, 1), date(2026, 4, 1), 30))
        self.assertEqual(chunks[-1][1], date(2026, 4, 1))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(chunks, chunks[1:])))

    def test_request_plan_preserves_requested_90_day_window(self):
        contracts = pd.DataFrame({
            "security_id": ["61093"],
            "expiry_date": [pd.Timestamp("2026-07-28")],
        })
        config = {
            "contract_history_days": 180, "chunk_days": 90,
            "exchange_segment": "NSE_FNO", "instrument": "FUTIDX",
            "interval_minutes": 1, "include_open_interest": True,
        }
        plans = plan_contract_requests(
            contracts, date(2026, 4, 23), date(2026, 7, 22), config
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["payload"]["fromDate"], "2026-04-23 00:00:00")
        self.assertEqual(plans[0]["payload"]["toDate"], "2026-07-22 00:00:00")
        self.assertEqual(plans[0]["payload"]["securityId"], "61093")

    def test_epoch_is_converted_to_kolkata(self):
        response = {key: [value] for key, value in {"timestamp": 0, "open": 1, "high": 2, "low": .5, "close": 1.5, "volume": 10}.items()}
        frame = response_to_frame(response, "Asia/Kolkata")
        self.assertEqual(str(frame.timestamp.dt.tz), "Asia/Kolkata")
        self.assertEqual(frame.timestamp.iloc[0].hour, 5)

    def test_roll_date_is_nth_observed_day_before_expiry(self):
        days = pd.to_datetime(["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28"])
        self.assertEqual(trading_day_roll_date("2026-07-28", days, 3), pd.Timestamp("2026-07-23"))

    def test_future_expiry_extends_observed_sessions(self):
        days = pd.to_datetime(["2026-07-20", "2026-07-21"])
        self.assertEqual(trading_day_roll_date("2026-07-28", days, 3), pd.Timestamp("2026-07-23"))

    def test_completed_chunk_is_resumed_without_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def requester(payload):
                calls.append(payload)
                return {key: [] for key in ("timestamp", "open", "high", "low", "close", "volume", "open_interest")}
            config = {
                "paths": {"chunk_directory": str(root / "chunks"), "manifest": str(root / "chunks/manifest.json"),
                          "request_log": str(root / "requests.log")},
                "credentials": {"client_id_env": "TEST_CLIENT", "access_token_env": "TEST_TOKEN"},
                "contract_history_days": 10, "chunk_days": 30, "exchange_segment": "NSE_FNO",
                "instrument": "FUTIDX", "interval_minutes": 1, "include_open_interest": True,
                "timezone": "Asia/Kolkata", "max_retries": 2, "retry_backoff_seconds": 0,
                "request_timeout_seconds": 1,
            }
            contracts = pd.DataFrame({"security_id": ["123"], "expiry_date": pd.to_datetime(["2026-01-10"])})
            downloader = DhanHistoricalDownloader(config, requester=requester, sleeper=lambda _: None)
            first = downloader.download_contracts(contracts, date(2026, 1, 1), date(2026, 1, 11))
            second = downloader.download_contracts(contracts, date(2026, 1, 1), date(2026, 1, 11))
            downloader.close()
            self.assertEqual(len(calls), 1)
            self.assertEqual(first, second)

    def test_downloader_processes_every_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def requester(payload):
                calls.append(payload.copy())
                return {key: [] for key in (
                    "timestamp", "open", "high", "low", "close", "volume", "open_interest"
                )}
            config = {
                "paths": {"chunk_directory": str(root / "chunks"),
                          "manifest": str(root / "chunks/manifest.json"),
                          "request_log": str(root / "requests.log")},
                "credentials": {"client_id_env": "TEST_CLIENT", "access_token_env": "TEST_TOKEN"},
                "contract_history_days": 180, "chunk_days": 30,
                "exchange_segment": "NSE_FNO", "instrument": "FUTIDX",
                "interval_minutes": 1, "include_open_interest": True,
                "timezone": "Asia/Kolkata", "max_retries": 1,
                "retry_backoff_seconds": 0, "request_timeout_seconds": 1,
            }
            contracts = pd.DataFrame({
                "security_id": ["61093"], "expiry_date": pd.to_datetime(["2026-07-28"])
            })
            with DhanHistoricalDownloader(config, requester=requester, sleeper=lambda _: None) as downloader:
                files = downloader.download_contracts(
                    contracts, date(2026, 4, 23), date(2026, 7, 22)
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(files), 3)
            self.assertEqual(calls[0]["fromDate"], "2026-04-23 00:00:00")
            self.assertEqual(calls[-1]["toDate"], "2026-07-22 00:00:00")

    def test_api_error_retains_exact_payload(self):
        payload = {"status": "failure", "errorCode": "XYZ", "errorMessage": "expired security"}
        with self.assertRaises(DhanAPIError) as caught:
            response_to_frame(payload, "Asia/Kolkata")
        self.assertIn('"errorCode": "XYZ"', str(caught.exception))
        self.assertIn("expired security", str(caught.exception))

    def test_chunk_validation_requires_oi_and_valid_ohlc(self):
        ts = int(pd.Timestamp("2026-01-02 09:15", tz="Asia/Kolkata").timestamp())
        frame = response_to_frame({
            "timestamp": [ts], "open": [100], "high": [99], "low": [98], "close": [100],
            "volume": [1], "open_interest": [10],
        }, "Asia/Kolkata")
        request = {"fromDate": "2026-01-02 00:00:00", "toDate": "2026-01-03 00:00:00"}
        with self.assertRaisesRegex(ValueError, "invalid high"):
            validate_chunk(frame, request, "Asia/Kolkata", True)

    def test_retry_then_success(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts = []
            config = {
                "paths": {"chunk_directory": str(Path(directory) / "chunks"),
                          "manifest": str(Path(directory) / "manifest.json"),
                          "request_log": str(Path(directory) / "requests.log")},
                "credentials": {"client_id_env": "X", "access_token_env": "Y"},
                "timezone": "Asia/Kolkata", "max_retries": 2, "retry_backoff_seconds": 0,
                "include_open_interest": True, "request_timeout_seconds": 1,
            }
            def requester(payload):
                attempts.append(1)
                if len(attempts) == 1:
                    raise DhanAPIError("temporary", 500, '{"error":"temporary"}')
                return {key: [] for key in ("timestamp", "open", "high", "low", "close", "volume", "open_interest")}
            downloader = DhanHistoricalDownloader(config, requester=requester, sleeper=lambda _: None)
            frame, validation = downloader._download_with_retry(
                "key", {"securityId": "1", "fromDate": "2026-01-01 00:00:00", "toDate": "2026-01-02 00:00:00"}
            )
            downloader.close()
            self.assertEqual(len(attempts), 2)
            self.assertTrue(frame.empty)
            self.assertTrue(validation["valid"])


if __name__ == "__main__":
    unittest.main()
