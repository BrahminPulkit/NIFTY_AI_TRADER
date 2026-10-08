# Public-clone validation

Validated on Windows with Python 3.12 from a fresh public clone, without private datasets or broker credentials.

| Check | Result |
| --- | --- |
| Install documented Python dependencies in an isolated environment | Passed after dependency fixes |
| `python -m pip check` | Passed |
| `npm ci` and `npm run build` in `apps/web` | Passed |
| API health/request-ID/sanitization and risk smoke tests | 7 passed, 1 deselected |
| Local market workspace in headless Chrome | Rendered in disconnected mode |

The dependency fixes align Streamlit 1.65.0 with pandas 3, declare CatBoost and
scikit-learn for inference/research imports, and include HTTPX for TestClient.

Reproduce the focused checks:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests/test_risk_manager.py tests/test_local_api.py -k 'risk or health or sanitized or request_id or credentials' -q
npm --prefix apps/web ci
npm --prefix apps/web run build
```

These checks are also defined in `.github/workflows/portfolio-checks.yml`.
The selected tests do not exercise the entire historical-data suite. Full research,
replay and model-quality validation requires the excluded local datasets.
No broker authentication, live order execution or trading-return verification was
performed. Current Starlette emits an HTTPX deprecation warning; the smoke tests
still pass.

The screenshot is a real empty/disconnected state from the public clone. It does
not depict live prices, predictions, a broker account or measured trading returns.
