# NIFTY AI Trader

Single-user local platform for NIFTY intraday research, live market signals,
paper trading, monitoring and chronological validation.

## Highlights

- React and TypeScript trading workspace with a FastAPI backend.
- Streamlit research interface for strategy analysis and monitoring.
- Python pipelines for market features, model inference and chronological validation.
- Dhan market-data integration and a paper-trading workflow.
- Automated tests covering data pipelines, API contracts and trading logic.

## Local Setup

Requires Python 3.12 or later, Node.js and npm.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm --prefix apps/web ci
```

This public repository contains application code, tests, configuration and
research code. Private market datasets, notebook sessions, generated reports,
runtime journals and backups are excluded. Historical research and model
workflows require the corresponding local datasets; a fresh clone does not
include those inputs. Broker credentials must be supplied locally through
the application or process environment, never committed to Git.

## Repository Layout

| Path | Ownership |
|---|---|
| `apps/research_ui/` | Streamlit research and operations interface |
| `commands/` | Categorized CLI entrypoints |
| `src/` | Trading, broker, feature, inference and research libraries |
| `tests/` | Unit, contract, integration and regression coverage |
| `production_model/` | Frozen production-candidate model package |
| `data/` | Immutable and generated datasets |
| `reports/` | Reproducible research and operational outputs |
| `logs/` | Runtime journals and health logs |
| `config/` | Non-secret application configuration |
| `docs/` | Architecture, validation and operational documentation |
| `research_lab/` | Isolated exploratory work; not an application dependency |

## Run the Commercial Platform

```powershell
.\scripts\run_platform.ps1
```

The React workspace opens at `http://127.0.0.1:5173`. Its local FastAPI
adapter reads the existing Python decision outputs and process-wide Dhan cache;
it does not implement trading or prediction logic.

Commercial source ownership:

- `apps/web/src/pages/`: routed trading workspaces
- `apps/web/src/features/`: signals, charts and notifications
- `apps/web/src/hooks/`: query and broker-action hooks
- `apps/web/src/services/`: validated API client
- `apps/api/routers/`: `/api/v1` endpoint groups
- `apps/api/schemas/`: public Pydantic contracts

## Run the Legacy Research UI

```powershell
.\scripts\run_research_ui.ps1
```

or:

```powershell
streamlit run apps/research_ui/app.py
```

## Run Tests

```powershell
.\scripts\test.ps1
```

## Safety Boundary

The application is read-only with respect to broker orders. Model, feature,
threshold and decision behavior are owned by `src/` and frozen artifacts, not
by UI code or command wrappers.
