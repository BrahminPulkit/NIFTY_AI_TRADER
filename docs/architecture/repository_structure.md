# Repository Structure

## Dependency Direction

```text
apps / commands
       |
       v
      src
       |
       v
data / production_model / config
```

`src/` must never import from `apps/` or `commands/`. Applications may present
or orchestrate library behavior but may not reimplement model, feature,
decision, risk or execution rules.

## Application Ownership

`apps/web/` is the commercial React client and `apps/api/` is its local,
versioned FastAPI presentation adapter. `apps/research_ui/` remains the
internal Streamlit research application. A future Tauri wrapper belongs in
`apps/desktop/`. Applications communicate through typed contracts while
Python remains the source of truth.

The React client owns navigation, presentation state, caching and responsive
workspaces. It never owns model, signal, strategy, risk, broker or paper-trade
behavior. FastAPI routers translate HTTP requests only; existing `src/`
services retain domain ownership.

## Command Ownership

- `commands/data/`: acquisition, cleaning and deterministic dataset builds
- `commands/training/`: model training and frozen-package creation
- `commands/backtesting/`: replay, backtest and temporal validation
- `commands/operations/`: live inference, monitoring and readiness
- `commands/research/`: offline experiments with no production mutation

Commands must remain thin entrypoints. Reusable behavior belongs in `src/`.

## Artifact Ownership

`data/`, `reports/`, `logs/` and `production_model/` retain their current paths
because frozen manifests and reproducibility metadata refer to them. They must
not be moved during a source-layout refactor.
