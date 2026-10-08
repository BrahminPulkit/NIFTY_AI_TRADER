import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
text = "".join(nb["cells"][7]["source"])
replacements = {
    "LIVE_MAX_SIGNALS_PER_DAY = 5":
        "LIVE_MAX_BEST_SIGNALS_CAP = 5  # safety ceiling, never a daily target",
    "signal_count < LIVE_MAX_SIGNALS_PER_DAY":
        "signal_count < LIVE_MAX_BEST_SIGNALS_CAP",
    "signal_count >= LIVE_MAX_SIGNALS_PER_DAY":
        "signal_count >= LIVE_MAX_BEST_SIGNALS_CAP",
    "Daily best-trade limit reached; monitoring stopped.":
        "Safety cap of 5 qualified best-trade signals reached; monitoring stopped.",
    "A trade is valid only when BEST TRADE SIGNAL is printed.":
        "A trade is valid only when BEST TRADE SIGNAL is printed; 0 trades is valid when quality is absent.",
}
for old,new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one match for {old!r}, found {text.count(old)}")
    text = text.replace(old,new,1)
nb["cells"][7]["source"] = text.splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(path)
