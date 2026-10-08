import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
text = "".join(nb["cells"][7]["source"])
old = """# One immediate connection and signal check:
live_signal = check_live_signal()

# For continuous monitoring, run this manually in a separate execution:
# run_live_monitor(poll_seconds=10)
"""
new = """# Running this cell starts continuous paper-signal monitoring automatically.
# Stop safely with Jupyter's Interrupt Kernel button.
RUN_CONTINUOUS_MONITOR = True
if RUN_CONTINUOUS_MONITOR:
    run_live_monitor(poll_seconds=10)
else:
    live_signal = check_live_signal()
"""
if text.count(old) != 1:
    raise RuntimeError(f"Continuous-monitor insertion point not found ({text.count(old)})")
nb["cells"][7]["source"] = text.replace(old,new,1).splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(path)
