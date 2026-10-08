import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
text = "".join(nb["cells"][7]["source"])
old = "LIVE_MAX_SIGNALS_PER_DAY = 3"
new = "LIVE_MAX_SIGNALS_PER_DAY = 5"
if text.count(old) != 1:
    raise RuntimeError(f"Expected one live limit, found {text.count(old)}")
nb["cells"][7]["source"] = text.replace(old,new,1).splitlines(keepends=True)
nb["cells"][7]["outputs"] = []
nb["cells"][7]["execution_count"] = None
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(path)
