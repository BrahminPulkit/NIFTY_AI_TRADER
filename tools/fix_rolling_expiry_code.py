import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
text = "".join(nb["cells"][3]["source"])
old = "'expiryFlag':'WEEK', 'expiryCode':0,"
new = "'expiryFlag':'WEEK', 'expiryCode':1,"
if text.count(old) != 1:
    raise RuntimeError(f"Expected one expiryCode insertion point, found {text.count(old)}")
nb["cells"][3]["source"] = text.replace(old, new, 1).splitlines(keepends=True)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
