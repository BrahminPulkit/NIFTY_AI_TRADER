"""Preserve percentage drawdown protection for legacy trade rows without pnl."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "notebooks" / "NIFTY_Scalping_Paper_Trading.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
shape = [(c.get("id"), c.get("metadata"), c.get("outputs")) for c in nb["cells"]]
text = "".join(nb["cells"][10]["source"])
old = "    realized_pnl = sum(float(t.get('pnl', 0.0)) for t in trades)\n"
new = ("    realized_pnl = sum(float(t['pnl']) if 'pnl' in t else\n"
       "                       float(t.get('result_r',0.0))*CFG.paper_capital*0.01\n"
       "                       for t in trades)\n")
if text.count(old) != 1:
    raise RuntimeError("Risk fallback insertion point missing")
nb["cells"][10]["source"] = text.replace(old, new, 1).splitlines(keepends=True)
if shape != [(c.get("id"), c.get("metadata"), c.get("outputs")) for c in nb["cells"]]:
    raise RuntimeError("Notebook structure changed")
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
