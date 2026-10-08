"""CLI: data validation only. It cannot invoke strategy code."""
from __future__ import annotations
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[1]
if str(LAB.parent) not in sys.path: sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.data_validation.engine import InstitutionalDataValidator
from institutional_hybrid.data_validation.reporting import export

def main():
    validator=InstitutionalDataValidator()
    result,loaded,tables=validator.run()
    export(result,loaded,tables)
    print(f"{result['status']}: Data Quality Score {result['data_quality_score']}/100")
    for issue in result["critical_issues"]: print(f"- {issue}")
    return 0 if result["can_research_continue"] else 2

if __name__=="__main__": raise SystemExit(main())

