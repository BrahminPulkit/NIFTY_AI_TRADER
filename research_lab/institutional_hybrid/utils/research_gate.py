"""Hard gate preventing research execution before data approval."""
from __future__ import annotations
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[1]
GATE=LAB/"data_validation"/"outputs"/"research_data_gate.json"

class ResearchDataGateError(RuntimeError): pass

def require_approved_data() -> dict:
    if not GATE.exists():
        raise ResearchDataGateError("Data validation gate is absent; strategy execution prohibited.")
    payload=json.loads(GATE.read_text(encoding="utf-8"))
    if payload.get("approved") is not True or payload.get("status")!="PASS":
        reasons="; ".join(payload.get("critical_issues",[]))
        raise ResearchDataGateError(f"Data validation gate denied strategy execution: {reasons}")
    return payload
