"""Four declared hybrids. No search, parameter grid, or optimiser exists."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Hybrid:
    name: str
    entry_package: str
    filter_package: str
    exit_package: str

HYBRIDS = (
    Hybrid("hybrid_v1", "linda_raschke", "oliver_velez", "al_brooks"),
    Hybrid("hybrid_v2", "al_brooks", "linda_raschke", "adam_grimes"),
    Hybrid("hybrid_v3", "mark_minervini", "oliver_velez", "linda_raschke"),
    Hybrid("hybrid_v4", "ict_research", "adam_grimes", "al_brooks"),
)
