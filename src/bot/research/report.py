"""Research report data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bot.engines.tuner import ParamChange


@dataclass
class ResearchReport:
    """Result of a research experiment."""

    experiment_name: str
    hypothesis: str
    methodology: str
    data_period: str
    results: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    recommended_changes: list[ParamChange] = field(default_factory=list)
    improvement_significant: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "hypothesis": self.hypothesis,
            "methodology": self.methodology,
            "data_period": self.data_period,
            "results": self.results,
            "conclusion": self.conclusion,
            "recommended_changes": [c.to_dict() for c in self.recommended_changes],
            "improvement_significant": self.improvement_significant,
            "timestamp": self.timestamp,
        }
