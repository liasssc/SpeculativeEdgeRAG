"""Configuration defaults for the baseline experiment framework."""
from dataclasses import dataclass

@dataclass
class SystemConfig:
    max_servers: int = 4
    default_deadline_ms: float = 1000.0
    selector_mode: str = "joint"
    top_k_evidence: int = 3
    max_draft_branches: int = 4

DEFAULT_CONFIG = SystemConfig()
