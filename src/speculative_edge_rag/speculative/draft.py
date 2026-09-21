"""Baseline multi draft branch data structures."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

@dataclass(frozen=True)
class DraftToken:
    token: str
    probability: float
    position: int = 0

@dataclass
class ProposalDistribution:
    probabilities: Mapping[str, float] = field(default_factory=dict)

@dataclass
class TargetDistribution:
    probabilities: Mapping[str, float] = field(default_factory=dict)

@dataclass
class DraftBranch:
    server_id: str
    tokens: list[DraftToken] = field(default_factory=list)
    proposal: ProposalDistribution = field(default_factory=ProposalDistribution)
    latency_ms: float = 0.0

    @classmethod
    def from_tokens(cls, server_id: str, tokens: Sequence[str], probability: float = 1.0) -> "DraftBranch":
        # Mock q is a categorical distribution over this branch's tokens.
        # ``probability`` is retained as an optional scale only for compatibility;
        # the resulting distribution is always normalized.
        unique = list(dict.fromkeys(tokens))
        if not unique:
            return cls(server_id, [], ProposalDistribution({}))
        q = 1.0 / len(unique)
        ts = [DraftToken(t, q, i) for i, t in enumerate(unique)]
        return cls(server_id, ts, ProposalDistribution({t: q for t in unique}))
