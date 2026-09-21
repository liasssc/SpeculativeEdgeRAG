"""Token-level selection across arbitrary draft branches (baseline)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .draft import DraftBranch, DraftToken

@dataclass(frozen=True)
class SelectedCandidate:
    token: str
    probability: float
    server_id: str
    position: int

class MultiDraftSelector:
    """Select candidates with importance-style score p(proposal), independent of server selection."""
    def __init__(self, max_branches: int = 4): self.max_branches = max_branches
    def select(self, branches: Sequence[DraftBranch], gamma: int = 4) -> list[SelectedCandidate]:
        candidates=[]
        for b in list(branches)[:self.max_branches]:
            for t in b.tokens[:gamma]: candidates.append(SelectedCandidate(t.token,t.probability,b.server_id,t.position))
        # retain one candidate per position, highest proposal probability
        out=[]
        for pos in range(gamma):
            at=[c for c in candidates if c.position==pos]
            if at: out.append(max(at,key=lambda c:c.probability))
        return out

    def select_candidates(self, branches: Sequence[DraftBranch], gamma: int = 4) -> list[SelectedCandidate]:
        return self.select(branches, gamma)
