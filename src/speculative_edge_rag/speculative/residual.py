"""Residual distribution correction used after speculative rejection."""
from __future__ import annotations
from typing import Mapping

def normalize(dist: Mapping[str,float]) -> dict[str,float]:
    s = sum(max(0.0,v) for v in dist.values())
    return {k: max(0.0,v)/s for k,v in dist.items()} if s else {}

class ResidualCorrection:
    """Compute residual max(p_c - p_q, 0), normalized."""
    def correct(self, target: Mapping[str,float], proposal: Mapping[str,float]) -> dict[str,float]:
        return normalize({k:max(0.0,target.get(k,0.0)-proposal.get(k,0.0)) for k in set(target)|set(proposal)})

    # Alias used by integrations that prefer an explicit method name.
    def compute(self, target: Mapping[str,float], proposal: Mapping[str,float]) -> dict[str,float]:
        return self.correct(target, proposal)
