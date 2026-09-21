"""Cloud target distribution speculative verification baseline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
from .draft import DraftToken, TargetDistribution
from .residual import ResidualCorrection

@dataclass
class VerificationResult:
    accepted_tokens: list[str]
    rejected_tokens: list[str]
    residual_distribution: dict[str,float]

class SpeculativeVerifier:
    """Deterministic mock baseline verifier.

    Paper steps are intentionally deferred: randomized ``min(1,p/q)``
    acceptance, first in-block rejection handling, true residual sampling, and
    committed-prefix rollback.
    """
    def __init__(self, residual: ResidualCorrection|None=None): self.residual=residual or ResidualCorrection()
    def verify(self, candidates: Sequence[DraftToken|object], target: TargetDistribution|Mapping[str,float], threshold: float = 0.0) -> VerificationResult:
        probs = target.probabilities if isinstance(target,TargetDistribution) else target
        accepted=[]; rejected=[]; proposal={}
        for c in candidates:
            tok = c.token
            q = float(getattr(c,'probability',0.0)); p=float(probs.get(tok,0.0)); proposal[tok]=q
            if p >= q and p >= threshold: accepted.append(tok)
            else: rejected.append(tok)
        return VerificationResult(accepted,rejected,self.residual.correct(probs,proposal))

    def verify_candidates(self, candidates: Sequence[DraftToken|object], target: TargetDistribution|Mapping[str,float], threshold: float = 0.0) -> VerificationResult:
        return self.verify(candidates, target, threshold)
