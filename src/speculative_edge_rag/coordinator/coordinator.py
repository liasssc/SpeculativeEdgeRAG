"""End-to-end baseline Coordinator orchestration."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from .tag_router import HierarchicalMultiLabelTagger
from .server_registry import ServerRegistry
from .server_selector import ResourceAwareServerSelector
from ..schemas import RequestState
from ..edge.evidence import EvidenceAggregator
from ..speculative.draft import DraftBranch, TargetDistribution
from ..speculative.multi_draft_selector import MultiDraftSelector
from ..speculative.verifier import SpeculativeVerifier

class EvidenceAwareDraftProvider(Protocol):
    def build_branch(self, server_id: str, query: str, evidence_capsule_or_context: Any,
                     committed_prefix: Any, gamma: int) -> DraftBranch: ...

class TargetProvider(Protocol):
    def distribution(self, query: str, aggregated_evidence: Mapping[str, object],
                     committed_prefix: Any) -> TargetDistribution: ...

class MockEvidenceAwareDraftProvider:
    """Mock provider; replaceable by a real edge SLM in a later phase."""
    def build_branch(self, server_id, query, evidence_capsule_or_context, committed_prefix, gamma):
        chunks = getattr(evidence_capsule_or_context, "chunks", ())
        words = []
        for chunk in chunks:
            words.extend(chunk.text.split())
        # Keep vocabulary tokens natural; server_id is carried by DraftBranch
        # and SelectedCandidate metadata.
        tokens = [w.strip('.,!?').lower() for w in words if w.strip()]
        if not tokens:
            tokens = ["evidence-empty"]
        return DraftBranch.from_tokens(server_id, tokens[:gamma])

class MockEvidenceAwareTargetProvider:
    """Mock provider; replaceable by a real cloud LLM target in a later phase."""
    def distribution(self, query, aggregated_evidence, committed_prefix):
        # Tokens are natural text. Server identity remains in capsule/candidate
        # metadata and is never encoded into the target vocabulary.
        tokens = []
        for chunk in aggregated_evidence.get("chunks", ()):
            words = chunk.text.split()
            tokens.extend(words)
        tokens.extend(query.split())
        unique = list(dict.fromkeys(t.strip(".,!?").lower() for t in tokens if t.strip()))
        if not unique: unique = ["evidence-empty"]
        # Deterministic mock calibration: concentrate enough mass on the first
        # observed token to demonstrate acceptance, while leaving other tokens
        # below q for rejection. This is not paper acceptance sampling.
        if len(unique) == 1:
            probs = {unique[0]: 1.0}
        else:
            head = 0.4
            tail = (1.0 - head) / (len(unique) - 1)
            probs = {token: (head if i == 0 else tail) for i, token in enumerate(unique)}
        return TargetDistribution(probs)

@dataclass
class Coordinator:
    tagger: HierarchicalMultiLabelTagger
    registry: ServerRegistry
    selector: ResourceAwareServerSelector
    edge_servers: Mapping[str, object]
    aggregator: EvidenceAggregator = field(default_factory=EvidenceAggregator)
    draft_selector: MultiDraftSelector = field(default_factory=MultiDraftSelector)
    verifier: SpeculativeVerifier = field(default_factory=SpeculativeVerifier)
    draft_provider: EvidenceAwareDraftProvider = field(default_factory=MockEvidenceAwareDraftProvider)
    target_provider: TargetProvider = field(default_factory=MockEvidenceAwareTargetProvider)

    def process_request(self, query: str, request_state: RequestState | None = None) -> dict[str, object]:
        state = request_state or RequestState(request_id=str(uuid.uuid4()), deadline_ms=500)
        tags = self.tagger.tag(query)
        candidates = self.registry.candidates(tags)
        selected = self.selector.select(candidates, tags, state)
        capsules = [self.edge_servers[s.server_id].retrieve_evidence(query=query, request_id=state.request_id,
                    query_id=state.query_id, top_k=3, selected_tags=tuple(t.level2 for t in tags.tags))
                    for s in selected.servers]
        evidence = self.aggregator.aggregate(capsules)
        branches = [self.draft_provider.build_branch(s.server_id, query, cap, (), state.gamma)
                    for s, cap in zip(selected.servers, capsules)]
        choices = self.draft_selector.select(branches, gamma=4)
        target = self.target_provider.distribution(query, evidence, ())
        verification = self.verifier.verify(choices, target)
        return {"selected_servers": [s.server_id for s in selected.servers], "query_tags": tags,
                "evidence_summary": evidence, "draft_branches": branches,
                "target_distribution": target, "accepted_tokens": verification.accepted_tokens,
                "rejected_tokens": verification.rejected_tokens,
                "residual_distribution": verification.residual_distribution,
                "latency_statistics": {"estimated_ms": selected.latency_ms},
                "selection_explanation": selected.explanation}
