"""Transparent exhaustive baseline resource-aware server selector."""
from __future__ import annotations
from itertools import combinations
from dataclasses import dataclass
from ..schemas import ServerRecord, QueryTags, RequestState
from .resource_state import subset_latency_ms

@dataclass
class SelectionResult:
    servers: list[ServerRecord]
    score: float
    explanation: str
    latency_ms: float

class ResourceAwareServerSelector:
    def __init__(self, max_servers: int = 4) -> None: self.max_servers = max_servers
    def select(self, candidates: list[ServerRecord], query_tags: QueryTags, request: RequestState | None = None, mode: str | None = None) -> SelectionResult:
        request = request or RequestState(max_servers=self.max_servers)
        mode = mode or request.selection_mode
        best: SelectionResult | None = None
        limit = min(self.max_servers, request.max_servers, len(candidates))
        for n in range(1, limit + 1):
            for subset in combinations(candidates, n):
                latency = subset_latency_ms(list(subset))
                if latency > request.deadline_ms: continue
                coverage = len(set().union(*(s.level1_tags for s in subset)) & query_tags.level1) + len(set().union(*(s.level2_tags for s in subset)) & query_tags.level2)
                quality = sum(s.knowledge_quality for s in subset) / n
                if mode == "quality": score = quality + coverage * .1
                elif mode == "latency": score = -(latency / max(request.deadline_ms, 1))
                else: score = quality + coverage * .1 - latency / max(request.deadline_ms, 1)
                if best is None or score > best.score: best = SelectionResult(list(subset), score, f"mode={mode}; quality={quality:.3f}; coverage={coverage}; latency={latency:.1f}ms", latency)
        return best or SelectionResult([], float("-inf"), "no feasible server subset", float("inf"))
