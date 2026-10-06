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
        best_tie: tuple[float, float, float, int] | None = None
        limit = min(self.max_servers, request.max_servers, len(candidates))
        required = {("level1", tag) for tag in query_tags.level1} | {("level2", tag) for tag in query_tags.level2}
        for n in range(1, limit + 1):
            for subset in combinations(candidates, n):
                latency = subset_latency_ms(list(subset))
                if latency > request.deadline_ms: continue
                covered = {("level1", tag) for tag in set().union(*(s.level1_tags for s in subset)) & query_tags.level1} | {("level2", tag) for tag in set().union(*(s.level2_tags for s in subset)) & query_tags.level2}
                coverage = len(covered)
                coverage_ratio = coverage / len(required) if required else 1.0
                quality = sum(s.knowledge_quality for s in subset) / n
                # Coverage is the primary objective: a high-quality server that
                # misses a requested tag must not beat a complementary subset.
                if mode == "quality": score = coverage_ratio * 100.0 + quality
                elif mode == "latency": score = -(latency / max(request.deadline_ms, 1))
                else: score = coverage_ratio * 100.0 + quality - latency / max(request.deadline_ms, 1)
                tie = (coverage_ratio, quality, -latency, -n)
                if best is None or score > best.score or (score == best.score and tie > best_tie):
                    result = SelectionResult(list(subset), score, f"mode={mode}; quality={quality:.3f}; coverage={coverage}/{len(required)}; latency={latency:.1f}ms", latency)
                    best = result
                    best_tie = tie
        return best or SelectionResult([], float("-inf"), "no feasible server subset", float("inf"))
