"""Core typed schemas shared by coordinator and edge components."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import time

class TagSource(str, Enum):
    ORACLE = "oracle"
    PREDICTED = "predicted"

@dataclass(frozen=True)
class QueryTag:
    level1: str
    level2: str
    confidence: float = 1.0
    source: TagSource = TagSource.PREDICTED

@dataclass
class QueryTags:
    tags: list[QueryTag] = field(default_factory=list)
    @property
    def level1(self) -> set[str]: return {t.level1 for t in self.tags}
    @property
    def level2(self) -> set[str]: return {t.level2 for t in self.tags}

@dataclass
class ResourceState:
    queue_length: int = 0
    load: float = 0.0
    kv_cache_utilization: float = 0.0
    available: bool = True

@dataclass
class ServerRecord:
    server_id: str
    level1_tags: set[str] = field(default_factory=set)
    level2_tags: set[str] = field(default_factory=set)
    knowledge_quality: float = 0.5
    queue_length: int = 0
    load: float = 0.0
    kv_cache_utilization: float = 0.0
    retrieval_latency_ms: float = 50.0
    draft_latency_ms: float = 50.0
    network_latency_ms: float = 20.0
    available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resource_state(self) -> ResourceState:
        return ResourceState(self.queue_length, self.load, self.kv_cache_utilization, self.available)

@dataclass
class RequestState:
    request_id: str = "request-1"
    query_id: str = "query-1"
    deadline_ms: float = 1000.0
    max_servers: int = 4
    selection_mode: str = "joint"
    gamma: int = 4

@dataclass(frozen=True)
class ProbeResult:
    """Privacy-safe summary returned by an edge-local relevance probe."""
    server_id: str
    matched_tag_pairs: tuple[tuple[str, str], ...] = ()
    top_relevance_score: float = 0.0
    mean_top_k_score: float = 0.0
    relevant_chunk_count: int = 0
    query_term_coverage: tuple[int, ...] = ()
    top_chunk_ids: tuple[str, ...] = ()
    estimated_probe_ms: float = 0.0
    uncertainty: float = 1.0

@dataclass
class EdgeBranchResult:
    """Future edge response: evidence and a draft branch travel together.

    This is an interface-only schema.  It deliberately contains no model or
    decoding behavior so offline data preparation remains model free.
    """
    server_id: str
    evidence_capsule: Any = None
    draft_branch: Any = None
    retrieval_score: float = 0.0
    covered_tags: list[str] = field(default_factory=list)
    latency_statistics: Mapping[str, float] = field(default_factory=dict)
