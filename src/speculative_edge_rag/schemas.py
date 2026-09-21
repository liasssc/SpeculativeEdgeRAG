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

