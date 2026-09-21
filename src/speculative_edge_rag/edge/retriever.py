"""Local retrieval baseline."""
from __future__ import annotations
from dataclasses import dataclass
from .local_kg import LocalKnowledgeGraph, KnowledgeNode, JsonKnowledgeGraph, load_server_graph

@dataclass(frozen=True)
class RetrievalResult:
    nodes: list[KnowledgeNode]
    success: bool
    score: float
    covered_tags: tuple[str, ...]

class LocalRetriever:
    def __init__(self, knowledge_graph: LocalKnowledgeGraph): self.knowledge_graph = knowledge_graph
    @classmethod
    def from_server(cls, server_id: str, graph_path: str) -> "LocalRetriever":
        return cls(load_server_graph(server_id, graph_path))
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        nodes = self.knowledge_graph.search(query, top_k)
        score = min(1.0, len(nodes) / max(1, top_k))
        tags = tuple(sorted({tag for n in nodes for tag in n.tags}))
        return RetrievalResult(nodes, bool(nodes), score, tags)
