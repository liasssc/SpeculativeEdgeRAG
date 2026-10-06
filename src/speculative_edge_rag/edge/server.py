"""Mock edge server integrating private retrieval and evidence export."""
from __future__ import annotations
from dataclasses import dataclass
from .local_kg import LocalKnowledgeGraph, load_server_graph
from .retriever import LocalRetriever
from .evidence import EvidenceBuilder, EvidenceCapsule
from .probe import LocalProbeIndex
from ..schemas import QueryTags, ProbeResult

@dataclass
class EdgeServer:
    server_id: str
    knowledge_graph: LocalKnowledgeGraph
    available: bool = True
    def __post_init__(self) -> None:
        self.retriever = LocalRetriever(self.knowledge_graph)
        self.evidence_builder = EvidenceBuilder()
        self.probe_index = LocalProbeIndex.from_graph(self.knowledge_graph)

    @classmethod
    def from_graph(cls, server_id: str, graph_path: str, *, available: bool = True) -> "EdgeServer":
        """Construct an edge server using only its private JSON graph."""
        return cls(server_id, load_server_graph(server_id, graph_path), available=available)
    def retrieve_evidence(self, query: str, request_id: str, query_id: str, top_k: int = 3, selected_tags: tuple[str, ...] = ()) -> EvidenceCapsule:
        result = self.retriever.retrieve(query, top_k)
        # selected_tags describes retrieval hits on this server only. Query tags
        # that did not occur in returned local nodes must not leak into the capsule.
        return self.evidence_builder.build(request_id, query_id, self.server_id, result, top_k)
    def probe(self,query: str,query_tags: QueryTags,top_k: int=5) -> ProbeResult:
        """Return a privacy-safe summary without constructing evidence or drafts."""
        return self.probe_index.probe(self.server_id,query,query_tags,top_k)
