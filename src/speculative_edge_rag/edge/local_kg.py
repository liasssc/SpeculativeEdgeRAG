"""Private local knowledge graph baseline implementations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Any
import json
from pathlib import Path
import re

@dataclass(frozen=True)
class KnowledgeNode:
    node_id: str
    text: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class KnowledgeEdge:
    source_id: str
    target_id: str
    relation: str

class LocalKnowledgeGraph(Protocol):
    def add_node(self, node: KnowledgeNode) -> None: ...
    def add_edge(self, edge: KnowledgeEdge) -> None: ...
    def search(self, query: str, top_k: int = 3) -> list[KnowledgeNode]: ...

class InMemoryKnowledgeGraph:
    """Baseline private graph; raw nodes remain local to an edge server."""
    def __init__(self, nodes: Iterable[KnowledgeNode] = (), edges: Iterable[KnowledgeEdge] = ()) -> None:
        self.nodes: dict[str, KnowledgeNode] = {n.node_id: n for n in nodes}
        self.edges: list[KnowledgeEdge] = list(edges)
    def add_node(self, node: KnowledgeNode) -> None: self.nodes[node.node_id] = node
    def add_edge(self, edge: KnowledgeEdge) -> None: self.edges.append(edge)
    def search(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        terms = set(query.lower().split())
        scored = []
        for n in self.nodes.values():
            score = len(terms & set(n.text.lower().split())) + len(terms & set(t.lower() for t in n.tags))
            if score: scored.append((score, n))
        scored.sort(key=lambda x: (-x[0], x[1].node_id))
        return [n for _, n in scored[:top_k]]


class JsonKnowledgeGraph(InMemoryKnowledgeGraph):
    """A read-only friendly graph loaded from one edge server's ``graph.json``.

    The loader intentionally accepts both ``node_id``/``source_id`` and the
    shorter ``id``/``source`` spellings so graphs produced by older scripts can
    be consumed without rewriting the source data.
    """
    def __init__(self, nodes: Iterable[KnowledgeNode] = (), edges: Iterable[KnowledgeEdge] = (), *, path: str | Path | None = None):
        if path is not None:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            nodes, edges = self._parse_payload(payload)
        super().__init__(nodes, edges)

    @staticmethod
    def _parse_payload(payload: Any) -> tuple[list[KnowledgeNode], list[KnowledgeEdge]]:
        raw_nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        raw_edges = payload.get("edges", []) if isinstance(payload, dict) else []
        nodes: list[KnowledgeNode] = []
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            node_id = str(raw.get("node_id", raw.get("id", "")))
            if not node_id:
                continue
            text = str(raw.get("text", raw.get("label", raw.get("title", ""))))
            tags = raw.get("tags", raw.get("level2_tags", [])) or []
            if isinstance(tags, str): tags = [tags]
            metadata = dict(raw.get("metadata") or {})
            if "type" in raw: metadata.setdefault("type", raw["type"])
            nodes.append(KnowledgeNode(node_id, text, tuple(map(str, tags)), metadata))
        edges: list[KnowledgeEdge] = []
        for raw in raw_edges:
            if not isinstance(raw, dict): continue
            source = raw.get("source_id", raw.get("source")); target = raw.get("target_id", raw.get("target"))
            relation = raw.get("relation", raw.get("type", "related"))
            if source is not None and target is not None:
                edges.append(KnowledgeEdge(str(source), str(target), str(relation)))
        return nodes, edges

    @classmethod
    def from_file(cls, path: str | Path) -> "JsonKnowledgeGraph":
        return cls(path=path)

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeNode]:
        # Match punctuation-insensitively while retaining the deterministic
        # lexical baseline used by InMemoryKnowledgeGraph.
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for node in self.nodes.values():
            haystack = set(re.findall(r"\w+", (node.text + " " + " ".join(node.tags)).lower()))
            score = len(terms & haystack)
            if score: scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1].node_id))
        return [node for _, node in scored[:top_k]]


def load_server_graph(server_id: str, path: str | Path) -> JsonKnowledgeGraph:
    """Load only ``server_id``'s graph from a graph root or direct JSON path."""
    candidate = Path(path)
    if candidate.is_dir():
        direct = candidate / "graph.json"
        candidate = direct if direct.exists() else candidate / server_id / "graph.json"
    return JsonKnowledgeGraph.from_file(candidate)


def load_server_documents(server_id: str, path: str | Path) -> list[dict[str, Any]]:
    """Read one server's JSONL partition without touching other partitions."""
    candidate = Path(path)
    if candidate.is_dir():
        direct = candidate / "documents.jsonl"
        candidate = direct if direct.exists() else candidate / server_id / "documents.jsonl"
    records: list[dict[str, Any]] = []
    with candidate.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): records.append(json.loads(line))
    return records
