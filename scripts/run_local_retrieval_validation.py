#!/usr/bin/env python3
"""Run a model-free, local retrieval validation over the prepared dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Allow running this file directly from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speculative_edge_rag.coordinator.server_registry import ServerRegistry
from speculative_edge_rag.coordinator.server_selector import ResourceAwareServerSelector
from speculative_edge_rag.coordinator.tag_router import OracleTagger
from speculative_edge_rag.edge.evidence import EvidenceAggregator
from speculative_edge_rag.edge.local_kg import InMemoryKnowledgeGraph, KnowledgeNode
from speculative_edge_rag.edge.server import EdgeServer
from speculative_edge_rag.schemas import QueryTag, RequestState, ServerRecord


def _records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_registry(path: Path) -> ServerRegistry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    registry = ServerRegistry()
    for raw in payload.get("servers", []):
        quality = (raw.get("quality") or {}).get("score")
        state = raw.get("resource_state") or {}
        registry.register(ServerRecord(
            server_id=str(raw["server_id"]),
            level1_tags=set(raw.get("level1_tags") or []),
            level2_tags=set(raw.get("level2_tags") or []),
            knowledge_quality=float(quality) if quality is not None else 0.5,
            queue_length=int(state.get("queue_length", 0)),
            load=float(state.get("load", 0.0)),
            available=bool(state.get("available", raw.get("resource_state", {}).get("status") != "offline")),
            metadata={"document_count": raw.get("document_count", 0), "chunk_count": raw.get("chunk_count", 0)},
        ))
    return registry


def _edge_from_chunks(server_id: str, chunk_path: Path) -> EdgeServer:
    nodes = []
    for raw in _records(chunk_path):
        tags = list(raw.get("level2_tags") or [])
        nodes.append(KnowledgeNode(
            node_id=str(raw.get("chunk_id", raw.get("id", ""))),
            text=str(raw.get("text", "")),
            tags=tuple(str(tag) for tag in tags),
            metadata={"level1_tags": ",".join(map(str, raw.get("level1_tags") or [])),
                      "domain": str(raw.get("domain", ""))},
        ))
    return EdgeServer(server_id, InMemoryKnowledgeGraph(nodes))


def _query_tags(raw: dict[str, Any]):
    level1 = raw.get("level1_tags") or []
    level2 = raw.get("level2_tags") or []
    return OracleTagger().tag({"level1": level1, "level2": level2})


def stratified_query_sample(queries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take a deterministic round-robin sample across level-1 domains."""
    if limit >= len(queries):
        return queries
    groups: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        tags = query.get("level1_tags") or [query.get("source_dataset", "unknown")]
        groups.setdefault(str(tags[0]), []).append(query)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0)); progressed = True
        if not progressed:
            break
    return selected


def validate(args: argparse.Namespace) -> dict[str, Any]:
    registry = _load_registry(Path(args.registry))
    queries = list(_records(Path(args.query_file)))
    if args.query_id:
        queries = [q for q in queries if q.get("query_id") == args.query_id]
    queries = stratified_query_sample(queries, args.query_limit)

    selector = ResourceAwareServerSelector(max_servers=args.max_servers)
    aggregator = EvidenceAggregator()
    results = []
    started = time.perf_counter()
    for index, query in enumerate(queries):
        query_id = str(query.get("query_id", f"query-{index}"))
        question = str(query.get("question", query.get("query", "")))
        tags = _query_tags(query)
        candidates = registry.candidates(tags)
        selected = selector.select(candidates, tags, RequestState(
            request_id=str(uuid.uuid4()), query_id=query_id, max_servers=args.max_servers,
        ))

        capsules = []
        retrieval_start = time.perf_counter()
        for server in selected.servers:
            chunk_path = Path(args.edge_dir) / server.server_id / "chunks.jsonl"
            if not chunk_path.exists():
                continue
            # Deliberately load only selected server partitions.
            edge = _edge_from_chunks(server.server_id, chunk_path)
            capsules.append(edge.retrieve_evidence(
                question, str(uuid.uuid4()), query_id, top_k=args.top_k,
                selected_tags=tuple(sorted(tags.level2)),
            ))
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        evidence = aggregator.aggregate(capsules, top_k=args.top_k)
        chunks = [{"chunk_id": c.source_id, "score": c.score, "tags": list(c.tags),
                   "server_id": next((cap.server_id for cap in capsules if c in cap.chunks), None)}
                  for c in evidence["chunks"]]
        results.append({
            "query_id": query_id,
            "query_tags": [{"level1": t.level1, "level2": t.level2} for t in tags.tags],
            "candidate_servers": [s.server_id for s in candidates],
            "selected_servers": [s.server_id for s in selected.servers],
            "chunks": chunks,
            "tags": list(evidence["tags"]),
            "evidence_count": len(evidence["chunks"]),
            "latency_statistics": {"retrieval_ms": round(retrieval_ms, 3),
                                    "selection_estimated_ms": selected.latency_ms},
            "empty_retrieval": not bool(evidence["chunks"]),
        })
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"queries": results, "summary": {"query_count": len(results),
            "empty_retrieval_count": sum(r["empty_retrieval"] for r in results),
            "elapsed_ms": round(elapsed_ms, 3)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--edge-dir", required=True)
    parser.add_argument("--query-id")
    parser.add_argument("--query-limit", type=int, default=12)
    parser.add_argument("--max-servers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.query_limit < 1 or args.max_servers < 1 or args.top_k < 1:
        parser.error("--query-limit, --max-servers and --top-k must be positive")
    print(json.dumps(validate(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
