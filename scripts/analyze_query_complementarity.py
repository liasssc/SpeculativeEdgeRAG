#!/usr/bin/env python3
"""Evaluate query-conditioned edge-server complementarity without using gold labels for routing."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speculative_edge_rag.coordinator.server_registry import ServerRegistry
from speculative_edge_rag.coordinator.server_selector import ResourceAwareServerSelector
from speculative_edge_rag.coordinator.tag_router import OracleTagger
from speculative_edge_rag.schemas import RequestState, ServerRecord


METHODS = ("current_selector", "single_server_oracle", "all_domain_servers", "gold_server_oracle")


def records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_registry(path: Path) -> tuple[ServerRegistry, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    registry = ServerRegistry()
    capabilities: dict[str, dict[str, Any]] = {}
    for raw in payload.get("servers", []):
        state = raw.get("resource_state") or {}
        quality = (raw.get("quality") or {}).get("score")
        server_id = str(raw["server_id"])
        registry.register(ServerRecord(
            server_id=server_id,
            level1_tags=set(raw.get("level1_tags") or []),
            level2_tags=set(raw.get("level2_tags") or []),
            knowledge_quality=float(quality) if quality is not None else 0.5,
            queue_length=int(state.get("queue_length", 0)),
            load=float(state.get("load", 0.0)),
            kv_cache_utilization=float(state.get("kv_cache_utilization", 0.0)),
            retrieval_latency_ms=float(raw.get("retrieval_latency_ms", 50.0)),
            draft_latency_ms=float(raw.get("draft_latency_ms", 50.0)),
            network_latency_ms=float(raw.get("network_latency_ms", 20.0)),
            available=bool(state.get("available", state.get("status") != "offline")),
            metadata={"document_count": raw.get("document_count", 0), "chunk_count": raw.get("chunk_count", 0)},
        ))
        capabilities[server_id] = {
            "level1_tags": sorted(raw.get("level1_tags") or []),
            "level2_tags": sorted(raw.get("level2_tags") or []),
        }
    return registry, capabilities


def load_partition_indexes(edge_dir: Path, partition_stats_path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    doc_servers: dict[str, set[str]] = defaultdict(set)
    chunk_server: dict[str, str] = {}
    for server_dir in sorted(edge_dir.iterdir()):
        if not server_dir.is_dir():
            continue
        server_id = server_dir.name
        for document in records(server_dir / "documents.jsonl"):
            doc_servers[str(document["doc_id"])].add(server_id)
        for chunk in records(server_dir / "chunks.jsonl"):
            chunk_id = str(chunk["chunk_id"])
            previous = chunk_server.setdefault(chunk_id, server_id)
            if previous != server_id:
                raise ValueError(f"chunk {chunk_id} is assigned to both {previous} and {server_id}")

    stats = json.loads(partition_stats_path.read_text(encoding="utf-8"))
    expected_docs = {str(doc_id): set(map(str, servers)) for doc_id, servers in stats.get("document_assignments", {}).items()}
    expected_chunks = {str(chunk_id): str(server_id) for chunk_id, server_id in stats.get("chunk_assignments", {}).items()}
    if expected_docs and dict(doc_servers) != expected_docs:
        raise ValueError("edge document index does not match partition_stats document_assignments")
    if expected_chunks and chunk_server != expected_chunks:
        raise ValueError("edge chunk index does not match partition_stats chunk_assignments")
    return dict(doc_servers), chunk_server


def query_tags(level1_tags: Iterable[str], level2_tags: Iterable[str]):
    return OracleTagger().tag({
        "level1": list(level1_tags),
        "level2": list(level2_tags),
    })


def current_selector_choice(
    query_id: str, level1_tags: Iterable[str], level2_tags: Iterable[str],
    registry: ServerRegistry, selector: ResourceAwareServerSelector,
    *, max_servers: int = 4, deadline_ms: float = 1000.0,
) -> tuple[list[str], list[str]]:
    """Route from non-gold query fields; gold labels cannot enter this function."""
    tags = query_tags(level1_tags,level2_tags)
    candidates = registry.candidates(tags)
    result = selector.select(candidates, tags, RequestState(
        request_id=str(uuid.uuid4()), query_id=query_id,
        max_servers=max_servers, deadline_ms=deadline_ms,
    ))
    return [server.server_id for server in candidates], [server.server_id for server in result.servers]


def covered_documents(selected: Iterable[str], gold_docs: list[str], doc_servers: dict[str, set[str]]) -> set[str]:
    selected_set = set(selected)
    return {doc_id for doc_id in gold_docs if doc_servers.get(doc_id, set()) & selected_set}


def minimum_cover(gold_docs: list[str], doc_servers: dict[str, set[str]]) -> tuple[int | None, list[str]]:
    mapped_docs = [doc_id for doc_id in gold_docs if doc_servers.get(doc_id)]
    if len(mapped_docs) != len(gold_docs):
        return None, []
    servers = sorted(set().union(*(doc_servers[doc_id] for doc_id in mapped_docs))) if mapped_docs else []
    for size in range(1, len(servers) + 1):
        for subset in combinations(servers, size):
            if len(covered_documents(subset, mapped_docs, doc_servers)) == len(mapped_docs):
                return size, list(subset)
    return (0, []) if not mapped_docs else (None, [])


def greedy_gold_order(servers: Iterable[str], gold_docs: list[str], doc_servers: dict[str, set[str]]) -> list[str]:
    remaining = set(servers); covered: set[str] = set(); ordered: list[str] = []
    while remaining:
        server_id = min(remaining, key=lambda sid: (-len(covered_documents([sid], gold_docs, doc_servers)-covered), sid))
        ordered.append(server_id); covered |= covered_documents([server_id], gold_docs, doc_servers); remaining.remove(server_id)
    return ordered


def method_evaluation(selected: list[str], gold_docs: list[str], doc_servers: dict[str, set[str]], gold_servers: set[str]) -> dict[str, Any]:
    covered = covered_documents(selected, gold_docs, doc_servers)
    mapped_docs = {doc_id for doc_id in gold_docs if doc_servers.get(doc_id)}
    selected_set = set(selected)
    first = covered_documents(selected[:1], gold_docs, doc_servers)
    first_two = covered_documents(selected[:2], gold_docs, doc_servers)
    return {
        "selected_servers": selected,
        "selected_server_count": len(selected),
        "supporting_documents_covered": len(covered),
        "supporting_document_recall": len(covered) / len(gold_docs) if gold_docs else 0.0,
        "mapped_supporting_document_recall": len(covered) / len(mapped_docs) if mapped_docs else 0.0,
        "gold_server_recall": len(selected_set & gold_servers) / len(gold_servers) if gold_servers else 0.0,
        "misses_required_evidence": bool(mapped_docs - covered),
        "missed_gold_servers": sorted(gold_servers - selected_set),
        "unnecessary_servers": sorted(selected_set - gold_servers),
        "second_server_marginal_supporting_document_gain": len(first_two - first) if len(selected) >= 2 else None,
    }


def analyze_query(
    query: dict[str, Any], registry: ServerRegistry, capabilities: dict[str, dict[str, Any]],
    doc_servers: dict[str, set[str]], selector: ResourceAwareServerSelector,
    *, max_servers: int = 4, deadline_ms: float = 1000.0,
) -> dict[str, Any]:
    gold_docs = list(dict.fromkeys(map(str, query.get("supporting_doc_ids") or [])))
    unmapped = [doc_id for doc_id in gold_docs if not doc_servers.get(doc_id)]
    gold_servers = set().union(*(doc_servers.get(doc_id, set()) for doc_id in gold_docs)) if gold_docs else set()
    minimum_count, minimum_servers = minimum_cover(gold_docs, doc_servers)

    candidates, current = current_selector_choice(
        str(query.get("query_id","query")),query.get("level1_tags") or [],query.get("level2_tags") or [],
        registry,selector,max_servers=max_servers,deadline_ms=deadline_ms,
    )
    single = greedy_gold_order(gold_servers, gold_docs, doc_servers)[:1]
    level1 = set(query.get("level1_tags") or [])
    all_domain = sorted(server_id for server_id, capability in capabilities.items() if level1 & set(capability["level1_tags"]))
    gold_oracle = greedy_gold_order(gold_servers, gold_docs, doc_servers)
    choices = {
        "current_selector": current,
        "single_server_oracle": single,
        "all_domain_servers": all_domain,
        "gold_server_oracle": gold_oracle,
    }
    evaluations = {name: method_evaluation(choice, gold_docs, doc_servers, gold_servers) for name, choice in choices.items()}
    return {
        "query_id": str(query.get("query_id", "")),
        "source_dataset": str(query.get("source_dataset", "")),
        "query_type": str(query.get("query_type", "")),
        "level1_tags": list(query.get("level1_tags") or []),
        "level2_tags": list(query.get("level2_tags") or []),
        "level2_tag_count": len(set(query.get("level2_tags") or [])),
        "gold_supporting_documents": gold_docs,
        "gold_supporting_document_count": len(gold_docs),
        "unmapped_supporting_documents": unmapped,
        "unmapped_supporting_document_count": len(unmapped),
        "gold_server_set": sorted(gold_servers),
        "gold_server_count": len(gold_servers),
        "minimum_required_server_count": minimum_count,
        "minimum_cover_servers": minimum_servers,
        "candidate_servers": candidates,
        "methods": evaluations,
    }


def aggregate(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "query_count": len(query_results),
        "gold_supporting_document_count": sum(row["gold_supporting_document_count"] for row in query_results),
        "unmapped_supporting_document_count": sum(row["unmapped_supporting_document_count"] for row in query_results),
        "gold_server_count_distribution": dict(sorted(Counter(str(row["gold_server_count"]) for row in query_results).items())),
        "minimum_required_server_count_distribution": dict(sorted(Counter("unmappable" if row["minimum_required_server_count"] is None else str(row["minimum_required_server_count"]) for row in query_results).items())),
        "methods": {},
    }
    for method in METHODS:
        evaluations = [row["methods"][method] for row in query_results]
        selected_count = sum(item["selected_server_count"] for item in evaluations)
        unnecessary_count = sum(len(item["unnecessary_servers"]) for item in evaluations)
        second = [item["second_server_marginal_supporting_document_gain"] for item in evaluations if item["second_server_marginal_supporting_document_gain"] is not None]
        multi = [(row, item) for row, item in zip(query_results, evaluations) if row["minimum_required_server_count"] is not None and row["minimum_required_server_count"] >= 2]
        output["methods"][method] = {
            "supporting_document_recall": sum(item["supporting_document_recall"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "gold_server_recall": sum(item["gold_server_recall"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "average_selected_server_count": selected_count / len(evaluations) if evaluations else 0.0,
            "multi_server_required_query_count": len(multi),
            "multi_server_required_query_recall": sum(item["supporting_document_recall"] == 1.0 for _,item in multi) / len(multi) if multi else 0.0,
            "unnecessary_server_rate": unnecessary_count / selected_count if selected_count else 0.0,
            "queries_with_unnecessary_servers_rate": sum(bool(item["unnecessary_servers"]) for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "second_server_evaluated_query_count": len(second),
            "average_second_server_marginal_supporting_document_gain": sum(second) / len(second) if second else 0.0,
            "second_server_positive_gain_query_rate": sum(gain > 0 for gain in second) / len(second) if second else 0.0,
        }
    return output


def grouped(query_results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_results:
        groups[str(row[field])].append(row)
    return {key: aggregate(value) for key,value in sorted(groups.items())}


def representative_examples(query_results: list[dict[str, Any]], limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    examples = {"single_server_correct": [], "misses_complementary_server": [], "unnecessary_server": [], "unmapped_supporting_document": []}
    for row in query_results:
        current = row["methods"]["current_selector"]
        compact = {
            "query_id": row["query_id"], "source_dataset": row["source_dataset"], "query_type": row["query_type"],
            "gold_supporting_documents": row["gold_supporting_documents"], "gold_server_set": row["gold_server_set"],
            "minimum_required_server_count": row["minimum_required_server_count"], "candidate_servers": row["candidate_servers"],
            "selected_servers": current["selected_servers"], "supporting_document_recall": current["supporting_document_recall"],
            "unnecessary_servers": current["unnecessary_servers"], "unmapped_supporting_documents": row["unmapped_supporting_documents"],
        }
        if len(current["selected_servers"]) == 1 and current["supporting_document_recall"] == 1.0 and len(examples["single_server_correct"]) < limit:
            examples["single_server_correct"].append(compact)
        if row["minimum_required_server_count"] and row["minimum_required_server_count"] >= 2 and current["supporting_document_recall"] < 1.0 and len(examples["misses_complementary_server"]) < limit:
            examples["misses_complementary_server"].append(compact)
        if current["unnecessary_servers"] and len(examples["unnecessary_server"]) < limit:
            examples["unnecessary_server"].append(compact)
        if row["unmapped_supporting_documents"] and len(examples["unmapped_supporting_document"]) < limit:
            examples["unmapped_supporting_document"].append(compact)
    return examples


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    registry, capabilities = load_registry(Path(args.registry))
    doc_servers, chunk_server = load_partition_indexes(Path(args.edge_dir), Path(args.partition_stats))
    queries = [query for query in records(Path(args.query_file)) if query.get("supporting_doc_ids")]
    selector = ResourceAwareServerSelector(max_servers=args.max_servers)
    results = [analyze_query(query, registry, capabilities, doc_servers, selector, max_servers=args.max_servers, deadline_ms=args.deadline_ms) for query in queries]
    report = {
        "schema_version": 1,
        "evaluation_constraint": "supporting_doc_ids are used only after candidate generation and selection",
        "index_summary": {"document_count": len(doc_servers), "chunk_count": len(chunk_server), "server_count": len(capabilities), "server_capabilities": capabilities},
        "overall": aggregate(results),
        "groups": {
            "source_dataset": grouped(results,"source_dataset"),
            "query_type": grouped(results,"query_type"),
            "gold_server_count": grouped(results,"gold_server_count"),
            "level2_tag_count": grouped(results,"level2_tag_count"),
        },
        "examples": representative_examples(results),
        "queries": results,
    }
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file",required=True)
    parser.add_argument("--partition-stats",required=True)
    parser.add_argument("--registry",required=True)
    parser.add_argument("--edge-dir",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-servers",type=int,default=4)
    parser.add_argument("--deadline-ms",type=float,default=1000.0)
    args=parser.parse_args()
    report=run_analysis(args)
    print(json.dumps({"output":args.output,"query_count":report["overall"]["query_count"],"methods":report["overall"]["methods"]},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
