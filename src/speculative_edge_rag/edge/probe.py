"""Privacy-safe inverted index for edge-local relevance probes."""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ..schemas import ProbeResult, QueryTags
from .local_kg import KnowledgeNode, LocalKnowledgeGraph


TOKEN_RE = re.compile(r"\w+",re.UNICODE)


def normalize_terms(text: str) -> tuple[str,...]:
    return tuple(TOKEN_RE.findall(text.casefold()))


@dataclass(frozen=True)
class LocalProbeChunk:
    chunk_id: str
    doc_id: str
    terms: frozenset[str]
    tag_pairs: frozenset[tuple[str,str]]


class LocalProbeIndex:
    """Built once per server; raw text is discarded after token indexing."""
    def __init__(self,chunks: Iterable[LocalProbeChunk]):
        self.chunks={chunk.chunk_id:chunk for chunk in chunks}
        term_postings: dict[str,set[str]]=defaultdict(set)
        tag_postings: dict[tuple[str,str],set[str]]=defaultdict(set)
        for chunk in self.chunks.values():
            for term in chunk.terms: term_postings[term].add(chunk.chunk_id)
            for pair in chunk.tag_pairs: tag_postings[pair].add(chunk.chunk_id)
        self.term_postings={term:frozenset(ids) for term,ids in term_postings.items()}
        self.tag_pair_postings={pair:frozenset(ids) for pair,ids in tag_postings.items()}

    @classmethod
    def from_graph(cls,graph: LocalKnowledgeGraph) -> "LocalProbeIndex":
        nodes=getattr(graph,"nodes",{})
        values=list(nodes.values()) if isinstance(nodes,dict) else list(nodes)
        chunk_nodes={node.node_id:node for node in values if str(node.metadata.get("type","chunk"))=="chunk"}
        level1_by_chunk: dict[str,set[str]]=defaultdict(set)
        level2_by_chunk: dict[str,set[str]]=defaultdict(set)
        for edge in getattr(graph,"edges",()):
            if edge.relation!="chunk_has_tag" or edge.source_id not in chunk_nodes: continue
            target=nodes.get(edge.target_id) if isinstance(nodes,dict) else None
            if target is None: continue
            kind=str(target.metadata.get("type","")); value=target.text
            if kind=="tag_level1": level1_by_chunk[edge.source_id].add(value)
            elif kind=="tag_level2": level2_by_chunk[edge.source_id].add(value)
        chunks=[]
        for node_id,node in chunk_nodes.items():
            level1=set(filter(None,str(node.metadata.get("level1_tags","")).split(',')))|level1_by_chunk[node_id]
            level2=set(node.tags)|level2_by_chunk[node_id]
            pairs={(l1,l2) for l1 in level1 for l2 in level2} or {("*",l2) for l2 in level2}
            chunks.append(LocalProbeChunk(node_id,str(node.metadata.get("doc_id","")),frozenset(normalize_terms(node.text)),frozenset(pairs)))
        return cls(chunks)

    def probe(self,server_id: str,query: str,query_tags: QueryTags,top_k: int=5) -> ProbeResult:
        started=time.perf_counter(); terms=normalize_terms(query); unique_terms=set(terms)
        requested={(tag.level1,tag.level2) for tag in query_tags.tags}
        tag_ids=set().union(*(self.tag_pair_postings.get(pair,frozenset())|self.tag_pair_postings.get(("*",pair[1]),frozenset()) for pair in requested)) if requested else set()
        term_ids=set().union(*(self.term_postings.get(term,frozenset()) for term in unique_terms)) if unique_terms else set()
        relevant=term_ids|tag_ids; scored=[]
        for chunk_id in relevant:
            chunk=self.chunks[chunk_id]
            lexical=len(unique_terms&chunk.terms)/max(1,len(unique_terms))
            tag_score=len(requested&chunk.tag_pairs)/max(1,len(requested))
            if not tag_score and any(("*",pair[1]) in chunk.tag_pairs for pair in requested): tag_score=1/max(1,len(requested))
            scored.append((0.8*lexical+0.2*tag_score,chunk_id))
        scored.sort(key=lambda item:(-item[0],item[1])); top=scored[:top_k]
        covered_positions=tuple(i for i,term in enumerate(terms) if self.term_postings.get(term))
        matched=tuple(sorted(pair for pair in requested if self.tag_pair_postings.get(pair) or self.tag_pair_postings.get(("*",pair[1]))))
        scores=[score for score,_ in top]; mean=sum(scores)/len(scores) if scores else 0.0
        signal=(len(set(covered_positions))/max(1,len(terms))+mean+(len(matched)/max(1,len(requested)) if requested else 0.0))/3
        return ProbeResult(server_id,matched,scores[0] if scores else 0.0,mean,len(relevant),covered_positions,tuple(chunk_id for _,chunk_id in top),(time.perf_counter()-started)*1000,max(0.0,1.0-min(1.0,signal)))
