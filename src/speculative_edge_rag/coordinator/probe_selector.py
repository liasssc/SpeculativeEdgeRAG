"""Probe-aware, non-gold server-selection baseline."""
from __future__ import annotations

from dataclasses import dataclass,replace
from itertools import combinations
from pathlib import Path
from typing import Any,Mapping

import yaml

from ..schemas import ProbeResult,QueryTags,ServerRecord


@dataclass(frozen=True)
class ProbeSelectorConfig:
    tag_pair_weight: float
    term_coverage_weight: float
    top_relevance_weight: float
    mean_relevance_weight: float
    evidence_mass_weight: float
    registry_coverage_weight: float
    overlap_penalty: float
    server_count_penalty: float
    marginal_gain_threshold: float
    evidence_mass_scale: float
    top_k: int
    calibration_marginal_gain_thresholds: tuple[float,...]
    calibration_server_count_penalties: tuple[float,...]
    calibration_overlap_penalties: tuple[float,...]
    objective_recall_weight: float
    objective_unnecessary_weight: float
    objective_server_count_weight: float

    @classmethod
    def from_mapping(cls,raw: Mapping[str,Any]) -> "ProbeSelectorConfig":
        tuning=raw.get("calibration",{}); objective=tuning.get("objective",{})
        return cls(
            **{key:raw[key] for key in ("tag_pair_weight","term_coverage_weight","top_relevance_weight","mean_relevance_weight","evidence_mass_weight","registry_coverage_weight","overlap_penalty","server_count_penalty","marginal_gain_threshold","evidence_mass_scale","top_k")},
            calibration_marginal_gain_thresholds=tuple(tuning["marginal_gain_thresholds"]),
            calibration_server_count_penalties=tuple(tuning["server_count_penalties"]),
            calibration_overlap_penalties=tuple(tuning["overlap_penalties"]),
            objective_recall_weight=float(objective["recall_weight"]),objective_unnecessary_weight=float(objective["unnecessary_weight"]),objective_server_count_weight=float(objective["server_count_weight"]),
        )

    @classmethod
    def from_yaml(cls,path: str|Path) -> "ProbeSelectorConfig":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def tuned(self,**changes: float) -> "ProbeSelectorConfig": return replace(self,**changes)


@dataclass(frozen=True)
class ProbeSelectionResult:
    servers: tuple[ServerRecord,...]
    score: float
    explanation: str
    features: Mapping[str,float]


class ProbeAwareServerSelector:
    def __init__(self,config: ProbeSelectorConfig,max_servers: int=4): self.config=config; self.max_servers=max_servers

    @staticmethod
    def _jaccard(left: set[Any],right: set[Any]) -> float:
        return len(left&right)/len(left|right) if left|right else 0.0

    def _features(self,subset: tuple[ServerRecord,...],probes: Mapping[str,ProbeResult],query_tags: QueryTags,query_term_count: int) -> dict[str,float]:
        selected=[probes[server.server_id] for server in subset]
        required={(tag.level1,tag.level2) for tag in query_tags.tags}
        matched=set().union(*(set(probe.matched_tag_pairs) for probe in selected))
        terms=set().union(*(set(probe.query_term_coverage) for probe in selected))
        overlaps=[]
        for left,right in combinations(selected,2):
            term_overlap=self._jaccard(set(left.query_term_coverage),set(right.query_term_coverage))
            tag_overlap=self._jaccard(set(left.matched_tag_pairs),set(right.matched_tag_pairs))
            overlaps.append((term_overlap+tag_overlap)/2)
        ratios=[]
        for tag in sorted(query_tags.level2):
            ratios.append(max((float(server.metadata.get("level2_coverage_ratios",{}).get(tag,0.0)) for server in subset),default=0.0))
        return {
            "tag_pair_coverage":len(matched&required)/len(required) if required else 1.0,
            "query_term_union_coverage":len(terms)/max(1,query_term_count),
            "top_relevance":max((probe.top_relevance_score for probe in selected),default=0.0),
            "mean_top_k_relevance":max((probe.mean_top_k_score for probe in selected),default=0.0),
            "evidence_mass":min(1.0,sum(probe.relevant_chunk_count for probe in selected)/self.config.evidence_mass_scale),
            "registry_level2_coverage":sum(ratios)/len(ratios) if ratios else 0.0,
            "probe_overlap":sum(overlaps)/len(overlaps) if overlaps else 0.0,
            "server_count":float(len(subset)),
        }

    def _score(self,features: Mapping[str,float]) -> float:
        c=self.config
        return c.tag_pair_weight*features["tag_pair_coverage"]+c.term_coverage_weight*features["query_term_union_coverage"]+c.top_relevance_weight*features["top_relevance"]+c.mean_relevance_weight*features["mean_top_k_relevance"]+c.evidence_mass_weight*features["evidence_mass"]+c.registry_coverage_weight*features["registry_level2_coverage"]-c.overlap_penalty*features["probe_overlap"]-c.server_count_penalty*features["server_count"]

    def select(self,candidates: list[ServerRecord],query_tags: QueryTags,probes: Mapping[str,ProbeResult],query_term_count: int,max_servers: int|None=None) -> ProbeSelectionResult:
        limit=min(max_servers or self.max_servers,self.max_servers,len(candidates)); scored: dict[tuple[str,...],tuple[float,dict[str,float],tuple[ServerRecord,...]]]={}; rejected=[]
        for size in range(1,limit+1):
            for subset in combinations(candidates,size):
                key=tuple(server.server_id for server in subset); features=self._features(subset,probes,query_tags,query_term_count); score=self._score(features)
                if size>1:
                    parents=[scored[tuple(server.server_id for server in parent_subset)][0] for parent_subset in combinations(subset,size-1) if tuple(server.server_id for server in parent_subset) in scored]
                    if not parents: continue
                    parent=max(parents)
                    marginal=score-parent
                    if marginal<self.config.marginal_gain_threshold:
                        rejected.append((marginal,key)); continue
                scored[key]=(score,features,subset)
        if not scored: return ProbeSelectionResult((),float("-inf"),"no probed candidate",{})
        key,(score,features,subset)=max(scored.items(),key=lambda item:(item[1][0],-len(item[0]),tuple(reversed(item[0]))))
        singleton=min((scored[(server.server_id,)] for server in subset),key=lambda value:(-value[0],value[2][0].server_id))
        first=singleton[2][0].server_id
        ordered=[singleton[2][0]]
        remaining=[server for server in subset if server.server_id!=first]
        while remaining:
            next_server=max(remaining,key=lambda server:(self._score(self._features(tuple(ordered+[server]),probes,query_tags,query_term_count)),-candidates.index(server)))
            ordered.append(next_server); remaining.remove(next_server)
        if len(subset)>1:
            first_score=singleton[0]; second=ordered[1].server_id
            second_features=self._features(tuple(ordered[:2]),probes,query_tags,query_term_count); marginal=self._score(second_features)-first_score
            added_terms=second_features["query_term_union_coverage"]-singleton[1]["query_term_union_coverage"]
            added_tags=second_features["tag_pair_coverage"]-singleton[1]["tag_pair_coverage"]
            explanation=f"first={first} strongest singleton score={first_score:.4f}; second={second} marginal_score={marginal:.4f}, term_gain={added_terms:.4f}, tag_gain={added_tags:.4f}; stopped at {len(subset)} server(s)"
        else:
            best_rejected=max((item[0] for item in rejected),default=float("-inf"))
            explanation=f"first={first} strongest singleton score={score:.4f}; no second server cleared marginal threshold={self.config.marginal_gain_threshold:.4f} (best={best_rejected:.4f})"
        return ProbeSelectionResult(tuple(ordered),score,explanation,features)
