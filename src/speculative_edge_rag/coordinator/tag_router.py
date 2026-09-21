"""Hierarchical multi-label query tagging baselines."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Mapping, Iterable
from ..schemas import QueryTag, QueryTags, TagSource

TAXONOMY: Mapping[str, tuple[str, ...]] = {
 "encyclopedic": ("person","organization","location","event","history","science"),
 "literature": ("character","event","relationship","motivation","theme","setting","summary"),
 "legal": ("statute","case","right","obligation","liability","procedure","penalty","remedy"),
 "medical": ("disease","symptom","risk","diagnosis","test","treatment","drug","prognosis"),
}

class HierarchicalMultiLabelTagger(ABC):
    @abstractmethod
    def tag(self, query: str, oracle_tags: Iterable[QueryTag] | None = None) -> QueryTags: ...

class OracleTagger(HierarchicalMultiLabelTagger):
    def tag(self, query: str, oracle_tags: Iterable[QueryTag] | None = None) -> QueryTags:
        if isinstance(query, Mapping) and oracle_tags is None:
            oracle_tags = [QueryTag(l1, l2) for l1 in query.get("level1", ()) for l2 in query.get("level2", ())]
        if oracle_tags is None: raise ValueError("OracleTagger requires oracle_tags")
        return QueryTags([QueryTag(t.level1, t.level2, t.confidence, TagSource.ORACLE) for t in oracle_tags])

class RuleBasedTagger(HierarchicalMultiLabelTagger):
    KEYWORDS = {
      "legal": {"law":"statute", "statute":"statute", "court":"case", "contract":"obligation", "penalty":"penalty", "right":"right"},
      "medical": {"disease":"disease", "symptom":"symptom", "diagnosis":"diagnosis", "treatment":"treatment", "drug":"drug", "risk":"risk"},
      "literature": {"novel":"summary", "character":"character", "plot":"event", "theme":"theme", "story":"summary"},
      "encyclopedic": {"who":"person", "where":"location", "when":"event", "history":"history", "science":"science", "company":"organization"},
    }
    def tag(self, query: str, oracle_tags: Iterable[QueryTag] | None = None) -> QueryTags:
        q = query.lower(); out: list[QueryTag] = []
        for l1, kws in self.KEYWORDS.items():
            hits = [sub for key, sub in kws.items() if key in q]
            for sub in dict.fromkeys(hits): out.append(QueryTag(l1, sub, 0.7, TagSource.PREDICTED))
        if not out: out.append(QueryTag("encyclopedic", "summary", 0.2, TagSource.PREDICTED))
        return QueryTags(out)
