"""Secure evidence context baseline (sanitization and aggregation only)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from typing import Iterable
from .retriever import RetrievalResult

@dataclass(frozen=True)
class EvidenceChunk:
    text: str
    source_id: str
    score: float
    tags: tuple[str, ...] = ()
    provenance: str = "local"

@dataclass(frozen=True)
class EvidenceCapsule:
    request_id: str
    server_id: str
    query_id: str
    selected_tags: tuple[str, ...]
    chunks: tuple[EvidenceChunk, ...]
    provenance: str
    retrieval_score: float
    expires_at: datetime

    @property
    def evidence_chunks(self) -> tuple[EvidenceChunk, ...]:
        """Compatibility alias for callers using the protocol terminology."""
        return self.chunks

    @property
    def expiration(self) -> datetime:
        return self.expires_at

class EvidenceSanitizer:
    """Remove simple sensitive fields; replace values with placeholders."""
    PATTERNS = [(re.compile(r'(?i)(ssn|身份证|email|phone)\s*[:=]\s*[^,;\s]+'), r'\1=[REDACTED]')]
    def sanitize(self, text: str) -> str:
        for pattern, repl in self.PATTERNS: text = pattern.sub(repl, text)
        return text
    def chunks(self, result: RetrievalResult, top_k: int = 3) -> list[EvidenceChunk]:
        return [EvidenceChunk(self.sanitize(n.text), n.node_id, result.score, n.tags) for n in result.nodes[:top_k]]

class EvidenceBuilder:
    def __init__(self, sanitizer: EvidenceSanitizer | None = None): self.sanitizer = sanitizer or EvidenceSanitizer()
    def build(self, request_id: str, query_id: str, server_id: str, result: RetrievalResult, top_k: int = 3, ttl_seconds: int = 300) -> EvidenceCapsule:
        return EvidenceCapsule(request_id, server_id, query_id, result.covered_tags, tuple(self.sanitizer.chunks(result, top_k)), "local", result.score, datetime.now(timezone.utc)+timedelta(seconds=ttl_seconds))

class EvidenceAggregator:
    def aggregate(self, capsules: Iterable[EvidenceCapsule], top_k: int = 8) -> dict[str, object]:
        caps = list(capsules); chunks = [c for c in caps for c in c.chunks]
        chunks.sort(key=lambda c: c.score, reverse=True)
        return {"chunks": tuple(chunks[:top_k]), "servers": tuple(c.server_id for c in caps), "tags": tuple(sorted({t for c in caps for t in c.selected_tags}))}
