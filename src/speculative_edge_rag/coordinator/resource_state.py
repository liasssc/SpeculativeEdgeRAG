"""Helpers for estimating parallel edge-stage latency."""
from ..schemas import ServerRecord
def estimated_latency_ms(server: ServerRecord) -> float:
    congestion = server.queue_length * max(server.draft_latency_ms, 1.0) * 0.1
    return server.network_latency_ms + server.retrieval_latency_ms + server.draft_latency_ms + congestion
def subset_latency_ms(servers: list[ServerRecord]) -> float:
    return max((estimated_latency_ms(s) for s in servers), default=0.0)
