"""Coordinator-side registry of server capabilities; no private documents stored."""
from __future__ import annotations
from ..schemas import ServerRecord, QueryTags

class ServerRegistry:
    def __init__(self) -> None: self._servers: dict[str, ServerRecord] = {}
    def register(self, server: ServerRecord) -> None: self._servers[server.server_id] = server
    def unregister(self, server_id: str) -> None: self._servers.pop(server_id, None)
    def get(self, server_id: str) -> ServerRecord: return self._servers[server_id]
    def all(self) -> list[ServerRecord]: return list(self._servers.values())
    def candidates(self, query_tags: QueryTags) -> list[ServerRecord]:
        return [s for s in self._servers.values() if s.available and (s.level1_tags & query_tags.level1 or s.level2_tags & query_tags.level2)]
