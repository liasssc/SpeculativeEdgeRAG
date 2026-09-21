"""Committed prefix state for iterative speculative decoding."""
from dataclasses import dataclass, field

@dataclass
class CommittedPrefix:
    tokens: tuple[str, ...] = ()
    def append(self, token: str) -> "CommittedPrefix":
        self.tokens = self.tokens + (token,)
        return self
    def extend(self, tokens: list[str]) -> None: self.tokens = self.tokens + tuple(tokens)
    def text(self) -> str: return " ".join(self.tokens)
