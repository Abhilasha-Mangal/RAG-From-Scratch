"""
Shared data classes used across all RAG modules.
Keeping them here avoids circular imports between numbered modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single loaded document (content + metadata)."""
    content: str
    source: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:120].replace("\n", " ")
        return (
            f"Document(source={self.source!r}, "
            f"chars={len(self.content)}, "
            f"preview={preview!r}...)"
        )


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(idx={self.chunk_index}, "
            f"chars={len(self.text)}, "
            f"text={preview!r}...)"
        )
