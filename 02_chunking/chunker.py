"""
Module 2 — Chunking Strategies
================================
Chunking = splitting a large document into smaller pieces before embedding.

Why it matters:
    • Embedding models have a token limit (e.g. 512 tokens)
    • Smaller, focused chunks retrieve more precisely
    • Too small → context lost; Too large → noisy retrieval

Strategies implemented here:
    1. FixedChunker       — split every N characters
    2. SentenceChunker    — split on sentence boundaries (NLTK)
    3. RecursiveChunker   — try paragraph → sentence → word splits (recommended)
    4. SemanticChunker    — split when topic changes (embedding-based)

All chunkers return a list of Chunk objects with text + metadata.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import nltk

# Add repo root to path
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import Chunk  # noqa: E402

# Download NLTK sentence tokenizer data (once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fixed-size chunker
# ─────────────────────────────────────────────────────────────────────────────

class FixedChunker:
    """
    Split text into fixed-size windows of `chunk_size` characters
    with `overlap` characters of overlap between consecutive chunks.

    Pros: Simple, predictable
    Cons: May cut sentences mid-way
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source: str = "", metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=idx,
                    metadata={**metadata, "strategy": "fixed"},
                ))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sentence-based chunker
# ─────────────────────────────────────────────────────────────────────────────

class SentenceChunker:
    """
    Split text by sentences (NLTK), then group N sentences per chunk.

    Pros: Preserves sentence boundaries, readable chunks
    Cons: Chunk sizes vary a lot
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def split(self, text: str, source: str = "", metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = nltk.sent_tokenize(text)
        chunks = []
        idx = 0
        step = self.sentences_per_chunk - self.overlap_sentences
        for i in range(0, len(sentences), max(step, 1)):
            window = sentences[i: i + self.sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=idx,
                    metadata={**metadata, "strategy": "sentence"},
                ))
                idx += 1
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recursive chunker (recommended default)
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveChunker:
    """
    Try to split on progressively smaller separators:
        \\n\\n (paragraphs) → \\n (lines) → .  (sentences) → ' ' (words)

    This is the same strategy used by LangChain's RecursiveCharacterTextSplitter
    and is the recommended default for most RAG pipelines.

    Pros: Respects natural text structure, consistent size
    Cons: Slightly more complex logic
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str, source: str = "", metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)
        # Merge small chunks and apply overlap
        merged = self._merge_chunks(raw_chunks)
        return [
            Chunk(
                text=c,
                source=source,
                chunk_index=i,
                metadata={**metadata, "strategy": "recursive"},
            )
            for i, c in enumerate(merged)
            if c.strip()
        ]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split on the best available separator."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            # Final fallback: character-level split
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining))
        return result

    def _merge_chunks(self, parts: list[str]) -> list[str]:
        """Merge small parts into chunks respecting chunk_size + overlap."""
        merged: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) + 1 <= self.chunk_size:
                current = (current + " " + part).strip()
            else:
                if current:
                    merged.append(current)
                # Include overlap from end of previous chunk
                if self.overlap > 0 and current:
                    overlap_text = current[-self.overlap:]
                    current = (overlap_text + " " + part).strip()
                else:
                    current = part

        if current:
            merged.append(current)
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# 4. Semantic chunker
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Split text when the topic changes, measured by cosine similarity
    between consecutive sentence embeddings.

    Pros: Best semantic coherence — each chunk is truly about one topic
    Cons: Requires embedding model at chunk time (slower)
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        threshold: float = 0.4,
        min_chunk_size: int = 100,
    ):
        """
        Parameters
        ----------
        embed_fn   : A callable that takes a list of strings and returns embeddings
        threshold  : Cosine distance threshold to trigger a new chunk (0–1)
        """
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

    def split(self, text: str, source: str = "", metadata: dict | None = None) -> list[Chunk]:
        import numpy as np

        metadata = metadata or {}
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return [Chunk(text=text, source=source, chunk_index=0, metadata=metadata)]

        embeddings = self.embed_fn(sentences)
        embeddings = np.array(embeddings)

        # Normalise for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        embeddings = embeddings / norms

        # Compute cosine similarity between consecutive sentences
        similarities = [
            float(embeddings[i] @ embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Split where similarity drops below threshold
        split_points = [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < (1 - self.threshold)
        ]

        # Build chunks from split points
        boundaries = [0] + split_points + [len(sentences)]
        chunks = []
        for ci, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            chunk_text = " ".join(sentences[start:end]).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=ci,
                    metadata={**metadata, "strategy": "semantic"},
                ))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def get_chunker(strategy: str, **kwargs):
    """Return a chunker by strategy name (reads from config)."""
    strategies = {
        "fixed": FixedChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
    }
    cls = strategies.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Choose from: {list(strategies.keys())}"
        )
    return cls(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Compare all chunking strategies on a sample text.
    Run: python 02_chunking/chunker.py
    """
    from rich.console import Console
    from rich.table import Table
    from pathlib import Path

    console = Console()
    console.rule("[bold blue]Module 2 — Chunking Strategies")

    # Load any processed markdown
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    md_files = list(processed_dir.glob("*.md"))

    if md_files:
        sample_text = md_files[0].read_text(encoding="utf-8")[:5000]
        source = md_files[0].name
    else:
        # Built-in sample text
        sample_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
the natural intelligence displayed by animals including humans.

AI research has been defined as the field of study of intelligent agents, which refers
to any system that perceives its environment and takes actions that maximize its chance
of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that
mimic and display "human" cognitive skills associated with the human mind, such as
"learning" and "problem-solving".

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based
and generation-based approaches for natural language processing tasks. It retrieves
relevant documents from a large corpus and uses them as context for generating answers.

Vector databases store high-dimensional vectors and support efficient similarity search.
They are a core component of modern RAG pipelines, enabling fast retrieval of semantically
relevant chunks from millions of documents.
        """ * 4
        source = "sample_text"

    strategies = {
        "Fixed (512 chars)": FixedChunker(chunk_size=512, overlap=64),
        "Sentence (5 sent)": SentenceChunker(sentences_per_chunk=5, overlap_sentences=1),
        "Recursive (512)":   RecursiveChunker(chunk_size=512, overlap=64),
    }

    table = Table(title="Chunking Strategy Comparison", show_lines=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("# Chunks", justify="right")
    table.add_column("Min chars", justify="right")
    table.add_column("Max chars", justify="right")
    table.add_column("Avg chars", justify="right")
    table.add_column("Sample chunk (first 80 chars)")

    for name, chunker in strategies.items():
        chunks = chunker.split(sample_text, source=source)
        sizes = [len(c.text) for c in chunks]
        if not sizes:
            continue
        table.add_row(
            name,
            str(len(chunks)),
            str(min(sizes)),
            str(max(sizes)),
            f"{sum(sizes) / len(sizes):.0f}",
            chunks[0].text[:80].replace("\n", " "),
        )

    console.print(table)
    console.print(f"\n[bold green]✓ Chunking demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
