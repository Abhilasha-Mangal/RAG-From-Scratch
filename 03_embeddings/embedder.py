"""
Module 3 — Embeddings
======================
An embedding is a vector of numbers that captures the *meaning* of text.
Similar texts produce similar vectors (small cosine distance).

Providers implemented:
    • LocalEmbedder  — sentence-transformers (100% free, runs on CPU)
    • OpenAIEmbedder — text-embedding-ada-002 / text-embedding-3-small
    • OllamaEmbedder — any model pulled into local Ollama instance

All return the same interface: embed(texts) → list[list[float]]
The EmbedderFactory reads config.yaml to pick the right provider.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Common interface for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Local embedder — sentence-transformers (FREE, no API key)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Embed text using a local sentence-transformers model.

    Recommended models (all free, run on CPU):
    ┌─────────────────────────────────┬──────┬────────┬────────────────────┐
    │ Model                           │ Dim  │ Size   │ Use case           │
    ├─────────────────────────────────┼──────┼────────┼────────────────────┤
    │ all-MiniLM-L6-v2                │ 384  │ 90 MB  │ Fastest, great for │
    │                                 │      │        │ beginners          │
    │ BAAI/bge-small-en-v1.5          │ 384  │ 133 MB │ Better quality     │
    │ BAAI/bge-large-en-v1.5          │ 1024 │ 1.3 GB │ Best quality       │
    │ sentence-transformers/          │ 768  │ 420 MB │ Balanced           │
    │   all-mpnet-base-v2             │      │        │                    │
    └─────────────────────────────────┴──────┴────────┴────────────────────┘
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        print(f"[embed] Loading local model: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[embed] Model ready. Dimension: {self._dim}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# 2. OpenAI embedder
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    Embed text using OpenAI's API.
    Models: text-embedding-ada-002 (1536d) | text-embedding-3-small (1536d)
    Requires: OPENAI_API_KEY environment variable.
    """

    _DIMS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._dim = self._DIMS.get(model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI limits: 2048 strings per request
        results = []
        for i in range(0, len(texts), 2048):
            batch = texts[i: i + 2048]
            response = self._client.embeddings.create(model=self._model, input=batch)
            results.extend([item.embedding for item in response.data])
        return results

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ollama embedder — local LLM server
# ─────────────────────────────────────────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    """
    Embed using a model running in Ollama (local server, free).
    Pull a model first: `ollama pull nomic-embed-text`

    Good embedding models for Ollama:
        nomic-embed-text   (768d, excellent)
        mxbai-embed-large  (1024d, state-of-the-art)
        all-minilm         (384d, fastest)
    """

    _DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        import ollama as _ollama
        self._ollama = _ollama
        self._model = model
        self._base_url = base_url
        self._dim = self._DIMS.get(model, 768)

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            resp = self._ollama.embeddings(model=self._model, prompt=text)
            results.append(resp["embedding"])
        return results

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder(config: dict) -> BaseEmbedder:
    """
    Build an embedder from a config dict (embeddings section of config.yaml).

    Example config:
        embeddings:
          provider: local
          local:
            model: all-MiniLM-L6-v2
          batch_size: 64
    """
    provider = config.get("provider", "local")
    batch_size = config.get("batch_size", 64)

    if provider == "local":
        model = config.get("local", {}).get("model", "all-MiniLM-L6-v2")
        return LocalEmbedder(model_name=model, batch_size=batch_size)

    if provider == "openai":
        model = config.get("openai", {}).get("model", "text-embedding-3-small")
        return OpenAIEmbedder(model=model)

    if provider == "ollama":
        cfg = config.get("ollama", {})
        return OllamaEmbedder(
            model=cfg.get("model", "nomic-embed-text"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
        )

    raise ValueError(f"Unknown embedding provider: {provider!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo + Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Embed sample sentences and visualise with PCA.
    Run: python 03_embeddings/embedder.py
    """
    from rich.console import Console
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    console = Console()
    console.rule("[bold blue]Module 3 — Embeddings")

    # Sample sentences covering different topics
    sentences = [
        # RAG / AI
        "Retrieval-Augmented Generation combines retrieval with language models.",
        "Vector databases store embeddings for semantic search.",
        "Large language models can hallucinate facts not in their training data.",
        "Embeddings convert text into high-dimensional numerical vectors.",
        # Biology
        "Photosynthesis is the process by which plants use sunlight to make food.",
        "DNA carries genetic information in living organisms.",
        "Mitochondria are the powerhouses of the cell.",
        "Evolution occurs through natural selection over many generations.",
        # History
        "The French Revolution began in 1789 and ended the monarchy.",
        "World War II ended in 1945 with the Allied victory.",
        "The Roman Empire was one of the largest in ancient history.",
        "The Renaissance was a cultural revival in 14th–17th century Europe.",
    ]

    labels = ["RAG"] * 4 + ["Biology"] * 4 + ["History"] * 4
    colors = {"RAG": "#3b82d4", "Biology": "#16a34a", "History": "#d97706"}

    console.print("[cyan]Loading local embedding model...[/cyan]")
    embedder = LocalEmbedder("all-MiniLM-L6-v2")

    console.print(f"[cyan]Embedding {len(sentences)} sentences...[/cyan]")
    vectors = embedder.embed(sentences)

    # Cosine similarity table
    console.print("\n[bold]Cosine Similarities (sample):[/bold]")
    pairs = [
        (0, 1, "RAG1 vs RAG2"),
        (0, 4, "RAG1 vs Bio1"),
        (4, 5, "Bio1 vs Bio2"),
        (4, 8, "Bio1 vs History1"),
        (8, 9, "History1 vs History2"),
    ]
    for i, j, label in pairs:
        sim = BaseEmbedder.cosine_similarity(vectors[i], vectors[j])
        bar = "█" * int(sim * 30)
        console.print(f"  {label:<28} {sim:.3f} {bar}")

    # PCA visualisation
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (x, y) in enumerate(coords):
        lbl = labels[i]
        ax.scatter(x, y, color=colors[lbl], s=120, zorder=3)
        ax.annotate(
            sentences[i][:35] + "…",
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
            color=colors[lbl],
        )

    # Legend
    for lbl, col in colors.items():
        ax.scatter([], [], color=col, label=lbl, s=80)
    ax.legend(loc="lower right")
    ax.set_title("Sentence Embeddings — PCA (2D)\nSame-topic sentences cluster together")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = "data/processed/embeddings_pca.png"
    plt.savefig(out, dpi=150)
    console.print(f"\n[green]PCA plot saved → {out}[/green]")
    plt.show()
    console.print(f"\n[bold green]✓ Embedding demo complete. Dimension = {embedder.dimension}[/bold green]")


if __name__ == "__main__":
    demo()
