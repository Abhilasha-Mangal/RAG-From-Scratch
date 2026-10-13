"""
Module 4a — ChromaDB Vector Store
===================================
ChromaDB is the easiest vector database to get started with.
Zero server setup needed — data persists to a local folder.

What you will learn:
    • Creating a collection (like a table in SQL)
    • Upserting embeddings + text + metadata
    • Querying by similarity
    • Filtering by metadata (e.g. only search within one book)
    • Deleting and updating documents

ChromaDB docs: https://docs.trychroma.com
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import Chunk  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Store
# ─────────────────────────────────────────────────────────────────────────────

class ChromaStore:
    """
    Persistent ChromaDB vector store.

    Parameters
    ----------
    persist_directory : Path to store the database on disk.
    collection_name   : Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_store",
        collection_name: str = "rag_docs",
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # ChromaDB stores raw vectors; we pass our own embeddings
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name
        print(
            f"[chroma] Collection '{collection_name}' ready. "
            f"Docs in store: {self._collection.count()}"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Insert chunks + their embeddings into ChromaDB.

        ChromaDB requires:
            ids        : unique string per document
            embeddings : list of float vectors
            documents  : the raw text
            metadatas  : dict per document (optional but very useful)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [f"{c.source}_chunk_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {**c.metadata, "source": c.source, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        # Upsert: insert or update if id already exists
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[chroma] Upserted {len(chunks)} chunks. Total: {self._collection.count()}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Find the top-k most similar chunks.

        Parameters
        ----------
        query_embedding : Embedding of the user's question.
        top_k           : How many results to return.
        where           : Metadata filter, e.g. {"source": "deep_learning.pdf"}

        Returns
        -------
        List of dicts: {text, source, chunk_index, metadata, distance}
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        hits = []
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "metadata": meta,
                "distance": dist,
                "score": 1 - dist,  # cosine: distance → similarity
            })
        return hits

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Drop the entire collection (useful for re-indexing)."""
        self._client.delete_collection(self._collection_name)
        print(f"[chroma] Deleted collection '{self._collection_name}'")

    def get_all_sources(self) -> list[str]:
        """Return a list of all unique source documents in the store."""
        result = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in result["metadatas"]}
        return sorted(sources)

    def get_all_chunks(self) -> list[dict]:
        """Return every stored chunk as {text, source, chunk_index} (used to rebuild BM25)."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {
                "text": doc,
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
            }
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Ingest sample chunks into ChromaDB and run similarity queries.
    Run: python 04_vector_db/chroma/chroma_store.py
    """
    import importlib.util, sys as _sys
    _r = str(Path(__file__).parent.parent.parent)
    if _r not in _sys.path:
        _sys.path.insert(0, _r)
    from rich.console import Console
    from rich.table import Table

    def _imp(mp, a):
        spec = importlib.util.spec_from_file_location(a, Path(_r) / mp)
        m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
        return getattr(m, a)

    LocalEmbedder = _imp("03_embeddings/embedder.py", "LocalEmbedder")
    RecursiveChunker = _imp("02_chunking/chunker.py", "RecursiveChunker")

    console = Console()
    console.rule("[bold blue]Module 4a — ChromaDB Vector Store")

    # Sample corpus (mini encyclopedia-style texts)
    corpus = {
        "intro_to_ai.txt": """
Artificial intelligence (AI) is intelligence demonstrated by machines.
AI research aims to build agents that perceive their environment and act to
achieve goals. Machine learning, a subset of AI, enables systems to learn
from data. Deep learning uses neural networks with many layers.
        """,
        "intro_to_nlp.txt": """
Natural Language Processing (NLP) is the branch of AI that focuses on
human language. Core NLP tasks include text classification, named entity
recognition, machine translation, and question answering. Transformers
revolutionized NLP with self-attention mechanisms. BERT and GPT are foundational
transformer models.
        """,
        "intro_to_rag.txt": """
Retrieval-Augmented Generation (RAG) improves language model outputs by
retrieving relevant documents before generating an answer. RAG has three stages:
indexing (process documents), retrieval (find relevant chunks), and generation
(produce answer given chunks). It reduces hallucination by grounding responses
in real documents.
        """,
    }

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=200, overlap=30)
    store = ChromaStore(
        persist_directory="./data/chroma_store",
        collection_name="demo_collection",
    )
    store.delete_collection()
    store = ChromaStore(
        persist_directory="./data/chroma_store",
        collection_name="demo_collection",
    )

    # Ingest
    all_chunks = []
    for source, text in corpus.items():
        chunks = chunker.split(text.strip(), source=source)
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts)
    store.add_chunks(all_chunks, embeddings)

    # Query
    queries = [
        "What is deep learning?",
        "How does RAG reduce hallucinations?",
        "What are transformer models?",
    ]

    for q in queries:
        console.print(f"\n[bold yellow]Query:[/bold yellow] {q}")
        q_emb = embedder.embed_one(q)
        results = store.query(q_emb, top_k=3)

        table = Table(show_header=True, show_lines=True)
        table.add_column("Score", justify="right", style="green")
        table.add_column("Source", style="cyan")
        table.add_column("Text")

        for r in results:
            table.add_row(
                f"{r['score']:.3f}",
                r["source"],
                r["text"][:80].replace("\n", " "),
            )
        console.print(table)

    console.print(f"\n[bold green]✓ ChromaDB demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
