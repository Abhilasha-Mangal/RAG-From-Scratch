"""
Module 4b — Redis Vector Store (RedisVL)
==========================================
Redis is an in-memory data store with a vector search extension.
RedisVL (Redis Vector Library) makes it easy to create indexes and run
similarity searches.

Prerequisites:
    Start Redis with:  docker-compose up -d redis
    Or install locally: https://redis.io/docs/getting-started/installation/

Why Redis for RAG?
    • Ultra-fast (in-memory)
    • Supports hybrid search: vector similarity + full-text (RediSearch)
    • Metadata filtering with rich query syntax
    • Used in production at scale

RedisVL docs: https://www.redisvl.com
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import Chunk  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Redis Store
# ─────────────────────────────────────────────────────────────────────────────

class RedisStore:
    """
    Vector store backed by Redis + RedisVL.

    Parameters
    ----------
    host        : Redis host (default: localhost)
    port        : Redis port (default: 6379)
    password    : Redis password (optional)
    index_name  : Name of the RedisVL index (like a table)
    vector_dim  : Embedding vector dimension (must match your embedder)
    """

    SCHEMA = {
        "index": {
            "name": None,          # filled at runtime
            "prefix": "chunk",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "text",        "type": "text"},
            {"name": "source",      "type": "tag"},
            {"name": "chunk_index", "type": "numeric"},
            {"name": "strategy",    "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": None,          # filled at runtime
                    "distance_metric": "COSINE",
                    "algorithm": "HNSW",
                    "datatype": "FLOAT32",
                },
            },
        ],
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        index_name: str = "rag_docs",
        vector_dim: int = 384,
    ):
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        self._host = host
        self._port = port
        self._password = password or os.getenv("REDIS_PASSWORD", "")
        self._index_name = index_name
        self._vector_dim = vector_dim

        # Build schema
        schema_dict = dict(self.SCHEMA)
        schema_dict["index"] = {**schema_dict["index"], "name": index_name}
        schema_dict["fields"] = [
            {**f, "attrs": {**f.get("attrs", {}), "dims": vector_dim}}
            if f["type"] == "vector" else f
            for f in schema_dict["fields"]
        ]

        self._schema = IndexSchema.from_dict(schema_dict)
        self._index = SearchIndex(
            schema=self._schema,
            redis_url=self._redis_url(),
        )
        self._index.create(overwrite=False, drop=False)
        print(f"[redis] Index '{index_name}' ready on {host}:{port}")

    def _redis_url(self) -> str:
        if self._password:
            return f"redis://:{self._password}@{self._host}:{self._port}"
        return f"redis://{self._host}:{self._port}"

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks, embeddings: list[list[float]]) -> None:
        """Insert chunks + embeddings into Redis."""
        import numpy as np

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        data = []
        for chunk, emb in zip(chunks, embeddings):
            vec_bytes = np.array(emb, dtype="float32").tobytes()
            data.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "strategy": chunk.metadata.get("strategy", ""),
                "embedding": vec_bytes,
            })

        self._index.load(data)
        print(f"[redis] Loaded {len(data)} chunks into '{self._index_name}'")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform HNSW approximate nearest-neighbour search.

        Parameters
        ----------
        query_embedding : Float vector of the query.
        top_k           : Number of results.
        source_filter   : Optional tag filter, e.g. "deep_learning.pdf"

        Returns
        -------
        List of dicts: {text, source, chunk_index, score}
        """
        import numpy as np
        from redisvl.query import VectorQuery

        vec_bytes = np.array(query_embedding, dtype="float32").tobytes()

        vq = VectorQuery(
            vector=vec_bytes,
            vector_field_name="embedding",
            return_fields=["text", "source", "chunk_index", "strategy"],
            num_results=top_k,
        )

        if source_filter:
            vq = vq.filter(f"@source:{{{source_filter}}}")

        results = self._index.query(vq)

        hits = []
        for r in results:
            hits.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": int(r.get("chunk_index", -1)),
                "metadata": {"strategy": r.get("strategy", "")},
                "score": float(r.get("vector_distance", 1.0)),
            })
        return hits

    def count(self) -> int:
        """Return approximate count of indexed documents."""
        try:
            info = self._index.info()
            return int(info.get("num_docs", 0))
        except Exception:
            return -1

    def flush(self) -> None:
        """Drop and recreate the index (useful for re-indexing)."""
        self._index.delete(drop=True)
        self._index.create(overwrite=True)
        print(f"[redis] Index '{self._index_name}' flushed.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Ingest sample chunks into Redis and run similarity queries.
    Requires Redis running: docker-compose up -d redis

    Run: python 04_vector_db/redis/redis_store.py
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
    console.rule("[bold blue]Module 4b — Redis Vector Store")

    corpus = {
        "neural_networks.txt": """
Neural networks are computing systems inspired by the biological neural
networks in animal brains. They consist of layers of interconnected nodes.
Deep neural networks have many hidden layers and can learn complex patterns.
Backpropagation is used to train neural networks by adjusting weights.
        """,
        "transformers.txt": """
The Transformer architecture uses self-attention to process sequences in parallel.
BERT (Bidirectional Encoder Representations from Transformers) learns bidirectional
context. GPT models use decoder-only transformers for text generation.
Attention mechanisms allow the model to focus on relevant parts of the input.
        """,
    }

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=200, overlap=30)

    try:
        store = RedisStore(
            host="localhost",
            port=6379,
            index_name="rag_demo",
            vector_dim=embedder.dimension,
        )
    except Exception as e:
        console.print(f"[red]Redis connection failed: {e}[/red]")
        console.print("[yellow]Start Redis with: docker-compose up -d redis[/yellow]")
        return

    store.flush()

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
        "How are neural networks trained?",
        "What is the attention mechanism?",
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

    console.print("\n[bold green]✓ Redis demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
