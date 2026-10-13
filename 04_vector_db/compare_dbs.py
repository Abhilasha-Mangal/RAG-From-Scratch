"""
Module 4 — Vector DB Comparison
=================================
Compare ChromaDB vs Redis side-by-side on the same dataset.
Run this to understand the trade-offs.

Metrics compared:
    • Ingest speed (chunks/sec)
    • Query speed (ms/query)
    • Result quality (rank comparison)
    • Setup complexity
"""

from __future__ import annotations
import importlib.util
import time
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rich.console import Console
from rich.table import Table


def _imp(mp, a):
    spec = importlib.util.spec_from_file_location(a, Path(_ROOT) / mp)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return getattr(m, a)


LocalEmbedder = _imp("03_embeddings/embedder.py", "LocalEmbedder")
RecursiveChunker = _imp("02_chunking/chunker.py", "RecursiveChunker")


def run_comparison():
    console = Console()
    console.rule("[bold blue]Module 4 — ChromaDB vs Redis Comparison")

    # ── Build corpus ─────────────────────────────────────────────────────────
    corpus_texts = {
        "machine_learning.txt": """
Machine learning is a method of data analysis that automates analytical model
building. It is based on the idea that systems can learn from data, identify
patterns and make decisions with minimal human intervention. Supervised learning
uses labeled training data. Unsupervised learning finds patterns in unlabeled data.
Reinforcement learning trains agents through rewards and penalties.
        """ * 10,
        "nlp_fundamentals.txt": """
Natural language processing combines computational linguistics with statistical
and machine learning models to enable computers to process human language.
Tokenization breaks text into words or subwords. Part-of-speech tagging assigns
grammatical roles. Named entity recognition identifies proper nouns. Sentiment
analysis determines the emotional tone of text.
        """ * 10,
        "deep_learning.txt": """
Deep learning is a subset of machine learning that uses neural networks with many
layers. Convolutional neural networks excel at image recognition. Recurrent neural
networks handle sequential data. The transformer architecture uses attention
mechanisms and has become dominant in NLP. Batch normalization and dropout help
prevent overfitting in deep networks.
        """ * 10,
    }

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=300, overlap=50)

    all_chunks = []
    for source, text in corpus_texts.items():
        chunks = chunker.split(text.strip(), source=source)
        all_chunks.extend(chunks)

    console.print(f"[cyan]Total chunks to index: {len(all_chunks)}[/cyan]")
    texts = [c.text for c in all_chunks]

    console.print("[cyan]Embedding all chunks (shared for fair comparison)...[/cyan]")
    embeddings = embedder.embed(texts)
    console.print(f"[green]Embeddings ready: {len(embeddings)} × {len(embeddings[0])}d[/green]")

    queries = [
        "What is supervised learning?",
        "How do transformers work in NLP?",
        "What prevents overfitting in neural networks?",
        "Explain tokenization in NLP.",
        "What is reinforcement learning?",
    ]
    query_embeddings = [embedder.embed_one(q) for q in queries]

    results_summary = []

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    try:
        ChromaStore = _imp("04_vector_db/chroma/chroma_store.py", "ChromaStore")

        console.print("\n[bold cyan]Testing ChromaDB...[/bold cyan]")
        store_chroma = ChromaStore(
            persist_directory="./data/chroma_bench",
            collection_name="benchmark",
        )
        store_chroma.delete_collection()
        store_chroma = ChromaStore(
            persist_directory="./data/chroma_bench",
            collection_name="benchmark",
        )

        t0 = time.perf_counter()
        store_chroma.add_chunks(all_chunks, embeddings)
        ingest_time = time.perf_counter() - t0

        query_times = []
        for qe in query_embeddings:
            t0 = time.perf_counter()
            store_chroma.query(qe, top_k=5)
            query_times.append((time.perf_counter() - t0) * 1000)

        results_summary.append({
            "db": "ChromaDB",
            "setup": "Zero (pip only)",
            "ingest_s": f"{ingest_time:.2f}s",
            "query_ms": f"{sum(query_times)/len(query_times):.1f}ms",
            "persistence": "Local files",
            "server": "No",
            "docker": "No",
        })
        console.print("[green]  ChromaDB ✓[/green]")
    except Exception as e:
        console.print(f"[red]  ChromaDB failed: {e}[/red]")

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        RedisStore = _imp("04_vector_db/redis/redis_store.py", "RedisStore")

        console.print("[bold cyan]Testing Redis...[/bold cyan]")
        store_redis = RedisStore(
            host="localhost", port=6379,
            index_name="benchmark",
            vector_dim=embedder.dimension,
        )
        store_redis.flush()

        t0 = time.perf_counter()
        store_redis.add_chunks(all_chunks, embeddings)
        ingest_time = time.perf_counter() - t0

        query_times = []
        for qe in query_embeddings:
            t0 = time.perf_counter()
            store_redis.query(qe, top_k=5)
            query_times.append((time.perf_counter() - t0) * 1000)

        results_summary.append({
            "db": "Redis",
            "setup": "Docker required",
            "ingest_s": f"{ingest_time:.2f}s",
            "query_ms": f"{sum(query_times)/len(query_times):.1f}ms",
            "persistence": "In-memory + RDB",
            "server": "Yes",
            "docker": "Yes",
        })
        console.print("[green]  Redis ✓[/green]")
    except Exception as e:
        console.print(f"[yellow]  Redis skipped (not running): {e}[/yellow]")

    # ── Summary table ─────────────────────────────────────────────────────────
    if results_summary:
        table = Table(title="Vector DB Comparison", show_lines=True)
        for col in ["db", "setup", "ingest_s", "query_ms", "persistence", "server", "docker"]:
            table.add_column(col.replace("_", " ").title(), style="cyan" if col == "db" else "")

        for row in results_summary:
            table.add_row(*[str(row[k]) for k in row])
        console.print(table)

    console.print("\n[bold]Other Vector DBs (info only):[/bold]")
    other_dbs = [
        ("FAISS",    "facebook/faiss",       "pip only",      "In-memory numpy", "Fast CPU search, no persistence by default"),
        ("Qdrant",   "qdrant/qdrant",         "Docker",        "Disk + WAL",      "Rich filtering, gRPC + REST, production-ready"),
        ("Weaviate", "semitechnologies/weaviate", "Docker",    "Object store",    "Built-in BM25 hybrid, GraphQL API"),
        ("Milvus",   "milvusdb/milvus",       "Docker Compose","Disk + S3",       "Billion-scale, Kubernetes-native"),
        ("pgvector", "ankane/pgvector",        "Docker (Postgres)", "Postgres WAL","SQL queries + vector search together"),
    ]
    t2 = Table(show_lines=True)
    t2.add_column("DB", style="cyan")
    t2.add_column("Image")
    t2.add_column("Setup")
    t2.add_column("Storage")
    t2.add_column("Best for")
    for row in other_dbs:
        t2.add_row(*row)
    console.print(t2)

    console.print("\n[bold green]✓ Comparison complete.[/bold green]")


if __name__ == "__main__":
    run_comparison()
