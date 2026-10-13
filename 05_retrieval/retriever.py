"""
Module 5 — Retrieval Strategies
=================================
Retrieval = given a user question, find the most relevant chunks from the DB.

Strategies:
    1. SimilarityRetriever  — plain top-k cosine similarity
    2. MMRRetriever         — Maximal Marginal Relevance (diversity-aware)
    3. HybridRetriever      — BM25 + semantic + RRF fusion
    4. ReRankRetriever      — top-k → re-score with CrossEncoder

All retrievers share the same interface:
    retrieve(query: str, top_k: int) → list[dict]
"""

from __future__ import annotations

import importlib.util
import math
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import Chunk  # noqa: E402


def _import(module_path: str, attr: str):
    spec = importlib.util.spec_from_file_location(attr, Path(_ROOT) / module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """Common interface: all retrievers expose .retrieve(query, top_k)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks relevant to query."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. Similarity Retriever (plain top-k)
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityRetriever(BaseRetriever):
    """
    Retrieve the top-k chunks by cosine similarity.
    Simple, fast, and works well for most cases.
    """

    def __init__(self, store, embedder):
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self._embedder.embed_one(query)
        return self._store.query(q_emb, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MMR Retriever
# ─────────────────────────────────────────────────────────────────────────────

class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance (MMR) Retrieval.

    Fetch fetch_k candidates, then greedily select top_k that balance:
        • Relevance to the query (high similarity to query)
        • Diversity from already-selected chunks (low similarity to selected)

    lambda_val = 1.0 → pure relevance (same as top-k)
    lambda_val = 0.0 → pure diversity
    lambda_val = 0.7 → recommended (more relevance, some diversity)

    Why it matters: Without MMR, top-k results often repeat the same idea.
    """

    def __init__(self, store, embedder, lambda_val: float = 0.7, fetch_k: int = 20):
        self._store = store
        self._embedder = embedder
        self._lambda = lambda_val
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = np.array(self._embedder.embed_one(query))
        candidates = self._store.query(q_emb.tolist(), top_k=self._fetch_k)

        if not candidates:
            return []

        # Convert candidate texts back to embeddings for diversity calculation
        cand_texts = [c["text"] for c in candidates]
        cand_embs = np.array(self._embedder.embed(cand_texts))

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining_indices:
                break

            # Score = λ * relevance - (1-λ) * max_similarity_to_selected
            best_idx = None
            best_score = -float("inf")

            for idx in remaining_indices:
                relevance = float(q_emb @ cand_embs[idx])

                if not selected_indices:
                    diversity_penalty = 0.0
                else:
                    sims_to_selected = [
                        float(cand_embs[idx] @ cand_embs[sel])
                        for sel in selected_indices
                    ]
                    diversity_penalty = max(sims_to_selected)

                score = self._lambda * relevance - (1 - self._lambda) * diversity_penalty

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Retriever (BM25 + Semantic, RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Combine keyword search (BM25) with semantic (vector) search.

    Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists:
        RRF(d) = Σ  1 / (k + rank_i(d))
    where k=60 is a smoothing constant.

    Why hybrid? Some queries are better served by keywords (exact terms,
    product names) while others need semantics (paraphrases, synonyms).
    Hybrid search wins in benchmarks for most RAG use cases.

    alpha = 1.0 → pure semantic
    alpha = 0.0 → pure BM25
    alpha = 0.5 → equal weight (default)
    """

    def __init__(self, store, embedder, corpus_chunks: list, alpha: float = 0.5):
        from rank_bm25 import BM25Okapi

        self._store = store
        self._embedder = embedder
        self._alpha = alpha
        self._chunks = corpus_chunks

        # Build BM25 index from chunk texts
        tokenized = [c["text"].lower().split() for c in corpus_chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        fetch_k = top_k * 4  # fetch more, then fuse

        # ── Semantic results ──────────────────────────────────────────────────
        q_emb = self._embedder.embed_one(query)
        semantic_results = self._store.query(q_emb, top_k=fetch_k)

        # ── BM25 results ──────────────────────────────────────────────────────
        query_tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # ── RRF Fusion ────────────────────────────────────────────────────────
        k = 60  # RRF constant
        rrf_scores: dict[str, float] = {}

        # Score semantic results
        for rank, hit in enumerate(semantic_results):
            key = hit["text"][:50]
            rrf_scores[key] = rrf_scores.get(key, 0) + self._alpha / (k + rank + 1)

        # Score BM25 results
        for rank, idx in enumerate(bm25_top_indices):
            key = self._chunks[idx]["text"][:50]
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 - self._alpha) / (k + rank + 1)

        # Merge: attach scores back to semantic results
        all_hits = {h["text"][:50]: h for h in semantic_results}
        for idx in bm25_top_indices:
            c = self._chunks[idx]
            key = c["text"][:50]
            if key not in all_hits:
                all_hits[key] = c

        # Sort by fused score
        scored = sorted(all_hits.items(), key=lambda x: rrf_scores.get(x[0], 0), reverse=True)
        final = []
        for key, hit in scored[:top_k]:
            hit["rrf_score"] = rrf_scores.get(key, 0)
            final.append(hit)

        return final


# ─────────────────────────────────────────────────────────────────────────────
# 4. Re-Rank Retriever
# ─────────────────────────────────────────────────────────────────────────────

class ReRankRetriever(BaseRetriever):
    """
    Two-stage retrieval:
        Stage 1: Fast top-k vector search (get candidates)
        Stage 2: Slow but accurate CrossEncoder re-scores all candidates

    CrossEncoder reads the (query, passage) pair jointly — much more accurate
    than bi-encoder (separate query/doc embeddings) but too slow for full corpus.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (free, ~65MB)
    """

    def __init__(
        self,
        store,
        embedder,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 20,
    ):
        from sentence_transformers import CrossEncoder
        self._store = store
        self._embedder = embedder
        self._cross_encoder = CrossEncoder(rerank_model)
        self._fetch_k = fetch_k
        print(f"[rerank] CrossEncoder loaded: {rerank_model}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Stage 1: fetch_k candidates via vector search
        q_emb = self._embedder.embed_one(query)
        candidates = self._store.query(q_emb, top_k=self._fetch_k)

        if not candidates:
            return []

        # Stage 2: re-score with CrossEncoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        for hit, score in zip(candidates, scores):
            hit["rerank_score"] = float(score)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(strategy: str, store, embedder, corpus_chunks: list | None = None, **kwargs) -> BaseRetriever:
    """Build a retriever from strategy name."""
    if strategy == "similarity":
        return SimilarityRetriever(store, embedder)
    if strategy == "mmr":
        return MMRRetriever(store, embedder, **kwargs)
    if strategy == "hybrid":
        if not corpus_chunks:
            raise ValueError("HybridRetriever requires corpus_chunks for BM25 index")
        return HybridRetriever(store, embedder, corpus_chunks, **kwargs)
    if strategy == "rerank":
        return ReRankRetriever(store, embedder, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Compare all retrieval strategies on the same query.
    Run: python 05_retrieval/retriever.py
    """
    from rich.console import Console
    from rich.table import Table

    LocalEmbedder = _import("03_embeddings/embedder.py", "LocalEmbedder")
    RecursiveChunker = _import("02_chunking/chunker.py", "RecursiveChunker")
    ChromaStore = _import("04_vector_db/chroma/chroma_store.py", "ChromaStore")

    console = Console()
    console.rule("[bold blue]Module 5 — Retrieval Strategies")

    corpus_texts = {
        "ai_overview.txt": """
Artificial intelligence encompasses machine learning, deep learning, and symbolic AI.
Neural networks learn representations from data. Transformers use attention mechanisms.
Reinforcement learning trains agents with rewards. Natural language processing enables
computers to understand human text. Computer vision processes images and videos.
        """ * 6,
        "rag_guide.txt": """
Retrieval-Augmented Generation retrieves relevant documents before generating answers.
The retrieval step uses vector similarity search. Chunking splits documents into pieces.
Embeddings convert text to vectors. The generation step uses a language model.
Evaluation measures faithfulness, relevance, and groundedness of answers.
        """ * 6,
    }

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=200, overlap=30)
    store = ChromaStore("./data/chroma_retrieval_demo", "retrieval_demo")
    store.delete_collection()
    store = ChromaStore("./data/chroma_retrieval_demo", "retrieval_demo")

    all_chunks = []
    for source, text in corpus_texts.items():
        chunks = chunker.split(text.strip(), source=source)
        all_chunks.extend(chunks)

    store.add_chunks(all_chunks, embedder.embed([c.text for c in all_chunks]))

    corpus_dicts = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in all_chunks]

    query = "How does retrieval work in RAG?"
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    strategies = {
        "Similarity (top-k)": SimilarityRetriever(store, embedder),
        "MMR (λ=0.7)": MMRRetriever(store, embedder, lambda_val=0.7),
        "Hybrid (α=0.5)": HybridRetriever(store, embedder, corpus_dicts, alpha=0.5),
    }

    for name, retriever in strategies.items():
        results = retriever.retrieve(query, top_k=3)
        console.print(f"[bold cyan]{name}[/bold cyan]")
        t = Table(show_lines=True)
        t.add_column("#")
        t.add_column("Source", style="cyan")
        t.add_column("Text")
        for i, r in enumerate(results):
            t.add_row(str(i + 1), r["source"], r["text"][:80].replace("\n", " "))
        console.print(t)
        console.print()

    console.print("[bold green]✓ Retrieval demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
