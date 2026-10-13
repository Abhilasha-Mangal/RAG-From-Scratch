"""
Module 10 — Advanced RAG Techniques
=====================================
Once the basic pipeline works, these techniques push quality further.

Techniques covered:
    1. HyDE  — Hypothetical Document Embeddings
    2. Query Expansion — rewrite the query multiple ways
    3. Self-RAG — model decides when to retrieve
    4. GraphRAG concept — knowledge graph-based retrieval (overview only)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _imp(mp, a):
    spec = importlib.util.spec_from_file_location(a, Path(_ROOT) / mp)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return getattr(m, a)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HyDE — Hypothetical Document Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) — Gao et al., 2022.

    Insight: Embedding a *hypothetical answer* to the question often retrieves
    better results than embedding the question itself. Questions and answers
    live in different parts of the embedding space.

    Steps:
        1. Ask the LLM to generate a hypothetical answer (without context)
        2. Embed that hypothetical answer
        3. Use that embedding to query the vector store
        4. Use the *real* retrieved chunks to generate the final answer

    When it helps: complex questions, long-form answers, academic topics.
    """

    HYDE_PROMPT = (
        "Write a short, concise passage (2-3 sentences) that would directly answer "
        "the following question. Write it as if it were a factual paragraph from a "
        "textbook. Do NOT say 'I don't know' — generate a plausible answer.\n\n"
        "Question: {question}\n\nHypothetical passage:"
    )

    def __init__(self, store, embedder, llm):
        self._store = store
        self._embedder = embedder
        self._llm = llm

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        # Step 1: Generate hypothetical answer
        messages = [
            {"role": "user", "content": self.HYDE_PROMPT.format(question=question)}
        ]
        hypothetical_answer = self._llm.generate(messages)
        print(f"[HyDE] Hypothetical answer: {hypothetical_answer[:100]}...")

        # Step 2 & 3: Embed hypothetical answer and query
        hyp_emb = self._embedder.embed_one(hypothetical_answer)
        return self._store.query(hyp_emb, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Query Expansion
# ─────────────────────────────────────────────────────────────────────────────

class QueryExpansionRetriever:
    """
    Query Expansion — retrieve with multiple query variants, then fuse.

    Steps:
        1. Ask the LLM to rewrite the query N different ways
        2. Retrieve top-k for each variant
        3. Deduplicate and fuse with RRF

    Helps when the original query is ambiguous or uses different vocabulary
    than the indexed documents.
    """

    EXPANSION_PROMPT = (
        "Generate {n} different ways to ask the following question. "
        "Each rewrite should be semantically equivalent but use different words. "
        "Output ONLY the rewrites, one per line, no numbering.\n\n"
        "Original question: {question}"
    )

    def __init__(self, store, embedder, llm, n_expansions: int = 3):
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._n = n_expansions

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        # Step 1: Expand query
        messages = [{
            "role": "user",
            "content": self.EXPANSION_PROMPT.format(n=self._n, question=question)
        }]
        expansion_text = self._llm.generate(messages)
        variants = [question] + [q.strip() for q in expansion_text.strip().split("\n") if q.strip()]
        print(f"[QueryExp] Variants: {variants}")

        # Step 2: Retrieve for each variant
        all_results: dict[str, dict] = {}
        rank_lists: list[list[str]] = []

        for variant in variants:
            emb = self._embedder.embed_one(variant)
            results = self._store.query(emb, top_k=top_k)
            keys = []
            for r in results:
                key = r["text"][:60]
                all_results[key] = r
                keys.append(key)
            rank_lists.append(keys)

        # Step 3: RRF fusion
        k = 60
        rrf: dict[str, float] = {}
        for rank_list in rank_lists:
            for rank, key in enumerate(rank_list):
                rrf[key] = rrf.get(key, 0) + 1 / (k + rank + 1)

        fused = sorted(all_results.items(), key=lambda x: rrf.get(x[0], 0), reverse=True)
        return [v for _, v in fused[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Contextual Compression
# ─────────────────────────────────────────────────────────────────────────────

class ContextualCompressionRetriever:
    """
    Contextual Compression — extract only the relevant sentences from each chunk.

    Problem: Retrieved chunks often contain sentences unrelated to the query.
    Solution: Ask the LLM to extract only the relevant part of each chunk.

    This keeps the context window focused and improves answer quality.
    """

    COMPRESS_PROMPT = (
        "Given the following passage and a question, extract ONLY the sentences "
        "that are directly relevant to answering the question. "
        "If no sentences are relevant, output 'NOT RELEVANT'.\n\n"
        "Question: {question}\n\n"
        "Passage:\n{passage}\n\n"
        "Relevant sentences:"
    )

    def __init__(self, base_retriever, llm):
        self._base = base_retriever
        self._llm = llm

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        candidates = self._base.retrieve(question, top_k=top_k * 2)
        compressed = []

        for chunk in candidates:
            messages = [{
                "role": "user",
                "content": self.COMPRESS_PROMPT.format(
                    question=question,
                    passage=chunk["text"],
                )
            }]
            compressed_text = self._llm.generate(messages).strip()

            if compressed_text.upper() == "NOT RELEVANT":
                continue

            compressed_chunk = dict(chunk)
            compressed_chunk["text"] = compressed_text
            compressed_chunk["original_text"] = chunk["text"]
            compressed.append(compressed_chunk)

            if len(compressed) >= top_k:
                break

        return compressed


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.rule("[bold blue]Module 10 — Advanced RAG Techniques")

    try:
        RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")
        pipeline = RAGPipeline()
    except Exception as e:
        console.print(f"[red]Could not load pipeline: {e}[/red]")
        return

    if not pipeline.get_indexed_sources():
        raw_dir = Path("data/raw")
        if list(raw_dir.glob("*.pdf")):
            pipeline.index_directory(raw_dir)
        else:
            console.print("[yellow]Run: python data/download_samples.py first[/yellow]")
            return

    question = "How does the transformer architecture handle long-range dependencies?"
    console.print(f"\n[bold yellow]Question:[/bold yellow] {question}\n")

    # ── HyDE ──────────────────────────────────────────────────────────────────
    console.print("[bold cyan]1. HyDE Retrieval[/bold cyan]")
    try:
        hyde = HyDERetriever(pipeline._store, pipeline._embedder, pipeline._llm)
        hyde_results = hyde.retrieve(question, top_k=3)
        for i, r in enumerate(hyde_results, 1):
            console.print(Panel(
                r["text"][:200],
                title=f"[green]HyDE Result {i}[/green] — {r.get('source', '?')}",
                expand=False,
            ))
    except Exception as e:
        console.print(f"[red]HyDE failed: {e}[/red]")

    # ── Query Expansion ────────────────────────────────────────────────────────
    console.print("\n[bold cyan]2. Query Expansion[/bold cyan]")
    try:
        qe = QueryExpansionRetriever(pipeline._store, pipeline._embedder, pipeline._llm, n_expansions=3)
        qe_results = qe.retrieve(question, top_k=3)
        for i, r in enumerate(qe_results, 1):
            console.print(Panel(
                r["text"][:200],
                title=f"[green]QueryExp Result {i}[/green] — {r.get('source', '?')}",
                expand=False,
            ))
    except Exception as e:
        console.print(f"[red]Query Expansion failed: {e}[/red]")

    console.print("\n[bold]GraphRAG (concept overview):[/bold]")
    console.print("""
    GraphRAG extracts entities and relationships from documents as a knowledge graph.
    Queries traverse the graph to find multi-hop connections — e.g.:
        "transformer" → "self-attention" → "long-range dependencies"
    
    Libraries: Microsoft GraphRAG, LlamaIndex KnowledgeGraph, Neo4j + LangChain
    Best for: multi-hop reasoning, relationship queries, structured knowledge.
    """)

    console.print("[bold green]✓ Advanced techniques demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
