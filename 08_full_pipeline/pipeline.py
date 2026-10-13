"""
Module 8 — Full End-to-End RAG Pipeline
=========================================
This is the main pipeline class that wires every module together.

Pipeline stages (in order):
    1. Load documents          → list[Document]
    2. Chunk documents         → list[Chunk]
    3. Embed chunks            → list[list[float]]
    4. Store in vector DB      → ChromaStore | RedisStore
    5. Retrieve for a query    → list[dict]
    6. Build prompt            → BuiltPrompt
    7. Generate answer         → str

Everything is configured via config.yaml — no code changes needed to swap
components. Swap components by editing the config, not the pipeline code.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config_loader import load_config  # noqa: E402
from src.models import Document, Chunk  # noqa: E402

# Lazy imports via importlib to avoid numeric-prefix issues
def _import(module_path: str, attr: str):
    spec = importlib.util.spec_from_file_location(
        attr, Path(_ROOT) / module_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """The output of a RAG query."""
    question: str
    answer: str
    sources: list[dict]
    latency_s: float
    model: str
    num_chunks_retrieved: int

    def __str__(self) -> str:
        lines = [
            f"Q: {self.question}",
            f"A: {self.answer}",
            f"",
            f"Sources ({self.num_chunks_retrieved} chunks retrieved):",
        ]
        for i, s in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {s.get('source', '?')} — {s.get('text', '')[:60]}...")
        lines.append(f"\nModel: {self.model} | Latency: {self.latency_s:.2f}s")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full Retrieval-Augmented Generation Pipeline.

    Usage:
        pipeline = RAGPipeline()          # loads config.yaml
        pipeline.index("data/raw/")       # index all PDFs in a folder
        response = pipeline.query("What is RAG?")
        print(response)

    Or with a single file:
        pipeline.index_file("data/raw/deep_learning.pdf")
    """

    def __init__(self, config_path: str | None = None):
        self._cfg = load_config(config_path)
        self._indexed_chunks: list[Chunk] = []
        self._store = None
        self._retriever = None
        self._setup_components()

    def _setup_components(self):
        cfg = self._cfg

        # Loader
        DoclingLoader = _import("01_document_loading/loader.py", "DoclingLoader")
        self._loader = DoclingLoader(
            ocr=cfg["extraction"].get("ocr", False),
            table_mode=cfg["extraction"].get("table_mode", "fast"),
        )

        # Chunker
        chunk_cfg = cfg["chunking"]
        get_chunker = _import("02_chunking/chunker.py", "get_chunker")
        self._chunker = get_chunker(
            strategy=chunk_cfg.get("strategy", "recursive"),
            chunk_size=chunk_cfg.get("chunk_size", 512),
            overlap=chunk_cfg.get("chunk_overlap", 64),
        )

        # Embedder
        get_embedder = _import("03_embeddings/embedder.py", "get_embedder")
        self._embedder = get_embedder(cfg["embeddings"])

        # Vector store
        self._store = self._build_store(cfg)

        # Prompt builder
        prompt_cfg = cfg["prompting"]
        PromptBuilder = _import("06_prompting/prompt_builder.py", "PromptBuilder")
        self._prompt_builder = PromptBuilder(
            system_prompt=prompt_cfg.get("system_prompt"),
            template=cfg.get("context_format", "numbered"),
            max_context_tokens=prompt_cfg.get("max_context_tokens", 3000),
        )

        # LLM
        get_llm = _import("07_llm_integration/llm_client.py", "get_llm")
        self._llm = get_llm(cfg["llm"])

        print(f"\n[pipeline] ✓ All components ready")
        print(f"  Embedder : {self._embedder.__class__.__name__}")
        print(f"  Store    : {self._store.__class__.__name__}")
        print(f"  LLM      : {self._llm.model_name}")

    def _build_store(self, cfg: dict):
        db_cfg = cfg["vector_db"]
        backend = db_cfg.get("backend", "chroma")

        if backend == "chroma":
            ChromaStore = _import("04_vector_db/chroma/chroma_store.py", "ChromaStore")
            c = db_cfg["chroma"]
            return ChromaStore(
                persist_directory=c.get("persist_directory", "./data/chroma_store"),
                collection_name=c.get("collection_name", "rag_docs"),
            )

        if backend == "redis":
            RedisStore = _import("04_vector_db/redis/redis_store.py", "RedisStore")
            r = db_cfg["redis"]
            return RedisStore(
                host=r.get("host", "localhost"),
                port=r.get("port", 6379),
                password=r.get("password", ""),
                index_name=r.get("index_name", "rag_docs"),
                vector_dim=self._embedder.dimension,
            )

        raise ValueError(f"Unknown vector_db backend: {backend!r}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_file(self, file_path: str | Path) -> int:
        """Load, chunk, embed, and store a single file. Returns chunk count."""
        doc = self._loader.load_file(file_path)
        return self._index_documents([doc])

    def index_directory(self, directory: str | Path, glob: str = "**/*.pdf") -> int:
        """Index all matching files in a directory. Returns total chunk count."""
        docs = self._loader.load_directory(directory, glob=glob)
        return self._index_documents(docs)

    def index_url(self, url: str) -> int:
        """Index a document from a URL."""
        doc = self._loader.load_url(url)
        return self._index_documents([doc])

    def _index_documents(self, docs: list[Document]) -> int:
        if not docs:
            print("[pipeline] No documents to index.")
            return 0

        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self._chunker.split(
                doc.content,
                source=doc.metadata.get("filename", doc.source),
                metadata=doc.metadata,
            )
            # Filter out very short chunks
            min_size = self._cfg["chunking"].get("min_chunk_size", 100)
            chunks = [c for c in chunks if len(c.text) >= min_size]
            all_chunks.extend(chunks)

        if not all_chunks:
            print("[pipeline] Chunking produced 0 valid chunks.")
            return 0

        print(f"[pipeline] Embedding {len(all_chunks)} chunks ...")
        texts = [c.text for c in all_chunks]
        embeddings = self._embedder.embed(texts)

        print(f"[pipeline] Storing in vector DB ...")
        self._store.add_chunks(all_chunks, embeddings)

        self._indexed_chunks.extend(all_chunks)
        print(f"[pipeline] ✓ Indexed {len(all_chunks)} chunks from {len(docs)} document(s)")
        return len(all_chunks)

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """
        Full RAG query: retrieve → prompt → generate.

        Parameters
        ----------
        question : The user's question in natural language
        top_k    : Override the config top_k value

        Returns
        -------
        RAGResponse with answer, sources, and metadata
        """
        t_start = time.perf_counter()

        # Retrieval
        k = top_k or self._cfg["retrieval"].get("top_k", 5)
        retriever = self._get_retriever()
        chunks = retriever.retrieve(question, top_k=k)

        if not chunks:
            return RAGResponse(
                question=question,
                answer="I could not find relevant information in the indexed documents.",
                sources=[],
                latency_s=time.perf_counter() - t_start,
                model=self._llm.model_name,
                num_chunks_retrieved=0,
            )

        # Prompting
        messages = self._prompt_builder.format_messages(question, chunks)

        # Generation
        answer = self._llm.generate(messages)

        latency = time.perf_counter() - t_start

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            latency_s=latency,
            model=self._llm.model_name,
            num_chunks_retrieved=len(chunks),
        )

    def stream_query(self, question: str, top_k: int | None = None) -> Iterator[str]:
        """
        Stream the answer token by token.
        Yields retrieved sources dict first, then text tokens.
        """
        k = top_k or self._cfg["retrieval"].get("top_k", 5)
        retriever = self._get_retriever()
        chunks = retriever.retrieve(question, top_k=k)

        if not chunks:
            yield "I could not find relevant information in the indexed documents."
            return

        messages = self._prompt_builder.format_messages(question, chunks)
        yield from self._llm.stream(messages)

    def _get_retriever(self):
        """Build the retriever configured in config.yaml."""
        get_retriever = _import("05_retrieval/retriever.py", "get_retriever")

        ret_cfg = self._cfg["retrieval"]
        strategy = ret_cfg.get("strategy", "similarity")

        corpus_dicts = None
        if strategy == "hybrid":
            if self._indexed_chunks:
                corpus_dicts = [
                    {"text": c.text, "source": c.source, "chunk_index": c.chunk_index}
                    for c in self._indexed_chunks
                ]
            elif hasattr(self._store, "get_all_chunks"):
                # Index was built in an earlier run: rebuild the BM25 corpus from the store
                corpus_dicts = self._store.get_all_chunks()
            if not corpus_dicts:
                print("[pipeline] Hybrid needs indexed text for BM25; falling back to 'similarity'.")
                strategy = "similarity"

        kwargs = {}
        if strategy == "mmr":
            kwargs["lambda_val"] = ret_cfg.get("mmr_lambda", 0.7)
        if strategy == "hybrid":
            kwargs["alpha"] = ret_cfg.get("hybrid", {}).get("alpha", 0.5)
        if strategy == "rerank":
            kwargs["rerank_model"] = ret_cfg.get("rerank", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        return get_retriever(
            strategy=strategy,
            store=self._store,
            embedder=self._embedder,
            corpus_chunks=corpus_dicts,
            **kwargs,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_indexed_sources(self) -> list[str]:
        """List all source documents currently indexed."""
        return self._store.get_all_sources() if hasattr(self._store, "get_all_sources") else []

    def reset_index(self) -> None:
        """Clear the entire vector store and reset indexed chunks."""
        if hasattr(self._store, "delete_collection"):
            self._store.delete_collection()
        elif hasattr(self._store, "flush"):
            self._store.flush()
        self._indexed_chunks = []
        # Re-create store
        self._store = self._build_store(self._cfg)
        print("[pipeline] Index reset.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Interactive CLI for the RAG pipeline.
    Run: python 08_full_pipeline/pipeline.py
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.rule("[bold blue]RAG From Scratch — Interactive CLI")

    pipeline = RAGPipeline()

    # Auto-index data/raw if it has PDFs
    raw_dir = Path("data/raw")
    if raw_dir.exists() and list(raw_dir.glob("*.pdf")):
        console.print(f"[cyan]Found PDFs in {raw_dir}. Indexing...[/cyan]")
        n = pipeline.index_directory(raw_dir)
        console.print(f"[green]✓ {n} chunks indexed[/green]")
    else:
        console.print(
            "[yellow]No PDFs found in data/raw/. "
            "Run: python data/download_samples.py[/yellow]"
        )
        return

    sources = pipeline.get_indexed_sources()
    console.print(f"\n[dim]Indexed documents: {sources}[/dim]\n")

    console.print("[bold]Type your question (or 'quit' to exit):[/bold]\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        console.print("\n[cyan]Retrieving and generating...[/cyan]")
        response = pipeline.query(question)

        console.print(Panel(
            response.answer,
            title=f"[green]Answer[/green]  [dim]({response.latency_s:.1f}s | {response.num_chunks_retrieved} chunks)[/dim]",
            border_style="green",
        ))

        console.print("[dim]Sources:[/dim]")
        for i, s in enumerate(response.sources, 1):
            console.print(f"  [{i}] {s.get('source', '?')} — {s.get('text', '')[:70]}...")
        console.print()


if __name__ == "__main__":
    main()
