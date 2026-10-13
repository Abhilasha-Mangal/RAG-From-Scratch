"""
Module 9 — RAG Evaluation with RAGAS
======================================
Evaluating your RAG pipeline is as important as building it.

RAGAS (https://docs.ragas.io) is an open-source framework for evaluating RAG.

Metrics:
    • Faithfulness       — Is the answer grounded in the retrieved context?
                           (0=hallucinated, 1=fully faithful)
    • Answer Relevancy   — Is the answer relevant to the question?
    • Context Precision  — Are the retrieved chunks relevant?
    • Context Recall     — Did retrieval find all necessary information?

Workflow:
    1. Build a test set: (question, ground_truth_answer) pairs
    2. Run the RAG pipeline on each question
    3. Collect (question, answer, contexts, ground_truth) tuples
    4. Score with RAGAS
    5. Analyse weak spots

Run: python 09_evaluation/evaluator.py
"""

from __future__ import annotations

import importlib.util
import json
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
# Built-in test set (covers the sample Wikipedia data)
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_TEST_SET = [
    {
        "question": "What is the difference between machine learning and deep learning?",
        "ground_truth": (
            "Machine learning is a subset of AI that enables systems to learn from data. "
            "Deep learning is a subset of machine learning that uses neural networks with "
            "many layers. Deep learning excels at unstructured data like images and text."
        ),
    },
    {
        "question": "What are the three main stages of a RAG pipeline?",
        "ground_truth": (
            "The three stages of RAG are: (1) indexing — convert documents to embeddings, "
            "(2) retrieval — find relevant chunks using vector search, "
            "(3) generation — produce an answer using the LLM with retrieved chunks as context."
        ),
    },
    {
        "question": "What is the transformer architecture and how does it work?",
        "ground_truth": (
            "The transformer architecture uses self-attention mechanisms to process sequences "
            "in parallel. It consists of encoder and decoder stacks with multi-head attention "
            "and feed-forward layers. BERT uses the encoder, GPT uses the decoder."
        ),
    },
    {
        "question": "What is a vector database and why is it used in RAG?",
        "ground_truth": (
            "A vector database stores high-dimensional vectors (embeddings) and supports "
            "efficient approximate nearest-neighbour search. In RAG, it stores document "
            "embeddings and retrieves the most semantically similar chunks to a user query."
        ),
    },
    {
        "question": "How does tokenization work in NLP?",
        "ground_truth": (
            "Tokenization splits text into smaller units called tokens. Word tokenization splits "
            "on whitespace. Subword tokenization (BPE, WordPiece) handles unknown words by "
            "breaking them into known subword units. Modern LLMs use subword tokenization."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluate a RAG pipeline using RAGAS.

    Parameters
    ----------
    pipeline    : A RAGPipeline instance (from Module 8)
    llm_for_eval: An LLM to use for RAGAS metric computation
                  (can be a different/cheaper model than the pipeline LLM)
    """

    def __init__(self, pipeline, eval_llm=None):
        self._pipeline = pipeline
        self._eval_llm = eval_llm

    def build_dataset(self, test_cases: list[dict]) -> dict:
        """
        Run each test question through the pipeline and collect:
            question, answer, contexts, ground_truth

        Returns a dict ready for RAGAS evaluation.
        """
        questions, answers, contexts, ground_truths = [], [], [], []

        print(f"\n[eval] Running {len(test_cases)} test questions through pipeline...")

        for i, tc in enumerate(test_cases, 1):
            q = tc["question"]
            gt = tc["ground_truth"]
            print(f"  [{i}/{len(test_cases)}] {q[:70]}...")

            response = self._pipeline.query(q, top_k=5)

            questions.append(q)
            answers.append(response.answer)
            contexts.append([s.get("text", "") for s in response.sources])
            ground_truths.append(gt)

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

    def evaluate(self, test_cases: list[dict] | None = None) -> dict:
        """
        Run full RAGAS evaluation.

        Parameters
        ----------
        test_cases : List of {question, ground_truth} dicts.
                     If None, uses the built-in test set.

        Returns
        -------
        Dict of metric_name → score
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "RAGAS not installed. Run: pip install ragas datasets"
            )

        test_cases = test_cases or BUILTIN_TEST_SET
        data = self.build_dataset(test_cases)
        dataset = Dataset.from_dict(data)

        print("\n[eval] Computing RAGAS scores (this may take a few minutes)...")
        print("[eval] Note: RAGAS uses an LLM internally for some metrics.")
        print("[eval] If no LLM is set, it will use gpt-3.5-turbo by default.")

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        result = evaluate(dataset=dataset, metrics=metrics)
        return dict(result)

    def evaluate_manually(self, test_cases: list[dict] | None = None) -> dict:
        """
        Manual evaluation without RAGAS (no LLM needed).
        Uses simple heuristics: keyword overlap, answer length, source count.

        Useful for quick checks without API keys.
        """
        test_cases = test_cases or BUILTIN_TEST_SET
        data = self.build_dataset(test_cases)

        scores = {
            "avg_retrieved_chunks": 0,
            "avg_answer_length_tokens": 0,
            "context_coverage_score": 0,
            "no_answer_rate": 0,
        }

        no_answer_phrases = [
            "i don't have", "i cannot", "not enough information",
            "i don't know", "not in the context"
        ]
        total = len(data["answer"])
        no_answers = 0
        total_chunks = 0
        total_ans_len = 0
        total_coverage = 0

        for ans, ctxs, gt in zip(data["answer"], data["contexts"], data["ground_truth"]):
            if any(p in ans.lower() for p in no_answer_phrases):
                no_answers += 1

            total_chunks += len(ctxs)
            total_ans_len += len(ans.split())

            # Simple keyword coverage: how many GT words appear in context
            gt_words = set(gt.lower().split())
            ctx_words = set(" ".join(ctxs).lower().split())
            coverage = len(gt_words & ctx_words) / max(len(gt_words), 1)
            total_coverage += coverage

        scores["avg_retrieved_chunks"] = round(total_chunks / total, 2)
        scores["avg_answer_length_tokens"] = round(total_ans_len / total, 1)
        scores["context_coverage_score"] = round(total_coverage / total, 3)
        scores["no_answer_rate"] = round(no_answers / total, 3)

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold blue]Module 9 — RAG Evaluation with RAGAS")

    # Try to load the pipeline
    try:
        RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")
        pipeline = RAGPipeline()
    except Exception as e:
        console.print(f"[red]Could not load pipeline: {e}[/red]")
        return

    # Check if indexed
    if not pipeline.get_indexed_sources():
        raw_dir = Path("data/raw")
        pdfs = list(raw_dir.glob("*.pdf"))
        if pdfs:
            console.print("[cyan]Auto-indexing data/raw/...[/cyan]")
            pipeline.index_directory(raw_dir)
        else:
            console.print(
                "[yellow]No indexed documents. Run: python data/download_samples.py[/yellow]"
            )
            return

    evaluator = RAGEvaluator(pipeline)

    # Try RAGAS first, fall back to manual
    try:
        import ragas
        console.print("[cyan]Running RAGAS evaluation...[/cyan]")
        scores = evaluator.evaluate()
        console.print("\n[bold green]RAGAS Scores:[/bold green]")
        table = Table(show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Interpretation")
        interp = {
            "faithfulness": "1.0 = answer fully grounded in context",
            "answer_relevancy": "1.0 = answer perfectly answers question",
            "context_precision": "1.0 = all retrieved chunks are relevant",
            "context_recall": "1.0 = all necessary info was retrieved",
        }
        for metric, score in scores.items():
            table.add_row(metric, f"{score:.3f}", interp.get(metric, ""))
        console.print(table)

    except ImportError:
        console.print("[yellow]RAGAS not installed. Running manual evaluation...[/yellow]")
        scores = evaluator.evaluate_manually()

        table = Table(title="Manual Evaluation Scores", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        for k, v in scores.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

    # Save results
    out = Path("data/eval/eval_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(scores, f, indent=2)
    console.print(f"\n[green]Results saved → {out}[/green]")
    console.print("[bold green]✓ Evaluation complete.[/bold green]")


if __name__ == "__main__":
    demo()
