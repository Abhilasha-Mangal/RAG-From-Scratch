"""
Module 6 — Prompt Engineering for RAG
=======================================
The prompt is the bridge between retrieved chunks and the LLM.
A well-designed prompt:
    • Tells the LLM to ONLY use the provided context
    • Formats context clearly (numbered, XML, or plain)
    • Asks the model to cite sources
    • Handles the "I don't know" case gracefully

This module builds a PromptBuilder class and shows different formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jinja2 import Template


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates (Jinja2)
# ─────────────────────────────────────────────────────────────────────────────

# ── Numbered context format ───────────────────────────────────────────────────
NUMBERED_TEMPLATE = Template("""
You are a helpful and accurate assistant. Answer the user's question using ONLY
the context passages provided below. Do not use any prior knowledge.

If the answer cannot be found in the context, respond with:
"I don't have enough information in the provided documents to answer that."

Always cite the source by referring to the passage number, e.g. [1], [2].

CONTEXT:
{% for i, chunk in chunks %}
[{{ i }}] (Source: {{ chunk.source | default('unknown') }})
{{ chunk.text }}
{% endfor %}

QUESTION: {{ question }}

ANSWER:
""".strip())


# ── XML context format (works well with Claude-style prompting) ──────────────
XML_TEMPLATE = Template("""
<instructions>
Answer the question below based solely on the provided documents.
Cite which document your answer comes from.
If unsure, say "I don't know based on the available documents."
</instructions>

<documents>
{% for i, chunk in chunks %}
<document index="{{ i }}" source="{{ chunk.source | default('unknown') }}">
{{ chunk.text }}
</document>
{% endfor %}
</documents>

<question>{{ question }}</question>

<answer>
""".strip())


# ── Chain-of-thought template ─────────────────────────────────────────────────
COT_TEMPLATE = Template("""
You are a careful reasoning assistant. Use only the passages below to answer.

CONTEXT PASSAGES:
{% for i, chunk in chunks %}
--- Passage {{ i }} (from {{ chunk.source | default('unknown') }}) ---
{{ chunk.text }}
{% endfor %}

QUESTION: {{ question }}

Think step-by-step before answering:
1. Which passages are most relevant to this question?
2. What key facts do those passages contain?
3. How do those facts answer the question?

REASONING:
[Think through it here]

FINAL ANSWER:
""".strip())


# ── Concise Q&A template (for chatbots) ───────────────────────────────────────
CONCISE_TEMPLATE = Template("""
Context:
{% for i, chunk in chunks %}{{ chunk.text }} {% endfor %}

Based on the context above, answer concisely: {{ question }}
""".strip())


TEMPLATES = {
    "numbered": NUMBERED_TEMPLATE,
    "xml":      XML_TEMPLATE,
    "cot":      COT_TEMPLATE,
    "concise":  CONCISE_TEMPLATE,
}


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuiltPrompt:
    """The final prompt ready to send to an LLM."""
    system: str
    user: str
    template_name: str
    num_chunks: int
    estimated_tokens: int  # rough estimate


class PromptBuilder:
    """
    Build RAG prompts from retrieved chunks + user question.

    Parameters
    ----------
    system_prompt   : The LLM system instruction (role definition)
    template        : "numbered" | "xml" | "cot" | "concise"
    max_context_tokens : Trim context if it exceeds this many tokens
    """

    DEFAULT_SYSTEM = (
        "You are a helpful, accurate assistant that answers questions based "
        "strictly on the provided context. Never make up information."
    )

    def __init__(
        self,
        system_prompt: str | None = None,
        template: Literal["numbered", "xml", "cot", "concise"] = "numbered",
        max_context_tokens: int = 3000,
    ):
        self._system = system_prompt or self.DEFAULT_SYSTEM
        self._template = TEMPLATES[template]
        self._template_name = template
        self._max_context_tokens = max_context_tokens

    def build(self, question: str, chunks: list[dict]) -> BuiltPrompt:
        """
        Build the final prompt from a question and retrieved chunks.

        Parameters
        ----------
        question : The user's question
        chunks   : List of dicts from the retriever (must have 'text' key)

        Returns
        -------
        BuiltPrompt with system and user strings ready for the LLM
        """
        # Trim context to token budget (rough: 1 token ≈ 4 chars)
        trimmed = self._trim_chunks(chunks, self._max_context_tokens)

        # Render the template
        user_prompt = self._template.render(
            question=question,
            chunks=list(enumerate(trimmed, start=1)),
        )

        estimated_tokens = len(user_prompt) // 4

        return BuiltPrompt(
            system=self._system,
            user=user_prompt,
            template_name=self._template_name,
            num_chunks=len(trimmed),
            estimated_tokens=estimated_tokens,
        )

    def _trim_chunks(self, chunks: list[dict], max_tokens: int) -> list[dict]:
        """Remove chunks from the end until we're within the token budget."""
        max_chars = max_tokens * 4
        total = 0
        result = []
        for chunk in chunks:
            clen = len(chunk.get("text", ""))
            if total + clen > max_chars:
                break
            result.append(chunk)
            total += clen
        return result

    def format_messages(self, question: str, chunks: list[dict]) -> list[dict]:
        """
        Return OpenAI-compatible messages list:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        prompt = self.build(question, chunks)
        return [
            {"role": "system", "content": prompt.system},
            {"role": "user",   "content": prompt.user},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Show all four prompt templates with sample context.
    Run: python 06_prompting/prompt_builder.py
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()
    console.rule("[bold blue]Module 6 — Prompt Engineering for RAG")

    sample_chunks = [
        {
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model generator. The retrieval system finds relevant passages from a document corpus.",
            "source": "rag_survey.pdf",
            "chunk_index": 0,
        },
        {
            "text": "RAG reduces hallucination by grounding the model's response in real retrieved documents. If the answer is not in the retrieved context, a well-prompted model should say so.",
            "source": "rag_survey.pdf",
            "chunk_index": 1,
        },
        {
            "text": "The three stages of RAG are: (1) indexing — convert documents to embeddings; (2) retrieval — find relevant chunks; (3) generation — produce an answer using the chunks as context.",
            "source": "deep_learning_book.pdf",
            "chunk_index": 3,
        },
    ]

    question = "What are the three stages of RAG and how does it reduce hallucination?"

    for template_name in ["numbered", "xml", "cot", "concise"]:
        builder = PromptBuilder(template=template_name)
        prompt = builder.build(question, sample_chunks)

        console.print(f"\n[bold cyan]Template: {template_name.upper()}[/bold cyan]")
        console.print(f"  Chunks used: {prompt.num_chunks} | Est. tokens: {prompt.estimated_tokens}")
        console.print(Panel(
            prompt.user[:600] + ("..." if len(prompt.user) > 600 else ""),
            title=f"[green]{template_name}[/green]",
            border_style="dim",
            expand=False,
        ))

    console.print("\n[bold green]✓ Prompt builder demo complete.[/bold green]")


if __name__ == "__main__":
    demo()
