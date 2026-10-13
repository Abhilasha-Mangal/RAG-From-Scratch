"""
Module 7 — LLM Integration
============================
Pluggable LLM clients — swap provider in config.yaml with no code changes.

Providers:
    1. OllamaLLM   — local models (Llama 3, Mistral, Phi-3, Gemma) — FREE
    2. GroqLLM     — cloud API with free tier, very fast — NEEDS API KEY
    3. OpenAILLM   — GPT-4o / GPT-4o-mini — NEEDS API KEY

All providers expose:
    generate(messages)         → str (full response)
    stream(messages)           → Iterator[str] (token by token)

Recommended setup (no API key):
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. Set provider: ollama in config.yaml
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

# Load API keys from the project's .env (works no matter where you run from)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """All LLM providers share this interface."""

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """Generate a response from a list of messages."""
        ...

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Stream response tokens one by one."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollama — local, free
# ─────────────────────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """
    Generate text using a model running locally in Ollama.

    Setup:
        1. Install Ollama: https://ollama.ai
        2. ollama pull llama3.2       (3B — fast, 2GB RAM)
        3. ollama pull mistral        (7B — balanced)
        4. ollama pull phi3           (3.8B — great reasoning)
        5. ollama pull gemma2         (2B — tiny but capable)

    No internet required after model download. Runs on CPU (slow) or GPU.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        import ollama as _ollama
        self._ollama = _ollama
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        print(f"[llm] Using Ollama model: {model}")

    def generate(self, messages: list[dict]) -> str:
        response = self._ollama.chat(
            model=self._model,
            messages=messages,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        )
        return response["message"]["content"]

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._ollama.chat(
            model=self._model,
            messages=messages,
            stream=True,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Groq — cloud API, free tier, extremely fast
# ─────────────────────────────────────────────────────────────────────────────

class GroqLLM(BaseLLM):
    """
    Generate text using Groq's API (free tier available).

    Free tier available with rate limits. Groq is a very fast inference API.

    Setup:
        1. Get free API key: https://console.groq.com
        2. Set GROQ_API_KEY in your .env file

    Model names change over time. List the ones your key can use with:
        python 07_llm_integration/llm_client.py --models

    Examples: openai/gpt-oss-20b (fast, default), openai/gpt-oss-120b (smarter)
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")

        self._client = OpenAI(api_key=key, base_url=self.GROQ_BASE_URL)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        print(f"[llm] Using Groq model: {model}")

    def generate(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def model_name(self) -> str:
        return f"groq/{self._model}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenAI
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """
    Generate text using OpenAI's API.
    Models: gpt-4o-mini (cheap), gpt-4o (best quality)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        print(f"[llm] Using OpenAI model: {model}")

    def generate(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(config: dict) -> BaseLLM:
    """
    Build an LLM from the llm section of config.yaml.

    Example config:
        llm:
          provider: ollama
          ollama:
            model: llama3.2
            temperature: 0.1
    """
    provider = config.get("provider", "ollama")

    if provider == "ollama":
        cfg = config.get("ollama", {})
        return OllamaLLM(
            model=cfg.get("model", "llama3.2"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
        )

    if provider == "groq":
        cfg = config.get("groq", {})
        return GroqLLM(
            model=cfg.get("model", "openai/gpt-oss-20b"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
        )

    if provider == "openai":
        cfg = config.get("openai", {})
        return OpenAILLM(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 1024),
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Test the LLM with a simple RAG prompt.
    Tries Ollama first; falls back to Groq if configured.
    Run: python 07_llm_integration/llm_client.py
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.rule("[bold blue]Module 7 — LLM Integration")

    # Try to detect which provider is available
    llm = None

    # Option 1: Ollama (preferred — no API key)
    try:
        import ollama
        ollama.list()  # will fail if server is not running
        llm = OllamaLLM(model="llama3.2", temperature=0.1)
        console.print("[green]✓ Ollama server detected[/green]")
    except Exception:
        console.print("[yellow]Ollama not running. Trying Groq...[/yellow]")

    # Option 2: Groq (free API key)
    if llm is None and os.environ.get("GROQ_API_KEY"):
        try:
            llm = GroqLLM(model="openai/gpt-oss-20b", temperature=0.1)
            console.print("[green]✓ Groq API key found[/green]")
        except Exception as e:
            console.print(f"[red]Groq failed: {e}[/red]")

    if llm is None:
        console.print(
            "[red]No LLM available.\n"
            "• Install Ollama: https://ollama.ai and run 'ollama pull llama3.2'\n"
            "• OR set GROQ_API_KEY in .env (free at https://console.groq.com)[/red]"
        )
        return

    console.print(f"[cyan]Using: {llm.model_name}[/cyan]\n")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer based only on the context."
        },
        {
            "role": "user",
            "content": """Context:
RAG stands for Retrieval-Augmented Generation. It improves LLM responses by
providing relevant retrieved documents as context. This reduces hallucination
because the model answers from real sources rather than from memory.

Question: What does RAG stand for and why is it useful?"""
        }
    ]

    # Non-streaming
    console.print("[bold]Non-streaming response:[/bold]")
    response = llm.generate(messages)
    console.print(Panel(response, title=llm.model_name, border_style="green"))

    # Streaming
    console.print("[bold]Streaming response (token by token):[/bold] ", end="")
    for token in llm.stream(messages):
        console.print(token, end="", highlight=False)
    console.print()
    console.print("\n[bold green]✓ LLM integration demo complete.[/bold green]")


def list_groq_models():
    """Print the model IDs available to your GROQ_API_KEY."""
    llm = GroqLLM()
    for m in sorted(x.id for x in llm._client.models.list().data):
        print(m)


if __name__ == "__main__":
    import sys
    if "--models" in sys.argv:
        list_groq_models()
    else:
        demo()
