"""
Module 11 — Agentic RAG
=========================
Agentic RAG transforms static retrieval into an autonomous, reasoning-driven
workflow where an LLM agent actively controls the retrieval process.

Instead of the fixed linear flow (Query → Retrieve → Generate), an Agentic RAG
system can:
    1. Route Queries      — Decide whether retrieval is needed, or if direct
                            parametric knowledge / tools / summary is appropriate.
    2. Tool Use (ReAct)   — Reason iteratively with tools (vector search, doc lookup,
                            calculator, fallback search) using Thought-Action-Observation.
    3. Multi-Hop Search   — Break complex, multi-part questions into sub-queries
                            and synthesize findings across distinct chunks.
    4. Corrective RAG     — Evaluate retrieved chunk relevance (CRAG), rewrite bad
       & Self-Reflection    queries, and verify factual grounding against hallucinations.

Run:
    python 11_agentic_rag/agentic_rag.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config_loader import load_config  # noqa: E402


def _imp(mp: str, a: str):
    """Lazy import helper for numbered modules."""
    spec = importlib.util.spec_from_file_location(a, Path(_ROOT) / mp)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return getattr(m, a)


def _get_text_content(response: Any) -> str:
    """Helper to safely extract string content from any LLM return type."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content") and response.content is not None:
        return str(response.content)
    if isinstance(response, dict):
        return str(response.get("content", ""))
    return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tools Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentTool:
    """A tool that an agent can invoke during reasoning."""
    name: str
    description: str
    func: Callable[[str], str]
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The input argument or search query for the tool",
            }
        },
        "required": ["query"],
    })

    def run(self, argument: Any) -> str:
        """Run tool with string or dict arguments."""
        try:
            if isinstance(argument, dict):
                arg_val = argument.get("query") or argument.get("expression") or next(iter(argument.values()), "")
            elif isinstance(argument, str):
                arg_str = argument.strip()
                if arg_str.startswith("{") and arg_str.endswith("}"):
                    try:
                        parsed = json.loads(arg_str)
                        if isinstance(parsed, dict):
                            arg_val = parsed.get("query") or parsed.get("expression") or next(iter(parsed.values()), arg_str)
                        else:
                            arg_val = arg_str
                    except Exception:
                        arg_val = arg_str
                else:
                    arg_val = arg_str
            else:
                arg_val = str(argument)
            return str(self.func(str(arg_val).strip()))
        except Exception as e:
            return f"Error executing tool '{self.name}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agentic Router (Adaptive Query Classification)
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRouter:
    """
    Classifies an incoming query to determine the best execution route:
      - 'DIRECT'       : Conversational or general queries needing no document lookup.
      - 'RETRIEVAL'    : Single-domain factual questions answerable by vector search.
      - 'MULTI_HOP'    : Complex, multi-part, or comparative questions needing step-by-step search.
      - 'SUMMARIZATION': Broad overview requests covering whole collections or documents.
    """

    ROUTER_PROMPT = (
        "You are an expert query routing agent. Analyze the user request and select "
        "the single most appropriate execution strategy.\n\n"
        "Options:\n"
        "  - DIRECT: Greetings, general chit-chat, simple logic, or coding syntax not needing custom docs.\n"
        "  - RETRIEVAL: Specific factual question about domain topics (AI, RAG, Transformers, Databases, etc.).\n"
        "  - MULTI_HOP: Multi-step questions, comparisons between multiple concepts, or queries requiring multiple lookups.\n"
        "  - SUMMARIZATION: High-level summary or overview of all indexed documents or broad topics.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        "{{\"route\": \"DIRECT\" | \"RETRIEVAL\" | \"MULTI_HOP\" | \"SUMMARIZATION\", \"reasoning\": \"brief explanation\"}}\n\n"
        "User Query: {query}"
    )

    def __init__(self, llm):
        self._llm = llm

    def route(self, query: str) -> dict[str, str]:
        messages = [
            {"role": "user", "content": self.ROUTER_PROMPT.format(query=query)}
        ]
        raw_resp = self._llm.generate(messages)
        raw = _get_text_content(raw_resp).strip()
        try:
            # Extract JSON substring if LLM adds markdown fences
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                route = parsed.get("route", "RETRIEVAL").upper()
                if route in ("DIRECT", "RETRIEVAL", "MULTI_HOP", "SUMMARIZATION"):
                    return {"route": route, "reasoning": parsed.get("reasoning", "")}
        except Exception:
            pass

        # Fallback default
        return {"route": "RETRIEVAL", "reasoning": "Defaulting to document retrieval"}


# ─────────────────────────────────────────────────────────────────────────────
# 3. ReAct Agent (Reasoning + Acting Loop)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """A single thought-action-observation trace step."""
    step_num: int
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """The result of an Agentic execution trace."""
    question: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    route: str = "MULTI_HOP"
    total_time_s: float = 0.0


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent for RAG.

    Executes an iterative loop supporting both Native Tool Calling (Groq / OpenAI / Ollama)
    and text-based ReAct prompting:
        Thought: What information do I need next?
        Action: tool_name
        Action Input: argument
        Observation: <result from tool execution>
        ...
        Thought: I now have enough information to answer.
        Final Answer: <grounded answer>
    """

    SYSTEM_PROMPT = (
        "You are an intelligent knowledge agent that answers user questions by reasoning step-by-step "
        "and utilizing available tools.\n\n"
        "Available Tools:\n"
        "{tool_descriptions}\n\n"
        "Format your response strictly using this cycle:\n"
        "Thought: <your reasoning about what step to take next>\n"
        "Action: <tool_name>\n"
        "Action Input: <input to the tool>\n\n"
        "When you have collected all required information to answer the question, output:\n"
        "Thought: I now have enough information to provide the final answer.\n"
        "Final Answer: <your complete, accurate, grounded response>\n\n"
        "IMPORTANT RULES:\n"
        "1. Only use tools listed under Available Tools.\n"
        "2. Do not fabricate information. Rely on Observations for document facts.\n"
        "3. Output only ONE Thought + Action + Action Input per turn."
    )

    def __init__(self, llm, tools: list[AgentTool], max_iterations: int = 5):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._max_iterations = max_iterations
        self._tools_schema = self._build_tools_schema()

    def _build_tools_schema(self) -> list[dict]:
        """Build standard OpenAI-compatible tool schemas for function calling."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas

    def _format_tools_doc(self) -> str:
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _clean_tool_name(self, name: str) -> str:
        """Strip namespaces like 'developer.document_search' -> 'document_search'."""
        return name.split(".")[-1].strip()

    def run(self, question: str, verbose: bool = True) -> AgentResult:
        start_time = time.perf_counter()
        tool_docs = self._format_tools_doc()
        system_msg = self.SYSTEM_PROMPT.format(tool_descriptions=tool_docs)

        scratchpad = f"Question: {question}\n"
        steps: list[AgentStep] = []

        for iteration in range(1, self._max_iterations + 1):
            prompt = f"{scratchpad}\nThought:"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]

            # Pass tools schema to support providers that require/support native tool calls
            try:
                response = self._llm.generate(messages, tools=self._tools_schema)
            except Exception:
                # Fallback to generating without tools if provider doesn't support them
                response = self._llm.generate(messages)

            thought = ""
            action = ""
            action_input = ""

            # ── 1. Check for Native Tool Call Object ─────────────────────────
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]
                action = self._clean_tool_name(tc.function.name)
                action_input = tc.function.arguments
                thought = getattr(response, "content", "") or f"Deciding to call {action} to gather evidence."

            # ── 2. Check for Dictionary format ──────────────────────────────
            elif isinstance(response, dict):
                if "name" in response:
                    action = self._clean_tool_name(response["name"])
                    action_input = json.dumps(response.get("arguments", {}))
                    thought = f"Calling tool {action}."
                elif "tool_calls" in response and len(response["tool_calls"]) > 0:
                    tc = response["tool_calls"][0]
                    action = self._clean_tool_name(tc["function"]["name"])
                    action_input = tc["function"]["arguments"]
                    thought = response.get("content", "") or f"Calling tool {action}."
                else:
                    content_str = str(response.get("content", ""))
                    if "Final Answer:" in content_str:
                        parts = content_str.split("Final Answer:", 1)
                        steps.append(AgentStep(
                            step_num=iteration,
                            thought=parts[0].replace("Thought:", "").strip() or "Final answer reached.",
                            action="None",
                            action_input="None",
                            observation="Task Complete",
                        ))
                        return AgentResult(
                            question=question,
                            final_answer=parts[1].strip(),
                            steps=steps,
                            total_time_s=time.perf_counter() - start_time,
                        )

            # ── 3. Parse Text-Based ReAct Response ───────────────────────────
            if not action:
                full_turn = _get_text_content(response).strip()
                if not full_turn.startswith("Thought:"):
                    full_turn = f"Thought: {full_turn}"

                # Check if Final Answer reached in text
                if "Final Answer:" in full_turn:
                    parts = full_turn.split("Final Answer:", 1)
                    final_answer = parts[1].strip()
                    thought_part = parts[0].replace("Thought:", "").strip()
                    steps.append(AgentStep(
                        step_num=iteration,
                        thought=thought_part or "Final answer reached.",
                        action="None",
                        action_input="None",
                        observation="Task Complete",
                    ))
                    return AgentResult(
                        question=question,
                        final_answer=final_answer,
                        steps=steps,
                        total_time_s=time.perf_counter() - start_time,
                    )

                # Check if text contains JSON tool call like {"name": "document_search", "arguments": ...}
                json_tool_match = re.search(r'\{[^{}]*"name"\s*:\s*"([a-zA-Z0-9_.-]+)"[^{}]*\}', full_turn)
                if json_tool_match:
                    try:
                        parsed = json.loads(json_tool_match.group(0))
                        action = self._clean_tool_name(parsed.get("name", ""))
                        action_input = json.dumps(parsed.get("arguments", {}))
                        thought = "Executing tool call determined by model."
                    except Exception:
                        pass

                if not action:
                    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", full_turn, re.DOTALL)
                    action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-.]+)", full_turn)
                    input_match = re.search(r"Action Input:\s*(.*?)(?=\nThought:|\nAction:|$)", full_turn, re.DOTALL)

                    thought = thought_match.group(1).strip() if thought_match else "Analyzing next step..."
                    raw_act = action_match.group(1).strip() if action_match else ""
                    action = self._clean_tool_name(raw_act)
                    action_input = input_match.group(1).strip() if input_match else ""

            # If no valid tool was requested, check if the response is already a direct answer
            if not action or action not in self._tools:
                full_text = _get_text_content(response).strip()
                if len(steps) > 0 or ("answer" in full_text.lower() and len(full_text) > 40):
                    final_ans = full_text.replace("Thought:", "").replace("Final Answer:", "").strip()
                    return AgentResult(
                        question=question,
                        final_answer=final_ans,
                        steps=steps,
                        total_time_s=time.perf_counter() - start_time,
                    )
                obs = f"Invalid tool '{action}'. Available tools: {list(self._tools.keys())}"
            else:
                tool = self._tools[action]
                obs = tool.run(action_input)

            # Record step
            steps.append(AgentStep(
                step_num=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=obs[:400] + ("..." if len(obs) > 400 else ""),
            ))

            # Update scratchpad for next iteration
            scratchpad += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {obs}\n"

        # If max iterations reached, synthesize fallback
        synth_messages = [
            {"role": "system", "content": "Synthesize a concise final answer based strictly on the observations above."},
            {"role": "user", "content": f"{scratchpad}\nProvide the best Final Answer possible with what you gathered."},
        ]
        fallback_resp = self._llm.generate(synth_messages)
        fallback_answer = _get_text_content(fallback_resp).strip()
        if "Final Answer:" in fallback_answer:
            fallback_answer = fallback_answer.split("Final Answer:", 1)[1].strip()

        return AgentResult(
            question=question,
            final_answer=fallback_answer,
            steps=steps,
            total_time_s=time.perf_counter() - start_time,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Corrective RAG (CRAG) & Self-Reflection
# ─────────────────────────────────────────────────────────────────────────────

class CorrectiveRAG:
    """
    Corrective RAG (CRAG) — Yan et al., 2024.

    Self-reflective retrieval mechanism that:
      1. Grades retrieved documents for relevance.
      2. If documents are irrelevant / ambiguous, rewrites the search query.
      3. Validates whether generated response is faithful to the context (hallucination check).
    """

    GRADE_PROMPT = (
        "You are an evaluator assessing the relevance of a retrieved document chunk to a user question.\n\n"
        "Question: {question}\n\n"
        "Document Chunk:\n{chunk_text}\n\n"
        "Criteria: Does the document chunk contain information useful for answering the question?\n"
        "Respond with ONLY 'RELEVANT' or 'IRRELEVANT'."
    )

    REWRITE_PROMPT = (
        "The previous search query failed to retrieve relevant documents for the user's question.\n"
        "Original Question: {question}\n"
        "Previous Query: {last_query}\n\n"
        "Formulate a better, more specific search query optimized for vector search to retrieve the necessary facts.\n"
        "Output ONLY the new search query, nothing else."
    )

    HALLUCINATION_CHECK_PROMPT = (
        "You are a strict factual judge. Check if the generated answer is fully grounded in the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Is every factual claim in the answer supported by the context?\n"
        "Respond with ONLY 'GROUNDED' or 'UNGROUNDED'."
    )

    def __init__(self, pipeline, llm):
        self._pipeline = pipeline
        self._llm = llm

    def grade_chunk(self, question: str, chunk_text: str) -> bool:
        messages = [
            {"role": "user", "content": self.GRADE_PROMPT.format(question=question, chunk_text=chunk_text[:600])}
        ]
        res_raw = self._llm.generate(messages)
        res = _get_text_content(res_raw).strip().upper()
        return "RELEVANT" in res

    def rewrite_query(self, question: str, last_query: str) -> str:
        messages = [
            {"role": "user", "content": self.REWRITE_PROMPT.format(question=question, last_query=last_query)}
        ]
        res_raw = self._llm.generate(messages)
        return _get_text_content(res_raw).strip().strip('"')

    def check_hallucination(self, context: str, answer: str) -> bool:
        messages = [
            {"role": "user", "content": self.HALLUCINATION_CHECK_PROMPT.format(context=context[:2000], answer=answer)}
        ]
        res_raw = self._llm.generate(messages)
        res = _get_text_content(res_raw).strip().upper()
        return "GROUNDED" in res

    def _retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Safely retrieve from pipeline using available retriever method."""
        if hasattr(self._pipeline, "retrieve"):
            return self._pipeline.retrieve(query, top_k=top_k)
        elif hasattr(self._pipeline, "_get_retriever"):
            return self._pipeline._get_retriever().retrieve(query, top_k=top_k)
        elif hasattr(self._pipeline, "_retriever") and self._pipeline._retriever is not None:
            return self._pipeline._retriever.retrieve(query, top_k=top_k)
        elif hasattr(self._pipeline, "_store") and hasattr(self._pipeline, "_embedder"):
            emb = self._pipeline._embedder.embed_one(query)
            return self._pipeline._store.query(emb, top_k=top_k)
        return []

    def run(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Execute the Corrective RAG workflow."""
        current_query = question
        retries = 0
        max_retries = 2
        filtered_chunks = []
        chunks = []

        while retries <= max_retries:
            chunks = self._retrieve(current_query, top_k=top_k)
            relevant_chunks = [c for c in chunks if self.grade_chunk(question, c["text"])]

            if relevant_chunks:
                filtered_chunks = relevant_chunks
                break

            # If no chunks were graded relevant, reformulate query
            retries += 1
            if retries <= max_retries:
                current_query = self.rewrite_query(question, current_query)

        # If still no relevant chunks found, fallback to top retrieved chunks with a warning
        if not filtered_chunks:
            filtered_chunks = chunks[:2]
            action_status = "Fallback: low-confidence retrieval after query rewriting"
        else:
            action_status = f"Success: {len(filtered_chunks)} relevant chunk(s) identified (queries attempted: {retries + 1})"

        context_str = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(filtered_chunks)])
        gen_messages = [
            {"role": "system", "content": "Answer the question factually based ONLY on the provided context. If unsure, state lack of info."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"},
        ]
        raw_resp = self._llm.generate(gen_messages)
        raw_answer = _get_text_content(raw_resp).strip()
        is_grounded = self.check_hallucination(context_str, raw_answer)

        return {
            "question": question,
            "final_query_used": current_query,
            "status": action_status,
            "retries": retries,
            "chunks_used": filtered_chunks,
            "answer": raw_answer,
            "is_grounded": is_grounded,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Agentic RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    Unified Agentic RAG System:
      1. Classifies query intent via AgenticRouter.
      2. Direct queries answered without retrieval overhead.
      3. Standard queries answered via Corrective RAG (CRAG).
      4. Multi-hop queries answered via ReAct Agent reasoning loop.
    """

    def __init__(self, pipeline=None):
        if pipeline is None:
            RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")
            self._pipeline = RAGPipeline()
        else:
            self._pipeline = pipeline

        self._llm = self._pipeline._llm
        self._router = AgenticRouter(self._llm)
        self._crag = CorrectiveRAG(self._pipeline, self._llm)

        # Build tools for ReAct Agent
        tools = self._build_tools()
        self._agent = ReActAgent(self._llm, tools=tools, max_iterations=5)

    def _build_tools(self) -> list[AgentTool]:
        """Construct tools for the agent."""
        def search_docs(query: str) -> str:
            chunks = self._crag._retrieve(query, top_k=3)
            if not chunks:
                return "No matching document chunks found."
            formatted = []
            for i, c in enumerate(chunks, 1):
                src = c.get("source", "unknown")
                txt = c.get("text", "").replace("\n", " ")[:250]
                formatted.append(f"[{i}] ({src}): {txt}")
            return "\n".join(formatted)

        def list_sources(_: str = "") -> str:
            sources = self._pipeline.get_indexed_sources()
            if not sources:
                return "No documents currently indexed."
            return f"Indexed documents ({len(sources)}): " + ", ".join(sources)

        def calc(expression: str) -> str:
            # Safe basic evaluation for numeric queries
            allowed = re.sub(r"[^0-9+\-*/().\s]", "", expression)
            try:
                result = eval(allowed, {"__builtins__": None}, {})
                return str(result)
            except Exception as err:
                return f"Math eval error: {err}"

        return [
            AgentTool(
                name="document_search",
                description="Search the indexed vector store for factual information on AI, RAG, ML, Transformers, etc.",
                func=search_docs,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up in the vector database",
                        }
                    },
                    "required": ["query"],
                },
            ),
            AgentTool(
                name="list_available_sources",
                description="Get a list of all indexed documents in the knowledge base.",
                func=list_sources,
                parameters={
                    "type": "object",
                    "properties": {},
                },
            ),
            AgentTool(
                name="calculator",
                description="Perform arithmetic calculations and numeric comparisons.",
                func=calc,
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate, e.g. '1024 / 384'",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ]

    def query(self, question: str, force_route: str | None = None) -> dict[str, Any]:
        """Route and execute query agentically."""
        if force_route:
            route_info = {"route": force_route.upper(), "reasoning": "Forced by caller"}
        else:
            route_info = self._router.route(question)

        route = route_info["route"]

        if route == "DIRECT":
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide a direct, concise answer."},
                {"role": "user", "content": question},
            ]
            ans_raw = self._llm.generate(messages)
            ans = _get_text_content(ans_raw).strip()
            return {
                "route": route,
                "reasoning": route_info["reasoning"],
                "answer": ans,
                "steps": [],
                "sources": [],
            }

        elif route == "MULTI_HOP":
            res = self._agent.run(question)
            return {
                "route": route,
                "reasoning": route_info["reasoning"],
                "answer": res.final_answer,
                "steps": res.steps,
                "latency_s": res.total_time_s,
            }

        else:  # RETRIEVAL or SUMMARIZATION via Corrective RAG
            crag_res = self._crag.run(question)
            return {
                "route": route,
                "reasoning": route_info["reasoning"],
                "answer": crag_res["answer"],
                "status": crag_res["status"],
                "is_grounded": crag_res["is_grounded"],
                "sources": crag_res["chunks_used"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Interactive Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.rule("[bold magenta]Module 11 — Agentic RAG")

    console.print("""
[bold]Agentic RAG Architecture:[/bold]
  • [cyan]Dynamic Router[/cyan]       → Classifies question type (Direct vs Retrieval vs Multi-Hop)
  • [green]ReAct Agent[/green]          → Iterative Thought-Action-Observation loop with tool execution
  • [yellow]Corrective RAG (CRAG)[/yellow] → Document grading, query reformulation & hallucination checks
    """)

    try:
        RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")
        pipeline = RAGPipeline()
    except Exception as e:
        console.print(f"[red]Could not load pipeline: {e}[/red]")
        return

    # Check if documents are indexed; if not, index raw data
    if not pipeline.get_indexed_sources():
        raw_dir = Path("data/raw")
        if list(raw_dir.glob("*.pdf")):
            console.print("[yellow]Indexing documents in data/raw/...[/yellow]")
            pipeline.index_directory(raw_dir)
        else:
            console.print("[yellow]Run: python data/download_samples.py first to download sample data[/yellow]")
            return

    agentic_sys = AgenticRAGPipeline(pipeline=pipeline)

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 1: Agentic Router in action
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]1. Dynamic Query Routing[/bold cyan]")
    test_queries = [
        "Hello! Can you write a Python function to reverse a list?",
        "What is the self-attention mechanism in Transformers?",
        "Compare the role of deep learning in speech recognition vs natural language processing.",
    ]

    for q in test_queries:
        decision = agentic_sys._router.route(q)
        console.print(f"[bold yellow]Q:[/bold yellow] {q}")
        console.print(f"  [bold green]➔ Route:[/bold green] [bold]{decision['route']}[/bold] ({decision['reasoning']})\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 2: ReAct Multi-Hop Agent Reasoning Loop
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold green]2. ReAct Multi-Hop Agent Execution[/bold green]")
    multi_hop_q = "What is retrieval augmented generation, and which document in the index discusses it?"
    console.print(f"[bold yellow]Multi-Hop Question:[/bold yellow] {multi_hop_q}\n")

    result = agentic_sys._agent.run(multi_hop_q)

    # Display ReAct trace table
    table = Table(title="ReAct Agent Step-by-Step Trace", show_lines=True)
    table.add_column("Step", style="dim", width=6)
    table.add_column("Thought", style="cyan")
    table.add_column("Action / Tool", style="green")
    table.add_column("Observation", style="white")

    for s in result.steps:
        obs_display = s.observation[:120] + "..." if len(s.observation) > 120 else s.observation
        action_display = f"{s.action}({s.action_input})" if s.action != "None" else "None"
        table.add_row(str(s.step_num), s.thought, action_display, obs_display)

    console.print(table)
    console.print(Panel(result.final_answer, title="[bold green]ReAct Final Answer[/bold green]", expand=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Demo 3: Corrective RAG (CRAG) & Self-Reflection
    # ─────────────────────────────────────────────────────────────────────────
    console.rule("[bold yellow]3. Corrective RAG (CRAG) & Grounding Check[/bold yellow]")
    crag_q = "How does vector indexing speed up nearest neighbor search in vector databases?"
    console.print(f"[bold yellow]CRAG Question:[/bold yellow] {crag_q}\n")

    crag_res = agentic_sys._crag.run(crag_q, top_k=3)
    console.print(f"[bold]Query used:[/bold] {crag_res['final_query_used']}")
    console.print(f"[bold]CRAG Status:[/bold] {crag_res['status']}")
    console.print(f"[bold]Answer Grounded (No Hallucination):[/bold] {'✓ Yes' if crag_res['is_grounded'] else '✗ No'}")
    console.print(Panel(crag_res["answer"], title="[bold yellow]CRAG Grounded Answer[/bold yellow]", expand=False))


if __name__ == "__main__":
    demo()
