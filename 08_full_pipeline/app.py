"""
Module 8 — Gradio Web UI
=========================
A browser-based interface for the RAG pipeline.
You can upload your own PDFs and ask questions — no terminal needed.

Run:
    python 08_full_pipeline/app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402


def _imp(mp, a):
    spec = importlib.util.spec_from_file_location(a, Path(_ROOT) / mp)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return getattr(m, a)


RAGPipeline = _imp("08_full_pipeline/pipeline.py", "RAGPipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline singleton (shared across Gradio callbacks)
# ─────────────────────────────────────────────────────────────────────────────

pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────

def _turn(question: str, answer: str) -> list[dict]:
    """One chat turn in Gradio's 'messages' format."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def upload_and_index(files, status_text: str) -> tuple[str, str]:
    """
    Called when the user uploads PDF files.
    Returns (status message, updated sources list).
    """
    if not files:
        return "No files uploaded.", status_text

    p = get_pipeline()
    total = 0
    indexed_names = []

    for file_obj in files:
        file_path = getattr(file_obj, "name", file_obj)  # str path (Gradio 5+) or file object
        try:
            n = p.index_file(file_path)
            total += n
            indexed_names.append(Path(file_path).name)
        except Exception as e:
            return f"❌ Error indexing {Path(file_path).name}: {e}", status_text

    sources = p.get_indexed_sources()
    sources_str = "\n".join(f"• {s}" for s in sources) if sources else "None"
    status = f"✅ Indexed {total} chunks from {len(indexed_names)} file(s).\n\nIndexed documents:\n{sources_str}"
    return status, sources_str


def answer_question(
    question: str,
    history: list,
    top_k: int,
    show_sources: bool,
) -> tuple[list, str]:
    """
    Called when the user submits a question.
    Returns updated chat history and sources text.
    """
    if not question.strip():
        return history, ""

    p = get_pipeline()

    # Check if anything is indexed
    sources_list = p.get_indexed_sources()
    if not sources_list:
        history = history + _turn(question, "⚠️ No documents indexed yet. Please upload a PDF first.")
        return history, ""

    try:
        response = p.query(question, top_k=int(top_k))
    except Exception as e:
        history = history + _turn(question, f"❌ Error: {e}")
        return history, ""

    # Build sources text
    sources_md = ""
    if show_sources and response.sources:
        lines = [f"**Sources ({response.num_chunks_retrieved} chunks retrieved | {response.latency_s:.1f}s):**\n"]
        for i, s in enumerate(response.sources, 1):
            text_preview = s.get("text", "")[:120].replace("\n", " ")
            lines.append(f"**[{i}]** `{s.get('source', '?')}` — {text_preview}...")
        sources_md = "\n".join(lines)

    history = history + _turn(question, response.answer)
    return history, sources_md


def stream_answer(question: str, history: list, top_k: int):
    """Stream tokens for a smoother demo experience."""
    if not question.strip():
        yield history, ""
        return

    p = get_pipeline()

    if not p.get_indexed_sources():
        history = history + _turn(question, "⚠️ No documents indexed yet. Please upload a PDF first.")
        yield history, ""
        return

    history = history + _turn(question, "")
    collected = ""

    try:
        for token in p.stream_query(question, top_k=int(top_k)):
            collected += token
            history[-1] = {"role": "assistant", "content": collected}
            yield history, ""
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Error: {e}"}
        yield history, ""


def reset_index() -> tuple[str, str, list]:
    """Clear the entire index."""
    p = get_pipeline()
    p.reset_index()
    return "🗑️ Index cleared. Upload new documents to start fresh.", "", []


# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
#header { text-align: center; padding: 1rem 0 0.5rem; }
#header h1 { font-size: 1.8rem; margin: 0; }
#header p  { color: #666; margin: 0.3rem 0 0; }
.source-box { background: #f7f8fa; border-radius: 6px; padding: 0.8rem; font-size: 0.85rem; }
"""

with gr.Blocks(title="RAG From Scratch") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="header">
            <h1>📚 RAG From Scratch</h1>
            <p>Build Your First Retrieval-Augmented Generation Pipeline</p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Document Upload & Indexing ──────────────────────────────────
        with gr.Tab("📄 Upload & Index"):
            gr.Markdown("""
### Step 1: Upload your documents
Upload one or more PDF files. The pipeline will:
1. Extract text using **IBM Docling** (preserves tables, headings)
2. Split into chunks using the **Recursive** strategy
3. Embed each chunk using **sentence-transformers** (local, free)
4. Store embeddings in **ChromaDB** (or Redis if configured)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload PDF files",
                        file_count="multiple",
                        file_types=[".pdf"],
                    )
                    upload_btn = gr.Button("🔄 Index Documents", variant="primary")
                    reset_btn = gr.Button("🗑️ Reset Index", variant="stop")

                with gr.Column(scale=3):
                    index_status = gr.Textbox(
                        label="Indexing Status",
                        lines=8,
                        interactive=False,
                        placeholder="Upload files and click Index...",
                    )
                    sources_display = gr.Textbox(
                        label="Indexed Sources",
                        lines=4,
                        interactive=False,
                        placeholder="No documents indexed yet.",
                    )

        # ── Tab 2: Ask Questions ───────────────────────────────────────────────
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("""
### Step 2: Ask questions about your documents
The pipeline retrieves relevant chunks and generates a grounded answer.
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450,
                    )
                    with gr.Row():
                        question_box = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about the indexed documents...",
                            scale=5,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                with gr.Column(scale=2):
                    gr.Markdown("**Settings**")
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Top-K chunks to retrieve",
                        info="More chunks = more context but slower",
                    )
                    show_sources_cb = gr.Checkbox(
                        value=True, label="Show source chunks"
                    )
                    clear_btn = gr.Button("🗑️ Clear Chat")

                    sources_out = gr.Markdown(
                        label="Retrieved Sources",
                        elem_classes=["source-box"],
                    )

        # ── Tab 3: Pipeline Explorer ───────────────────────────────────────────
        with gr.Tab("🔍 Pipeline Explorer"):
            gr.Markdown("""
### How RAG Works — Pipeline Visualization

```
User Question
      │
      ▼
┌─────────────────┐
│   Embed Query   │  ← sentence-transformers (local)
└────────┬────────┘
         │  query vector
         ▼
┌─────────────────┐
│  Vector Search  │  ← ChromaDB or Redis (HNSW index)
│   (Top-K ANN)   │
└────────┬────────┘
         │  top-k chunks
         ▼
┌─────────────────┐
│  Build Prompt   │  ← inject chunks as numbered context
└────────┬────────┘
         │  messages list
         ▼
┌─────────────────┐
│  LLM Generate   │  ← Ollama (local) or Groq/OpenAI (API)
└────────┬────────┘
         │
         ▼
      Answer
```

### Current Configuration
            """)

            config_display = gr.JSON(
                label="config.yaml",
                value=None,
            )
            load_config_btn = gr.Button("Load Current Config")

        # ── Tab 4: About ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## RAG From Scratch

This application is the **hands-on working project** for
*"RAG From Scratch: Build Your First Retrieval-Augmented Generation Pipeline"*.

### Modules Covered
| Module | Topic |
|--------|-------|
| 0 | What is RAG and why does it exist? |
| 1 | Document Loading with IBM Docling |
| 2 | Chunking Strategies |
| 3 | Embeddings with sentence-transformers |
| 4 | Vector Databases (ChromaDB + Redis) |
| 5 | Retrieval (top-k, MMR, Hybrid, ReRank) |
| 6 | Prompt Engineering |
| 7 | LLM Integration (Ollama, Groq, OpenAI) |
| 8 | Full Pipeline (this app!) |
| 9 | Evaluation with RAGAS |
| 10 | Advanced Techniques |

### Tech Stack (All Open-Source / Free)
- **Extraction**: IBM Docling
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (default) or Redis + RedisVL
- **LLM**: Ollama (local) or Groq (free API)
- **UI**: Gradio

### Getting Started
```bash
# 1. Install dependencies
pip install -r requirements-minimal.txt

# 2. Download sample documents
python data/download_samples.py

# 3. Install and start Ollama (optional, for local LLM)
# https://ollama.ai
ollama pull llama3.2

# 4. Run the app
python 08_full_pipeline/app.py
```
            """)

    # ── Event wiring ────────────────────────────────────────────────────────────
    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_input, index_status],
        outputs=[index_status, sources_display],
    )

    reset_btn.click(
        fn=reset_index,
        outputs=[index_status, sources_display, chatbot],
    )

    ask_btn.click(
        fn=stream_answer,
        inputs=[question_box, chatbot, top_k_slider],
        outputs=[chatbot, sources_out],
    )

    question_box.submit(
        fn=stream_answer,
        inputs=[question_box, chatbot, top_k_slider],
        outputs=[chatbot, sources_out],
    )

    clear_btn.click(fn=lambda: [], outputs=[chatbot])

    def load_cfg():
        import yaml
        with open("config.yaml") as f:
            return yaml.safe_load(f)

    load_config_btn.click(fn=load_cfg, outputs=[config_display])


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public URL
        inbrowser=True,
    )
