# RAG From Scratch: Build Your First Retrieval-Augmented Generation Pipeline

[![GitHub Stars](https://img.shields.io/github/stars/Abhilasha-Mangal/RAG-From-Scratch?style=social)](https://github.com/Abhilasha-Mangal/RAG-From-Scratch)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A complete, runnable Retrieval-Augmented Generation (RAG) project in 11 small modules. Every step is a separate script you can run on its own and see the output. Everything is open source, and **no API key is required** (local embeddings + local LLM through Ollama).

⭐ **If you find this repository helpful, please consider giving it a star on GitHub!** It helps others discover the project.

```
 PDF / HTML / DOCX ──▶ Docling ──▶ Chunking ──▶ Embeddings ──▶ Vector DB (Chroma / Redis)
                                                                       │
 Answer + sources ◀── LLM (Ollama / Groq / OpenAI) ◀── Prompt ◀── Retrieval (top-k / MMR / hybrid / rerank)
```

| What | Default (free, local) | Alternatives |
|------|-----------------------|--------------|
| Document extraction | IBM Docling | pypdf (fallback) |
| Chunking | Recursive | fixed, sentence, semantic |
| Embeddings | `all-MiniLM-L6-v2` (384-d, CPU) | OpenAI, Ollama |
| Vector DB | ChromaDB (no server) | Redis Stack |
| Retrieval | Hybrid (BM25 + vectors) | similarity, MMR, rerank |
| LLM | Ollama `llama3.2` | Groq (free tier), OpenAI |
| UI | Gradio | CLI |

---

## Contents

1. [Setup](#1-setup)
2. [Data: download or add your own](#2-data-download-or-add-your-own)
3. [Run each step and see its output](#3-run-each-step-and-see-its-output)
4. [End-to-end pipeline](#4-end-to-end-pipeline)
5. [LLM setup and environment variables](#5-llm-setup-and-environment-variables)
6. [Using Redis instead of ChromaDB](#6-using-redis-instead-of-chromadb)
7. [Configuration reference](#7-configuration-reference)
8. [Project structure](#8-project-structure)
9. [Troubleshooting](#9-troubleshooting)
10. [Research and related work](#10-research-and-related-work)

---

## 1. Setup

**Requirements:** Python 3.10+ (tested on 3.12), about 3 GB of free disk space for the Python packages and models, and optionally Docker (only for Redis).

```bash
git clone <your-repo-url>
cd "RAG From Scratch"

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements-minimal.txt
```

`requirements-minimal.txt` has everything for the default local setup. `requirements.txt` adds Redis, OpenAI, RAGAS and LangChain.

**First-run downloads (one time, needs internet):**

| When | What is downloaded | Size |
|------|--------------------|------|
| First `loader.py` run | Docling layout and table models | a few hundred MB |
| First embedding run | `all-MiniLM-L6-v2` from Hugging Face | about 90 MB |
| `ollama pull llama3.2` | The local LLM | about 2 GB |

After that everything works offline.

---

## 2. Data: download or add your own

### Option A: download the sample data

```bash
python data/download_samples.py
```

This fetches 10 Wikipedia articles and converts them to PDFs in `data/raw/`:

```
  ✓ artificial_intelligence.pdf                   71 KB
  ✓ machine_learning.pdf                          47 KB
  ✓ natural_language_processing.pdf               28 KB
  ✓ transformers.pdf                              129 KB
  ✓ information_retrieval.pdf                     23 KB
  ✓ vector_database.pdf                           5 KB
  ✓ large_language_model.pdf                      49 KB
  ✓ deep_learning.pdf                             47 KB
  ✓ retrieval_augmented_generation.pdf            10 KB
  ✓ semantic_search.pdf                           4 KB
Done: 10 downloaded, 0 skipped.
```

To download different Wikipedia topics, edit the `WIKIPEDIA_TOPICS` list at the top of [`data/download_samples.py`](data/download_samples.py). Each entry is `("Wikipedia page title", "output_file.pdf")`.

### Option B: add your own documents

1. Copy your PDFs (books, papers, notes) into `data/raw/`.
2. Run the pipeline (Section 4). It indexes everything in that folder.

Other ways to load data from Python:

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pipeline", "08_full_pipeline/pipeline.py")
mod = importlib.util.module_from_spec(spec); sys.modules["pipeline"] = mod; spec.loader.exec_module(mod)

p = mod.RAGPipeline()
p.index_file("my_notes/chapter1.pdf")                       # one file
p.index_directory("my_notes/", glob="**/*.pdf")             # a folder
p.index_directory("my_notes/", glob="**/*.docx")            # Word files (Docling reads them)
p.index_url("https://arxiv.org/pdf/2005.11401")             # a PDF or web page by URL
```

Or use the web app (Section 4) and upload PDFs from the **Upload & Index** tab.

Re-indexing the same file is safe: chunks are stored by `<filename>_chunk_<n>`, so they are overwritten, not duplicated.

---

## 3. Run each step and see its output

Run all commands from the project root with the virtual environment active. Each script prints a result you can inspect. The outputs below are condensed from real runs on the sample data; your timings and scores will differ a little.

### Step 1. Document loading (Docling)

```bash
python 01_document_loading/loader.py
```

Reads every PDF in `data/raw/`, extracts text, tables and headings, and saves Markdown to `data/processed/`.

```
                         Loaded Documents
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ File                               ┃ Pages ┃ Characters ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ vector_database.pdf                │     2 │      3,791 │
│ deep_learning.pdf                  │    17 │     57,172 │
│ transformers.pdf                   │   107 │     76,574 │
│ retrieval_augmented_generation.pdf │     4 │     10,627 │
│ ...                                │       │            │
```

**See the result:** open `data/processed/vector_database.md`. It is clean Markdown.

### Step 2. Chunking

```bash
python 02_chunking/chunker.py
```

Splits text into pieces and compares strategies. It uses the first file in `data/processed/` (or a built-in sample if that folder is empty).

```
Strategy        # Chunks   Min chars   Max chars   Avg chars
Fixed (512)           12          72         512         475
Sentence (5)           7         374        1033         843
Recursive (512)       15         250         526         391
```

Fixed-size chunks can cut sentences in half. Recursive splits on paragraphs, then sentences, then words, so chunks stay readable. This is the default. Change it in `config.yaml` under `chunking:`.

### Step 3. Embeddings

```bash
python 03_embeddings/embedder.py
```

Turns sentences into 384-number vectors and compares them with cosine similarity.

```
[embed] Model ready. Dimension: 384
Cosine Similarities (sample):
  RAG1 vs RAG2                 0.415 ████████████
  RAG1 vs Bio1                 0.021
  Bio1 vs Bio2                 0.125 ███
PCA plot saved → data/processed/embeddings_pca.png
```

**See the result:** open `data/processed/embeddings_pca.png`. It is a 2D map where sentences about the same topic sit close together.

### Step 4. Vector databases

```bash
python 04_vector_db/chroma/chroma_store.py      # ChromaDB, no setup
python 04_vector_db/redis/redis_store.py        # Redis (needs Docker, see Section 6)
python 04_vector_db/compare_dbs.py              # side-by-side benchmark
```

The Chroma demo stores 8 small chunks and runs three searches:

```
Query: How does RAG reduce hallucinations?
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Score ┃ Source           ┃ Text                                              ┃
│ 0.649 │ intro_to_rag.txt │ answer. RAG has three stages: indexing (process   │
│ 0.214 │ intro_to_rag.txt │ Retrieval-Augmented Generation (RAG) improves     │
```

The score is cosine similarity (higher is closer). `compare_dbs.py` prints a table with ingest time, query time and setup effort for each database.

### Step 5. Retrieval strategies

```bash
python 05_retrieval/retriever.py
```

Indexes a small built-in corpus and runs the same question through **similarity**, **MMR** and **hybrid** retrieval so you can compare:

```
Query: How does retrieval work in RAG?

Similarity (top-k)   → 3 results from rag_guide.txt (near-duplicates)
MMR (λ=0.7)          → rag_guide.txt, ai_overview.txt, rag_guide.txt   (more varied)
Hybrid (α=0.5)       → vectors + BM25 keyword match merged
```

MMR picks varied chunks instead of repeating one. Hybrid helps when the question contains exact names or terms. Set the default with `retrieval.strategy` in `config.yaml` (`similarity`, `mmr`, `hybrid` or `rerank`).

### Step 6. Prompt building

```bash
python 06_prompting/prompt_builder.py
```

Shows the same retrieved chunks formatted with four templates (numbered, XML, chain-of-thought, concise) and the estimated token count for each:

```
Template: CONCISE
  Chunks used: 3 | Est. tokens: 166
Template: COT
  Chunks used: 3 | Est. tokens: 270
```

The system prompt in `config.yaml` (`prompting.system_prompt`) tells the model to answer only from the context. This is the main guard against made-up answers.

### Step 7. LLM integration

```bash
python 07_llm_integration/llm_client.py
```

Detects Ollama, or Groq if `GROQ_API_KEY` is set, and prints a normal and a streamed answer. If neither is available it tells you how to set one up (Section 5).

```
✓ Ollama server detected
Using: ollama/llama3.2
╭────────────────────────────── ollama/llama3.2 ───────────────────────────────╮
│ RAG stands for Retrieval-Augmented Generation. It is useful because it       │
│ improves Large Language Model (LLM) responses by providing relevant ...      │
```

### Step 8. Full pipeline

See [Section 4](#4-end-to-end-pipeline).

### Step 9. Evaluation

```bash
python 09_evaluation/evaluator.py
```

Runs 5 built-in questions through the pipeline and scores the answers. Without RAGAS installed it uses simple built-in checks:

```
Manual Evaluation Scores
  Avg Retrieved Chunks       5.0
  Avg Answer Length Tokens   112.4
  Context Coverage Score     0.527    ← share of the reference answer's words found in the retrieved text
  No Answer Rate             0.2      ← share of questions the model refused to answer
Results saved → data/eval/eval_results.json
```

Your numbers will differ a little, because LLM answers vary.

**RAGAS (faithfulness, relevancy, precision, recall):** install with `pip install -r requirements.txt` (it pins `ragas==0.2.15` and the LangChain 0.3 versions it needs). RAGAS uses an LLM as the judge. This project reuses the provider from `config.yaml` (Groq, OpenAI or Ollama) and embeds locally, so **no OpenAI key is needed**. Real run with Groq `openai/gpt-oss-20b` on the 5 built-in questions (takes about 2 to 4 minutes on the Groq free tier):

```
RAGAS Scores:
 faithfulness       0.950   1.0 = answer fully grounded in context
 answer_relevancy   0.843   1.0 = answer perfectly answers question
 context_precision  0.733   1.0 = all retrieved chunks are relevant
 context_recall     0.867   1.0 = all necessary info was retrieved
```

Scores vary a little between runs because an LLM does the judging. If a metric shows `nan`, the judge call timed out or was rate limited. Run it again.

To use your own questions, add `{"question": ..., "ground_truth": ...}` entries to `BUILTIN_TEST_SET` in `evaluator.py`.

### Step 10. Advanced techniques

```bash
python 10_advanced/advanced_rag.py
```

Needs an LLM (Ollama or Groq) and an indexed corpus (run Section 4 first). It runs two techniques on one question.

- **HyDE:** the LLM writes a hypothetical answer, and that text is used for the search.
- **Query expansion:** the LLM rewrites the question several ways and merges the results.

```
Question: How does the transformer architecture handle long-range dependencies?

1. HyDE Retrieval
[HyDE] Hypothetical answer: The transformer architecture addresses long-range dependencies through its self-attention mechanism,...

2. Query Expansion
[QueryExp] Variants: ['How does the transformer architecture handle long-range dependencies?',
  'How does the transformer model effectively manage distant relationships between tokens?', ...]
```

### Step 11. Agentic RAG

```bash
python 11_agentic_rag/agentic_rag.py
```

Agentic RAG upgrades fixed retrieval into an autonomous reasoning loop:

- **Dynamic Query Router:** Routes requests between direct LLM answers, vector retrieval, and multi-hop execution.
- **ReAct Agent (Reasoning + Acting):** Step-by-step `Thought ➔ Action ➔ Observation` loop with tool calling (`document_search`, `list_available_sources`, `calculator`).
- **Corrective RAG (CRAG) & Self-Reflection:** Evaluates chunk relevance, reformulates queries on low relevance, and validates factual grounding against hallucinations.

```
Question: What is retrieval augmented generation, and which document in the index discusses it?

ReAct Agent Step-by-Step Trace:
  Step 1: Thought: I should check which documents discuss RAG.
          Action:  document_search(retrieval augmented generation)
          Obs:     [1] (retrieval_augmented_generation.pdf): Retrieval-Augmented Generation...
  Step 2: Thought: I have found the explanation and source document.
          Final Answer: RAG enables LLMs to incorporate external knowledge...
```

---

## 4. End-to-end pipeline

### Command line

1. Install and start Ollama and pull a model (see Section 5).
2. Download data and run:

```bash
python data/download_samples.py
python 08_full_pipeline/pipeline.py
```

It indexes everything in `data/raw/` (about 40 seconds for the 10 sample PDFs on a laptop CPU, giving **1,100 chunks**), then lets you ask questions:

```
[pipeline] ✓ Indexed 1100 chunks from 10 document(s)
Indexed documents: ['artificial_intelligence.pdf', 'deep_learning.pdf', ... 'vector_database.pdf']

You: What is retrieval-augmented generation?
╭───────────────────────── Answer  (0.7s | 5 chunks) ──────────────────────────╮
│ According to [1], retrieval-augmented generation (RAG) is a technique that   │
│ enables large language models (LLMs) to retrieve and incorporate new         │
│ information from external data sources.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
Sources:
  [1] retrieval_augmented_generation.pdf — ## Retrieval-augmented generation ...
  [2] large_language_model.pdf — ...
  [3] vector_database.pdf — ...
```

Asking something the documents do not cover gives a refusal instead of a made-up answer:

```
You: Who won the 2022 FIFA World Cup?
│ I don't have enough information in the provided documents to answer that.    │
```

Type `quit` to exit. This script indexes at every start. To keep and reuse an index, use the Python API (Section 2) or the web app.

### Web app

```bash
python 08_full_pipeline/app.py
# open http://localhost:7860
```

- **Upload & Index:** drop in PDFs and click *Index Documents*.
- **Ask Questions:** chat with your documents. The *Retrieved Sources* box shows which chunks were used.
- **Pipeline Explorer:** shows the live `config.yaml`.

Use the **Reset Index** button, or delete `data/chroma_store/`, to start over.

### As a library

```python
p = mod.RAGPipeline()                       # reads config.yaml (loading code in Section 2)
p.index_directory("data/raw")
r = p.query("What is a vector database?", top_k=5)
print(r.answer)
for s in r.sources:
    print(s["source"], s["text"][:80])
```

---

## 5. LLM setup and environment variables

Pick one provider and set it in `config.yaml` under `llm.provider`.

### Option 1: Ollama (local, free, no key)

```bash
brew install ollama                          # macOS. Linux: curl -fsSL https://ollama.com/install.sh | sh
                                             # Windows: https://ollama.com/download
ollama serve                                 # skip if the Ollama app is already running
ollama pull llama3.2                         # about 2 GB
```

Check it is running: `curl localhost:11434` prints `Ollama is running`. To use another model, run `ollama pull mistral` and set `llm.ollama.model: "mistral"`.

### Option 2: Groq (free API key, very fast)

1. Create a key at <https://console.groq.com> (no credit card needed).
2. Put it in `.env` (below).
3. In `config.yaml` set `llm.provider: groq`.
4. Needs the `openai` package (included in the requirements files; Groq uses the OpenAI client).
5. Model names change over time. The default is `openai/gpt-oss-20b`. List what your key can use with `python 07_llm_integration/llm_client.py --models` and set `llm.groq.model` in `config.yaml`.

### Option 3: OpenAI (paid)

1. Create a key at <https://platform.openai.com/api-keys>.
2. Put it in `.env`.
3. In `config.yaml` set `llm.provider: openai`.

### Environment variables

Copy the template and fill in only what you use:

```bash
cp .env.example .env
```

| Variable | Needed for | Where to get it |
|----------|-----------|-----------------|
| `GROQ_API_KEY` | `llm.provider: groq` | console.groq.com |
| `OPENAI_API_KEY` | `llm.provider: openai`, `embeddings.provider: openai` | platform.openai.com |
| `REDIS_PASSWORD` | Redis with a password (leave empty for the Docker setup) | your Redis config |

The `.env` file is loaded automatically when the code starts and is listed in `.gitignore`, so your keys are never committed. You can also export a variable in the shell instead: `export GROQ_API_KEY=gsk_...`.

The code reads the variable names above directly. The `api_key_env` lines in `config.yaml` are only labels.

If a key is missing, Step 7 tells you. You do not need any of these for the default setup (Ollama + local embeddings + Chroma).

---

## 6. Using Redis instead of ChromaDB

Redis needs the Redis Stack image, which includes the search module. Install the extra Python packages and start the container:

```bash
pip install redis redisvl
docker compose up -d redis                   # RedisInsight UI: http://localhost:8001
python 04_vector_db/redis/redis_store.py     # standalone demo
```

To run the whole pipeline on Redis, set this in `config.yaml`:

```yaml
vector_db:
  backend: redis
```

Then run Step 4 or Section 4 again. Other vector databases (Qdrant, pgvector) are in `docker-compose.yml` as optional extras.

Hybrid retrieval builds its keyword (BM25) index from the chunks indexed in the current session, or from ChromaDB if the index was saved earlier. With Redis, if you restart the program and query without re-indexing, the pipeline prints a notice and falls back to plain vector search. Re-index in the same session to keep hybrid search.

---

## 7. Configuration reference

Every component is swapped in [`config.yaml`](config.yaml). No code changes are needed.

```yaml
chunking:
  strategy: recursive        # fixed | sentence | recursive | semantic
  chunk_size: 512
  chunk_overlap: 64

embeddings:
  provider: local            # local | openai | ollama
  local:
    model: "all-MiniLM-L6-v2"   # or BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5

vector_db:
  backend: chroma            # chroma | redis

retrieval:
  strategy: hybrid           # similarity | mmr | hybrid | rerank
  top_k: 5

llm:
  provider: ollama           # ollama | groq | openai
```

If you change the embedding model, **delete `data/chroma_store/` and re-index**, because vectors from different models cannot be compared. For Redis, also set `vector_db.redis.vector_dim` to the new model's dimension.

---

## 8. Project structure

```
.
├── 01_document_loading/loader.py       Docling: PDF/HTML/DOCX → Markdown
├── 02_chunking/chunker.py              fixed, sentence, recursive, semantic
├── 03_embeddings/embedder.py           local, OpenAI, Ollama + PCA plot
├── 04_vector_db/
│   ├── chroma/chroma_store.py          ChromaDB (local files)
│   ├── redis/redis_store.py            Redis + RedisVL
│   └── compare_dbs.py                  benchmark
├── 05_retrieval/retriever.py           similarity, MMR, hybrid (BM25 + RRF), rerank
├── 06_prompting/prompt_builder.py      4 prompt templates
├── 07_llm_integration/llm_client.py    Ollama, Groq, OpenAI
├── 08_full_pipeline/
│   ├── pipeline.py                     RAGPipeline class + CLI
│   └── app.py                          Gradio web app
├── 09_evaluation/evaluator.py          manual metrics + RAGAS
├── 10_advanced/advanced_rag.py         HyDE, query expansion, compression
├── 11_agentic_rag/agentic_rag.py       Router, ReAct agent, Corrective RAG (CRAG)
├── data/
│   ├── download_samples.py             Wikipedia → PDFs
│   ├── raw/                            input PDFs (put yours here)
│   ├── processed/                      Docling Markdown, PCA plot
│   ├── chroma_store/                   ChromaDB files (created on first index)
│   └── eval/                           evaluation results
├── src/                                config loader and shared data classes
├── config.yaml                         master configuration
├── docker-compose.yml                  Redis Stack (+ optional Qdrant, pgvector)
├── requirements-minimal.txt            local, free setup
└── requirements.txt                    everything
```

---

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No PDFs found in data/raw/` | Run `python data/download_samples.py` or copy PDFs into `data/raw/`. |
| `Failed to connect to Ollama` | Start it with `ollama serve`, then `ollama pull llama3.2`. Or switch to Groq. |
| `No LLM available` in Step 7 | Do one of the options in Section 5. If you set `GROQ_API_KEY`, also check: `.env` is in the project root, the line is `GROQ_API_KEY=gsk_...` (no spaces or quotes), and `pip install openai` has been run (Groq uses the OpenAI client). |
| `model_not_found` / 404 from Groq | Groq retires models. Run `python 07_llm_integration/llm_client.py --models` and set `llm.groq.model` in `config.yaml` to one from the list. |
| `KeyError: 'OPENAI_API_KEY'` or a missing `GROQ_API_KEY` message | Add the key to `.env` (Section 5), run from the project root, and make sure `llm.provider` matches the key. |
| First run is very slow | Models are downloading (Section 1). Later runs are fast. |
| `Hybrid needs indexed text for BM25` | Nothing is indexed yet. Index documents first (Section 4). |
| Answers ignore a new PDF | Check the file is in `data/raw/` or uploaded in the app, then re-run indexing. |
| Strange or empty results after changing the embedding model | Delete `data/chroma_store/` and re-index (Section 7). |
| `unknown command 'FT.INFO'` (Redis) | Your Redis has no search module. Use the Docker image from this repo (`redis/redis-stack`). |
| `port is already allocated` on 6379 | Another Redis is using the port. Stop it, or change the host port in `docker-compose.yml` and `vector_db.redis.port` in `config.yaml`. |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements-minimal.txt`. |

---

## 10. Research and related work

Papers:

- [Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers](https://ieeexplore.ieee.org/abstract/document/10707868)
- [MetaGen Blended RAG: Unlocking Zero-Shot Precision for Specialized Domain Question-Answering](https://ieeexplore.ieee.org/abstract/document/11486409)

Related open-source projects:

- [SuperKnowa](https://github.com/ibm-self-serve-assets/SuperKnowa): build enterprise RAG pipelines from plug-in components
- [Blended RAG](https://github.com/ibm-self-serve-assets/Blended-RAG): semantic search with hybrid query-based retrievers
- [MetaGen Blended RAG](https://github.com/ibm-self-serve-assets/MetaGen-Blended-RAG): code for the MetaGen Blended RAG paper

Author: [Abhilasha Mangal](https://github.com/Abhilasha-Mangal)

---

⭐ **Enjoying RAG From Scratch?** Please [star the repository](https://github.com/Abhilasha-Mangal/RAG-From-Scratch) on GitHub to show your support!

## License

MIT
