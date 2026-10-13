"""
Sample Data Downloader
=======================
Downloads a curated set of Wikipedia articles and converts them to PDFs.
These cover topics directly relevant to this project:

    1. Artificial Intelligence          → ai.pdf
    2. Machine Learning                 → machine_learning.pdf
    3. Natural Language Processing      → nlp.pdf
    4. Transformer (machine learning)   → transformers.pdf
    5. Information Retrieval            → information_retrieval.pdf
    6. Vector Database                  → vector_database.pdf
    7. Large Language Model             → llm.pdf
    8. Deep Learning                    → deep_learning.pdf

Also bundles a mini "textbook" PDF compiled from these topics.

Requirements: pip install wikipedia-api reportlab requests
Run: python data/download_samples.py
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

# ── Output directory ─────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Topics to download ────────────────────────────────────────────────────────
WIKIPEDIA_TOPICS = [
    ("Artificial intelligence",               "artificial_intelligence.pdf"),
    ("Machine learning",                       "machine_learning.pdf"),
    ("Natural language processing",            "natural_language_processing.pdf"),
    ("Transformer (deep learning architecture)", "transformers.pdf"),
    ("Information retrieval",                  "information_retrieval.pdf"),
    ("Vector database",                        "vector_database.pdf"),
    ("Large language model",                   "large_language_model.pdf"),
    ("Deep learning",                          "deep_learning.pdf"),
    ("Retrieval-augmented generation",         "retrieval_augmented_generation.pdf"),
    ("Semantic search",                        "semantic_search.pdf"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia → text
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wikipedia_text(title: str) -> str | None:
    """Fetch the full text of a Wikipedia article."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="RAG-From-Scratch/1.0 (educational project)"
        )
        page = wiki.page(title)
        if not page.exists():
            print(f"  [warn] Page not found: {title!r}")
            return None
        return page.text
    except ImportError:
        # Fallback: use requests + Wikipedia API directly
        return fetch_wikipedia_api(title)


def fetch_wikipedia_api(title: str) -> str | None:
    """Fallback Wikipedia fetch using the REST API (no extra library needed)."""
    import requests
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exintro": False,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


# ─────────────────────────────────────────────────────────────────────────────
# Text → PDF using ReportLab
# ─────────────────────────────────────────────────────────────────────────────

def text_to_pdf(text: str, title: str, output_path: Path) -> None:
    """Convert a plain text string to a well-formatted PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

    except ImportError:
        print("  [warn] reportlab not installed. Saving as .txt instead.")
        output_path.with_suffix(".txt").write_text(text, encoding="utf-8")
        return

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1f2328"),
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#3b82d4"),
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    source_style = ParagraphStyle(
        "SourceStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceAfter=12,
    )

    story = []

    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("Source: Wikipedia (en.wikipedia.org) — Educational Use", source_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 12))

    # Body: parse sections
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        # Section headings (all-caps lines or lines ending with ==)
        if re.match(r"^={2,}.*={2,}$", line):
            heading = re.sub(r"=+", "", line).strip()
            if heading:
                story.append(Paragraph(heading, h1_style))
        elif len(line) < 80 and line.isupper():
            story.append(Paragraph(line.title(), h1_style))
        else:
            # Escape XML chars for ReportLab
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))

    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def download_all():
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    console = Console()
    console.rule("[bold blue]Downloading Sample Data")
    console.print(f"Output directory: [cyan]{RAW_DIR}[/cyan]\n")

    success = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(WIKIPEDIA_TOPICS))

        for wiki_title, filename in WIKIPEDIA_TOPICS:
            out_path = RAW_DIR / filename

            if out_path.exists():
                console.print(f"  [dim]Skipped (exists):[/dim] {filename}")
                skipped += 1
                progress.advance(task)
                continue

            progress.update(task, description=f"Fetching: {wiki_title}")
            text = fetch_wikipedia_text(wiki_title)

            if not text:
                console.print(f"  [red]Failed:[/red] {wiki_title}")
                progress.advance(task)
                continue

            text_to_pdf(text, title=wiki_title, output_path=out_path)
            size_kb = out_path.stat().st_size / 1024
            console.print(f"  [green]✓[/green] {filename:45s} {size_kb:.0f} KB")
            success += 1
            progress.advance(task)

    console.rule()
    console.print(f"[bold green]Done: {success} downloaded, {skipped} skipped.[/bold green]")
    console.print(f"Files are in: [cyan]{RAW_DIR}[/cyan]")
    console.print("\n[yellow]Next step:[/yellow] python 08_full_pipeline/app.py")


def create_combined_textbook():
    """
    Combine all downloaded articles into a single 'RAG Textbook' PDF.
    Good for indexing a single multi-topic document.
    """
    from rich.console import Console
    console = Console()

    txt_files = list(RAW_DIR.glob("*.txt"))
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    all_content = []
    for pdf in pdf_files:
        # We can't easily read PDFs here, so we use the text fallbacks
        txt = pdf.with_suffix(".txt")
        if txt.exists():
            all_content.append(f"\n\n{'='*60}\n{pdf.stem.replace('_', ' ').title()}\n{'='*60}\n")
            all_content.append(txt.read_text(encoding="utf-8"))

    if all_content:
        combined_text = "\n".join(all_content)
        out = RAW_DIR / "rag_textbook_combined.pdf"
        text_to_pdf(combined_text, "RAG From Scratch — Reference Textbook", out)
        console.print(f"[green]✓ Combined textbook saved → {out}[/green]")


if __name__ == "__main__":
    # Install optional dependencies if missing
    for pkg, import_name in [
        ("reportlab", "reportlab"),
        ("wikipedia-api", "wikipediaapi"),
        ("rich", "rich"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            import subprocess
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

    download_all()
