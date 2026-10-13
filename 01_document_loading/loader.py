"""
Module 1 — Document Loading with IBM Docling
=============================================
Docling is an IBM open-source library that extracts richly structured content
from PDFs (including tables, headers, page numbers) and many other formats.

Supported input formats:
    • PDF  (digital + scanned via OCR)
    • DOCX / PPTX / XLSX
    • HTML / Markdown
    • Images (PNG, JPEG)

What this module teaches:
    1. What a "Document" object is (content + metadata)
    2. How Docling preserves document structure (tables, headings, lists)
    3. Batch loading an entire folder of PDFs
    4. Exporting to Markdown for downstream chunking
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

# Add repo root to path for src imports
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import Document  # noqa: E402

# ── Docling imports ───────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

class DoclingLoader:
    """
    Load one or many documents using IBM Docling.

    Parameters
    ----------
    ocr        : Enable OCR for scanned/image PDFs. Slower but handles scans.
    table_mode : "fast" (default) or "accurate" for table extraction.
    """

    def __init__(self, ocr: bool = False, table_mode: str = "fast"):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = (
            table_mode == "accurate"
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> Document:
        """Load a single file and return a Document."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        print(f"[docling] Converting: {path.name} ...")
        result = self._converter.convert(str(path))

        # Export as Markdown — preserves headings, tables, lists
        content_md = result.document.export_to_markdown()

        # num_pages is a method in recent Docling versions, an int in older ones
        num_pages = getattr(result.document, "num_pages", None)
        if callable(num_pages):
            num_pages = num_pages()

        metadata = {
            "source": str(path),
            "filename": path.name,
            "format": path.suffix.lower().lstrip("."),
            "num_pages": num_pages,
            "title": self._extract_title(result.document),
        }

        return Document(content=content_md, source=str(path), metadata=metadata)

    def load_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.pdf",
    ) -> list[Document]:
        """Load all matching files in a directory (recursive by default)."""
        directory = Path(directory)
        files = list(directory.glob(glob))
        if not files:
            print(f"[warn] No files matched '{glob}' in {directory}")
            return []

        print(f"[docling] Found {len(files)} file(s) in {directory}")
        docs = []
        for fp in files:
            try:
                docs.append(self.load_file(fp))
            except Exception as exc:
                print(f"[error] Failed to load {fp.name}: {exc}")
        return docs

    def load_url(self, url: str) -> Document:
        """Load a document directly from a URL (HTML page or hosted PDF)."""
        print(f"[docling] Fetching URL: {url} ...")
        result = self._converter.convert(url)
        content_md = result.document.export_to_markdown()
        metadata = {
            "source": url,
            "filename": url.split("/")[-1],
            "format": "url",
            "title": self._extract_title(result.document),
        }
        return Document(content=content_md, source=url, metadata=metadata)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(doc) -> str:
        """Try to extract a title from docling document metadata."""
        try:
            return doc.description.title or ""
        except AttributeError:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo / notebook entry point
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Quick demo — load the sample PDFs in data/raw/ and print summaries.
    Run: python 01_document_loading/loader.py
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold blue]Module 1 — Document Loading with Docling")

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    loader = DoclingLoader(ocr=False, table_mode="fast")

    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        console.print(
            "[yellow]No PDFs found in data/raw/. "
            "Run: python data/download_samples.py first.[/yellow]"
        )
        return

    docs = []
    for pdf in pdfs:
        doc = loader.load_file(pdf)
        docs.append(doc)

    # Pretty table summary
    table = Table(title="Loaded Documents", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Pages", justify="right")
    table.add_column("Characters", justify="right")
    table.add_column("Title")

    for d in docs:
        table.add_row(
            d.metadata["filename"],
            str(d.metadata.get("num_pages", "?")),
            f"{len(d.content):,}",
            d.metadata.get("title", "—")[:60],
        )
    console.print(table)

    # Save Markdown exports
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(exist_ok=True)
    for d in docs:
        stem = Path(d.metadata["filename"]).stem
        out_path = out_dir / f"{stem}.md"
        out_path.write_text(d.content, encoding="utf-8")
        console.print(f"[green]Saved:[/green] {out_path}")

    console.print(f"\n[bold green]✓ Loaded {len(docs)} documents.[/bold green]")
    return docs


if __name__ == "__main__":
    demo()
