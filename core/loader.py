"""
loader.py
---------
This module is responsible for turning uploaded files into plain text.

In a RAG system, the later steps do not want to care whether the original
file was Word, Excel, TXT, or Markdown. They only need a consistent shape:

{
    "content": "the extracted text",
    "metadata": {
        "filename": "report.docx",
        "filetype": "word",
        "sheets": ["Sheet1"],
    }
}

The loader is the first step of the pipeline:
file -> loader -> chunker -> embedder -> retriever -> generator
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TypedDict

from docx import Document
from markdown import markdown
from openpyxl import load_workbook


class LoadedDocument(TypedDict):
    """The unified structure returned by every loader function."""

    content: str
    metadata: dict[str, Any]


class MarkdownTextExtractor(HTMLParser):
    """
    A tiny HTML-to-text helper used after Markdown is converted to HTML.

    The markdown package turns Markdown into HTML. For RAG, we usually want
    readable plain text, so this parser collects the visible text parts.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


SUPPORTED_EXTENSIONS = {
    ".docx": "word",
    ".xlsx": "excel",
    ".txt": "text",
    ".md": "markdown",
}


def _base_metadata(file_path: Path, filetype: str) -> dict[str, Any]:
    """
    Build metadata that every file type should have.

    Metadata is useful later because chunks can remember where they came from.
    For example, the UI can show the original filename with each answer source.
    """

    return {
        "filename": file_path.name,
        "filetype": filetype,
    }


def _clean_lines(lines: list[str]) -> str:
    """
    Remove empty edges and join text lines into one clean string.

    We keep line breaks because they preserve document structure better than
    flattening everything into one long sentence.
    """

    cleaned = [line.strip() for line in lines if line and line.strip()]
    return "\n".join(cleaned)


def load_word(file_path: str | Path) -> LoadedDocument:
    """
    Load a Word .docx file.

    This extracts:
    - normal paragraphs
    - table cells

    Images are ignored for now because they require OCR or multimodal parsing.
    """

    path = Path(file_path)
    document = Document(path)

    lines: list[str] = []

    for paragraph in document.paragraphs:
        lines.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            cell_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cell_texts:
                lines.append(" | ".join(cell_texts))

    return {
        "content": _clean_lines(lines),
        "metadata": _base_metadata(path, "word"),
    }


def load_excel(file_path: str | Path) -> LoadedDocument:
    """
    Load an Excel .xlsx file.

    Each sheet is converted into readable text. Rows are joined by tabs so the
    table shape is still understandable when it becomes plain text.
    """

    path = Path(file_path)
    workbook = load_workbook(path, data_only=True)

    lines: list[str] = []
    sheet_names = workbook.sheetnames

    for sheet_name in sheet_names:
        sheet = workbook[sheet_name]
        lines.append(f"[Sheet: {sheet_name}]")

        for row in sheet.iter_rows(values_only=True):
            values = ["" if value is None else str(value) for value in row]
            row_text = "\t".join(values).strip()
            if row_text:
                lines.append(row_text)

    metadata = _base_metadata(path, "excel")
    metadata["sheets"] = sheet_names

    return {
        "content": _clean_lines(lines),
        "metadata": metadata,
    }


def load_text(file_path: str | Path) -> LoadedDocument:
    """Load a plain .txt file as UTF-8 text."""

    path = Path(file_path)

    return {
        "content": path.read_text(encoding="utf-8").strip(),
        "metadata": _base_metadata(path, "text"),
    }


def load_markdown(file_path: str | Path) -> LoadedDocument:
    """
    Load a Markdown .md file and convert it to readable plain text.

    We keep the final output as text because the embedding model should focus
    on meaning, not Markdown syntax such as #, **, or table pipes.
    """

    path = Path(file_path)
    raw_markdown = path.read_text(encoding="utf-8")
    html = markdown(raw_markdown)

    extractor = MarkdownTextExtractor()
    extractor.feed(html)

    return {
        "content": extractor.get_text().strip(),
        "metadata": _base_metadata(path, "markdown"),
    }


def load_document(file_path: str | Path) -> LoadedDocument:
    """
    Load one supported document and return the unified structure.

    This is the main function other modules should call. It checks the file
    extension and delegates the real parsing work to the matching loader.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if suffix == ".docx":
        return load_word(path)
    if suffix == ".xlsx":
        return load_excel(path)
    if suffix == ".txt":
        return load_text(path)
    if suffix == ".md":
        return load_markdown(path)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file type: {suffix}. Supported types: {supported}")


def load_documents(file_paths: list[str | Path]) -> list[LoadedDocument]:
    """
    Load multiple documents.

    Keeping this helper here makes later code simpler in Streamlit:
    uploaded files can be collected into a list and loaded in one call.
    """

    return [load_document(file_path) for file_path in file_paths]
