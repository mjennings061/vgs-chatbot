"""Document parsing and chunking utilities."""

from __future__ import annotations

import io
import logging
import re
from typing import List, Tuple

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)


def read_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract plain text from each page of a PDF.

    Args:
        file_bytes: Raw bytes representing the PDF file.

    Returns:
        list[tuple[int, str]]: Sequence of page numbers and their extracted text.
    """
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((index, text))
    logger.debug("Extracted text from %s PDF pages.", len(pages))
    return pages


def read_docx(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from a Word document, approximating pages.

    Args:
        file_bytes: Raw bytes representing the Word document.

    Returns:
        list[tuple[int, str]]: Approximate page number and associated text.
    """
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    if not paragraphs:
        return [(1, "")]
    # Word does not expose pagination without layout; treat blocks of ~30 paragraphs as a page.
    chunk_size = 30
    pages: List[Tuple[int, str]] = []
    for index in range(0, len(paragraphs), chunk_size):
        page_no = len(pages) + 1
        page_text = "\n".join(paragraphs[index : index + chunk_size])
        pages.append((page_no, page_text))
    logger.debug(
        "Extracted %s DOCX paragraphs across %s pseudo-pages.",
        len(paragraphs),
        len(pages),
    )
    return pages


def detect_sections(text: str) -> List[Tuple[str, str]]:
    """Split page text into coarse sections using heading heuristics.

    Args:
        text: Page-level text to analyse.

    Returns:
        list[tuple[str, str]]: Detected section titles and their bodies.
    """
    headings: List[Tuple[str, List[str]]] = []
    current_title: str | None = None
    current_body: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_heading(line):
            if current_title and current_body:
                headings.append((current_title, current_body[:]))
            current_title = line
            current_body = []
        else:
            current_body.append(line)

    if current_title and current_body:
        headings.append((current_title, current_body))

    if not headings and text.strip():
        return [("General", text.strip())]

    sections: List[Tuple[str, str]] = []
    for title, lines in headings:
        body = "\n".join(lines).strip()
        if body:
            sections.append((title, body))
    logger.debug("Detected %s sections.", len(sections))
    return sections


def chunk_text(text: str, target_chars: int = 900, overlap: int = 120) -> List[str]:
    """Chunk text into roughly target-sized pieces with soft paragraph boundaries.

    Args:
        text: Text to segment.
        target_chars: Preferred chunk size in characters.
        overlap: Overlap length applied when splitting long paragraphs.

    Returns:
        list[str]: Ordered list of chunk strings.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > target_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_paragraph(paragraph, target_chars, overlap))
            continue

        if len(buffer) + len(paragraph) + 1 <= target_chars:
            buffer = f"{buffer}\n{paragraph}".strip()
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = paragraph

    if buffer:
        chunks.append(buffer.strip())

    logger.debug("Chunked text into %s segments.", len(chunks))
    return chunks


def _looks_like_heading(line: str) -> bool:
    """Return True if a line resembles a heading.

    Args:
        line: Line of text to evaluate.

    Returns:
        bool: True when the line meets heading heuristics.
    """
    if len(line) < 4:
        return False
    if re.match(r"^\d+(\.\d+)*\s+[A-Z].*", line):
        return True
    alpha_chars = [ch for ch in line if ch.isalpha()]
    if not alpha_chars:
        return False
    uppercase = sum(1 for ch in alpha_chars if ch.isupper())
    return uppercase / len(alpha_chars) > 0.8


def _split_long_paragraph(text: str, target_chars: int, overlap: int) -> List[str]:
    """Split a long paragraph into overlapping character windows.

    Args:
        text: Paragraph text to segment.
        target_chars: Preferred window size in characters.
        overlap: Overlap between consecutive windows.

    Returns:
        list[str]: Sequence of overlapping segments.
    """
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    if len(cleaned) <= target_chars:
        return [cleaned]
    step = max(1, target_chars - overlap)
    segments: List[str] = []
    for start in range(0, len(cleaned), step):
        end = min(len(cleaned), start + target_chars)
        segment = cleaned[start:end].strip()
        if segment:
            segments.append(segment)
        if end == len(cleaned):
            break
    logger.debug(
        "Split long paragraph into %s overlapping segments.",
        len(segments),
    )
    return segments
