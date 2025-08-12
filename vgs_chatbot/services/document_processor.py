"""Document processor service implementation with RAG pipeline."""

import uuid
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import chromadb
import openpyxl
from docx import Document as DocxDocument
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from vgs_chatbot.interfaces.document_processor_interface import (
    DocumentProcessorInterface,
)
from vgs_chatbot.models.document import Document, ProcessedDocument


class RAGDocumentProcessor(DocumentProcessorInterface):
    """RAG-based document processor implementation."""

    def __init__(
        self,
        embedding_model: str = "multi-qa-MiniLM-L6-cos-v1",
        persist_directory: str | None = None,
    ) -> None:
        """Initialize document processor.

        Args:
            embedding_model: Name of sentence transformer model for embeddings
            persist_directory: Optional path for persistent Chroma storage
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        if persist_directory:
            # Use persistent client to retain embeddings across Streamlit reruns
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()

        # Try to get existing collection; create if not present
        try:
            self.collection = self.chroma_client.get_collection(name="documents")
        except Exception:
            self.collection = self.chroma_client.create_collection(name="documents")
        self.chunk_size = 1000
        self.chunk_overlap = 200

    async def process_documents(
        self, documents: list[Document]
    ) -> list[ProcessedDocument]:
        """Process documents into searchable format.

        Args:
            documents: List of raw documents

        Returns:
            List of processed documents
        """
        processed_docs = []

        for document in documents:
            try:
                processed_doc = await self._process_single_document(document)
                processed_docs.append(processed_doc)
            except Exception as e:
                # Log error but continue with other documents
                print(f"Error processing document {document.name}: {e}")
                continue

        return processed_docs

    async def _process_single_document(self, document: Document) -> ProcessedDocument:
        """Process a single document.

        Args:
            document: Document to process

        Returns:
            Processed document
        """
        # Extract text content based on file type
        content = await self._extract_text_content(document)

        # Split into chunks
        chunks = self._split_text_into_chunks(content)

        # Generate embeddings
        try:
            raw_embeddings: Any = self.embedding_model.encode(chunks, convert_to_numpy=True)  # type: ignore[no-untyped-call]
            if hasattr(raw_embeddings, "tolist"):
                embeddings = raw_embeddings.tolist()  # type: ignore[assignment]
            else:  # Fallback (already python list)
                embeddings = raw_embeddings  # type: ignore[assignment]
            print(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            embeddings = None

        # Extract metadata for better context
        metadata = self._extract_metadata(content, document.name)

        processed_doc = ProcessedDocument(
            id=str(uuid.uuid4()),
            original_document=document,
            content=content,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
            processed_at=datetime.now(UTC),
        )

        return processed_doc

    async def _extract_text_content(self, document: Document) -> str:
        """Extract text content from document based on file type.

        Args:
            document: Document to extract text from

        Returns:
            Extracted text content
        """
        from pathlib import Path

        try:
            file_path = Path(document.file_path)
            if not file_path.exists():
                return f"Document: {document.name}\nError: File not found at {document.file_path}"

            # Read file content as bytes
            file_content = file_path.read_bytes()

            if (
                document.file_type == "application/pdf"
                or file_path.suffix.lower() == ".pdf"
            ):
                return await self._extract_pdf_text(file_content)
            elif document.file_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ] or file_path.suffix.lower() in [".docx", ".doc"]:
                return await self._extract_docx_text(file_content)
            elif document.file_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ] or file_path.suffix.lower() in [".xlsx", ".xls"]:
                return await self._extract_excel_text(file_content)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                # Handle text files directly
                return file_content.decode("utf-8", errors="replace")
            else:
                # Try to decode as text
                try:
                    return file_content.decode("utf-8", errors="replace")
                except Exception:
                    return f"Document: {document.name}\nBinary file content cannot be extracted as text"

        except Exception as e:
            return f"Document: {document.name}\nError extracting content: {str(e)}"

    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content with improved structure detection.

        Args:
            content: PDF file content as bytes

        Returns:
            Extracted text with page markers and structure
        """
        if not content:
            return "PDF content placeholder"

        try:
            reader = PdfReader(BytesIO(content))
            text = ""
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Add page marker with more context
                        text += f"\n--- PAGE {page_num} of {total_pages} ---\n"

                        # Try to preserve more structure
                        enhanced_text = self._enhance_pdf_text_structure(page_text)
                        text += enhanced_text + "\n"

                except Exception as e:
                    # Log individual page errors but continue processing
                    print(f"Warning: Error extracting page {page_num}: {e}")
                    text += f"\n--- PAGE {page_num} (EXTRACTION ERROR) ---\n"
                    continue

            # Clean up text - remove excessive whitespace but preserve structure
            text = self._clean_extracted_text(text)
            return text if text else "No text content found in PDF"

        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

    def _enhance_pdf_text_structure(self, page_text: str) -> str:
        """Enhance PDF text structure by identifying headings, lists, and paragraphs.

        Args:
            page_text: Raw text from PDF page

        Returns:
            Enhanced text with better structure
        """
        lines = page_text.split("\n")
        enhanced_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify potential headings (all caps, short lines, etc.)
            if len(line) < 80 and (
                line.isupper()
                or any(
                    word in line.upper()
                    for word in ["CHAPTER", "SECTION", "PART", "APPENDIX"]
                )
                or line.endswith(":")
            ):
                enhanced_lines.append(f"\n=== {line} ===\n")

            # Identify bullet points and lists
            elif any(line.startswith(marker) for marker in ["•", "-", "*", "○"]):
                enhanced_lines.append(f"  {line}")

            # Identify numbered lists
            elif len(line) > 3 and line[:3].replace(".", "").replace(")", "").isdigit():
                enhanced_lines.append(f"  {line}")

            # Regular text
            else:
                enhanced_lines.append(line)

        return "\n".join(enhanced_lines)

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text while preserving important structure.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text with preserved structure
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and not cleaned_lines[-1] == "":
                # Preserve paragraph breaks
                cleaned_lines.append("")

        # Remove excessive empty lines (more than 2 consecutive)
        result_lines = []
        empty_count = 0

        for line in cleaned_lines:
            if line == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    async def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content with structure preservation.

        Args:
            content: DOCX file content as bytes

        Returns:
            Extracted text with structure and tables
        """
        if not content:
            return "DOCX content placeholder"

        try:
            doc = DocxDocument(BytesIO(content))
            text = ""

            # Extract paragraphs with style information
            for paragraph in doc.paragraphs:
                para_text = paragraph.text.strip()
                if para_text:
                    # Check if paragraph is a heading
                    style_name = (
                        getattr(getattr(paragraph, "style", None), "name", "") or ""
                    )
                    if style_name.startswith("Heading"):
                        text += f"\n## {para_text}\n"
                    else:
                        text += para_text + "\n"

            # Extract tables
            for table_num, table in enumerate(doc.tables, 1):
                text += f"\n--- TABLE {table_num} ---\n"
                for row in table.rows:
                    row_cells = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_cells.append(cell_text)
                    if row_cells:
                        text += " | ".join(row_cells) + "\n"
                text += "\n"

            return text if text.strip() else "No text content found in DOCX"

        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}"

    async def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel content with improved structure handling.

        Args:
            content: Excel file content as bytes

        Returns:
            Extracted text with sheet and row structure
        """
        if not content:
            return "Excel content placeholder"

        try:
            workbook = openpyxl.load_workbook(BytesIO(content), data_only=True)
            text = ""

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\n--- SHEET: {sheet_name} ---\n"

                # Get data range to avoid empty rows
                if sheet.max_row > 0:
                    # Extract header row if it exists
                    first_row = list(
                        sheet.iter_rows(min_row=1, max_row=1, values_only=True)
                    )[0]
                    if first_row and any(cell is not None for cell in first_row):
                        headers = [
                            str(cell) if cell is not None else "" for cell in first_row
                        ]
                        text += "Headers: " + " | ".join(headers) + "\n"
                        start_row = 2
                    else:
                        start_row = 1

                    # Extract data rows
                    row_count = 0
                    for row in sheet.iter_rows(min_row=start_row, values_only=True):
                        if row and any(cell is not None for cell in row):
                            row_text = " | ".join(
                                str(cell) if cell is not None else "" for cell in row
                            )
                            if row_text.strip():
                                text += row_text + "\n"
                                row_count += 1
                                # Limit rows to prevent excessive content
                                if row_count > 100:
                                    text += "[... truncated after 100 rows ...]\n"
                                    break

            return text if text.strip() else "No data found in Excel file"

        except Exception as e:
            return f"Error extracting Excel text: {str(e)}"

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks with enhanced table preservation for aviation documents.

        Args:
            text: Text to split

        Returns:
            List of text chunks with better preservation of regulatory tables
        """
        # Special handling for weather limitations table - preserve as single chunk
        weather_table_chunk = self._extract_weather_limitations_table(text)
        if weather_table_chunk:
            # Remove the table from the original text to avoid duplication
            text_without_table = text.replace(
                weather_table_chunk, "\n[WEATHER_TABLE_REMOVED]\n"
            )

            # Process the rest of the document normally
            other_chunks = self._split_text_standard(text_without_table)

            # Insert the weather table as the first chunk (highest priority)
            return [weather_table_chunk] + other_chunks

        # Standard processing if no weather table found
        return self._split_text_standard(text)

    def _extract_weather_limitations_table(self, text: str) -> str | None:
        """Extract the complete weather limitations table as a single chunk."""
        lines = text.split("\n")

        # Find weather table boundaries
        start_idx = -1
        end_idx = -1

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Look for table start
            if "weather limitations table" in line_lower:
                start_idx = i

                # Look for the end - find where wind/crosswind data ends
                for j in range(i + 1, min(i + 25, len(lines))):
                    next_line = lines[j].strip()

                    # Table continues while we have wind/crosswind data
                    if (
                        (
                            "crosswind" in next_line.lower()
                            and "kts" in next_line.lower()
                        )
                        or ("maximum wind speed" in next_line.lower())
                        or (
                            (
                                "20kts" in next_line
                                or "25kts" in next_line
                                or "30kts" in next_line
                            )
                            and len(next_line.split()) < 20
                        )
                    ):  # Short lines with wind data
                        end_idx = j
                    # Stop at major section break
                    elif (
                        (
                            next_line.startswith("===")
                            and "weather" not in next_line.lower()
                            and "cfs" not in next_line.lower()
                        )
                        or next_line.startswith("DHO ")
                        or "minimum cloud base" in next_line.lower()
                    ):
                        break

                break

        # Extract the complete table if found
        if start_idx >= 0 and end_idx > start_idx:
            # Include some context before the table
            context_start = max(0, start_idx - 2)
            table_lines = lines[context_start : end_idx + 1]
            return "\n".join(table_lines).strip()

        return None

    def _split_text_standard(self, text: str) -> list[str]:
        """Standard text splitting logic."""
        chunks = []

        # First, try to identify important sections like page markers
        sections = []
        current_section = ""

        for line in text.split("\n"):
            if line.startswith("--- PAGE"):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())

        # If no page sections found, split by paragraphs
        if len(sections) <= 1:
            sections = [s.strip() for s in text.split("\n\n") if s.strip()]

        # If still no good splits, use the whole text
        if len(sections) <= 1:
            sections = [text.strip()]

        current_chunk = ""
        for section in sections:
            if not section or section == "[WEATHER_TABLE_REMOVED]":
                continue

            # Check if adding this section exceeds our chunk size
            test_chunk = (current_chunk + "\n\n" + section).strip()

            if len(test_chunk.split()) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Handle large sections
                if len(section.split()) > self.chunk_size:
                    # Split large sections more carefully, trying to keep related content together
                    lines = section.split("\n")
                    temp_chunk = ""

                    for line in lines:
                        test_line_chunk = (temp_chunk + "\n" + line).strip()
                        if len(test_line_chunk.split()) <= self.chunk_size:
                            temp_chunk = test_line_chunk
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = line

                    # Handle remaining content
                    if temp_chunk.strip():
                        current_chunk = temp_chunk.strip()
                    else:
                        current_chunk = ""
                else:
                    current_chunk = section

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure we have content
        if not chunks and text.strip():
            # Final fallback: simple word-based chunking
            words = text.split()
            for i in range(0, len(words), self.chunk_size):
                chunk_words = words[i : i + self.chunk_size]
                chunks.append(" ".join(chunk_words))

        return chunks if chunks else []

    def _extract_metadata(self, content: str, document_name: str) -> dict[str, Any]:
        """Extract metadata from document content.

        Args:
            content: Document text content
            document_name: Name of the document

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "document_name": document_name,
            "total_pages": 0,
            "sections": [],
            "key_terms": [],
        }

        # Count pages from page markers
        page_markers = content.count("--- PAGE")
        if page_markers > 0:
            metadata["total_pages"] = page_markers

        # Extract section headings (improved heuristic)
        lines = content.split("\n")
        sections = []
        for line in lines:
            line = line.strip()
            # Look for enhanced structure headings
            if line.startswith("===") and line.endswith("==="):
                # Extract content between === markers
                section_title = line.replace("===", "").strip()
                if section_title:
                    sections.append(section_title)
            # Look for traditional section patterns
            elif line.isupper() and len(line) > 5 and len(line) < 100:
                sections.append(line)
            # Look for markdown-style headings
            elif line.startswith("#") and len(line) > 5:
                sections.append(line)
            # Look for numbered sections and chapters
            elif any(
                word in line.upper()
                for word in ["CHAPTER", "SECTION", "PART", "APPENDIX"]
            ):
                sections.append(line)
            # Look for topic-specific sections
            elif any(
                word in line.upper()
                for word in [
                    "WEATHER",
                    "OPERATIONS",
                    "LIMITS",
                    "PROCEDURES",
                    "EMERGENCY",
                    "SAFETY",
                ]
            ):
                if len(line) < 100:  # Avoid capturing long paragraphs
                    sections.append(line)

        metadata["sections"] = sections[:10]  # Limit to first 10 sections

        # Extract aviation-specific key terms that might be relevant
        aviation_terms = [
            "wind limit",
            "wind speed",
            "crosswind",
            "headwind",
            "tailwind",
            "weather",
            "conditions",
            "limits",
            "maximum",
            "minimum",
            "operational",
            "knots",
            "kt",
            "mph",
            "m/s",
            "gusts",
            "turbulence",
            "visibility",
            "gliding",
            "glider",
            "aircraft",
            "operations",
            "safety",
            "emergency",
            "procedure",
            "altitude",
            "approach",
            "landing",
            "takeoff",
            "circuit",
        ]

        found_terms = []
        content_lower = content.lower()
        for term in aviation_terms:
            if term in content_lower:
                found_terms.append(term)

        # Also extract any numeric values that might be limits
        import re

        numeric_patterns = re.findall(
            r"\d+\s*(?:knots|kt|mph|m/s|feet|ft|meters|m)\b", content_lower
        )
        if numeric_patterns:
            # Add unique numeric limits to key terms
            for pattern in set(numeric_patterns[:5]):  # Limit to 5 most common
                found_terms.append(pattern)

        metadata["key_terms"] = found_terms

        return metadata

    async def search_documents(
        self, query: str, top_k: int = 5
    ) -> list[ProcessedDocument]:
        """Search processed documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant processed documents
        """
        # Generate query embedding
        try:
            qe: Any = self.embedding_model.encode([query], convert_to_numpy=True)  # type: ignore
            if hasattr(qe, "tolist"):
                qe_list = qe.tolist()  # type: ignore
            else:
                qe_list = qe  # type: ignore
            query_embedding = qe_list[0] if qe_list else []  # type: ignore[index]
            print(f"Generated query embedding for: '{query[:50]}...'")
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        # Search in ChromaDB
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                print("Warning: No documents indexed in ChromaDB")
                return []

            results: dict[str, Any] = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection_count),
            )  # type: ignore[assignment]
            print(
                f"ChromaDB search returned {len(results.get('ids', [[]])[0])} results"
            )
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []

        if not results or not results.get("ids") or not results.get("ids")[0]:  # type: ignore[index]
            return []

        # Group chunks by document and reconstruct ProcessedDocuments
        document_chunks: dict[str, Any] = {}
        ids_list = results.get("ids", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        documents_list = results.get("documents", [[]])[0]

        for i, _chunk_id in enumerate(ids_list):
            try:
                md = metadatas_list[i]
                document_id = str(md.get("document_id", "unknown"))
                document_name = str(md.get("document_name", "unknown"))
                file_type = str(md.get("file_type", "application/octet-stream"))
                directory_path = str(md.get("directory_path", "."))
                chunk_content = str(documents_list[i])
            except Exception:
                continue

            if document_id not in document_chunks:
                document_chunks[document_id] = {
                    "name": document_name,
                    "file_type": file_type,
                    "directory_path": directory_path,
                    "chunks": [],
                    "full_content": "",
                }

            document_chunks[document_id]["chunks"].append(chunk_content)
            document_chunks[document_id]["full_content"] += chunk_content + " "

        # Reconstruct ProcessedDocument objects
        processed_docs = []
        for doc_id, doc_data in document_chunks.items():
            processed_doc = ProcessedDocument(
                id=doc_id,
                original_document=Document(
                    name=doc_data["name"],
                    file_path=f"{doc_data['directory_path']}/{doc_data['name']}",
                    file_type=doc_data["file_type"],
                    directory_path=doc_data["directory_path"],
                ),
                content=doc_data["full_content"].strip(),
                chunks=doc_data["chunks"],
                processed_at=datetime.now(UTC),
            )
            processed_docs.append(processed_doc)

        return processed_docs

    async def index_documents(self, processed_docs: list[ProcessedDocument]) -> None:
        """Index processed documents for fast retrieval.

        Args:
            processed_docs: List of processed documents to index
        """
        for doc in processed_docs:
            # Store document chunks with embeddings in ChromaDB
            if not doc.embeddings or len(doc.embeddings) != len(doc.chunks):
                print(
                    f"Warning: Document {doc.original_document.name} has invalid embeddings"
                )
                continue

            for i, (chunk, embedding) in enumerate(
                zip(doc.chunks, doc.embeddings, strict=False)
            ):
                chunk_id = f"{doc.id}_chunk_{i}"

                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "document_id": doc.id,
                            "document_name": doc.original_document.name,
                            "chunk_index": i,
                            "file_type": doc.original_document.file_type,
                            "directory_path": doc.original_document.directory_path,
                        }
                    ],
                    ids=[chunk_id],
                )

        print(
            f"Indexed {len(processed_docs)} documents with {sum(len(doc.chunks) for doc in processed_docs)} total chunks"
        )


if __name__ == "__main__":
    import argparse
    import asyncio
    from pathlib import Path

    from vgs_chatbot.models.document import Document

    parser = argparse.ArgumentParser(
        description="Reprocess and reindex all documents for embeddings."
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="data/documents",
        help="Directory containing documents to process (default: data/documents)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/vectors/chroma",
        help="Directory for persistent ChromaDB storage (default: data/vectors/chroma)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="SentenceTransformer model to use (default: multi-qa-MiniLM-L6-cos-v1)",
    )
    args = parser.parse_args()

    documents_dir = Path(args.documents_dir)
    persist_dir = args.persist_dir
    embedding_model = args.embedding_model

    # Remove existing ChromaDB index if present
    chroma_path = Path(persist_dir)
    if chroma_path.exists():
        import shutil

        print(f"Removing existing ChromaDB index at {chroma_path}...")
        shutil.rmtree(chroma_path)

    # Gather all documents
    doc_files = [f for f in documents_dir.glob("*") if f.is_file()]
    if not doc_files:
        print(f"No documents found in {documents_dir}. Exiting.")
        exit(1)

    # Guess file types
    EXT_TO_MIME = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    documents = []
    for f in doc_files:
        ext = f.suffix.lower()
        file_type = EXT_TO_MIME.get(ext, "application/octet-stream")
        documents.append(
            Document(
                name=f.name,
                file_path=str(f),
                file_type=file_type,
                directory_path=str(f.parent),
            )
        )

    print(
        f"Found {len(documents)} documents. Processing and indexing with model '{embedding_model}'..."
    )

    processor = RAGDocumentProcessor(
        embedding_model=embedding_model,
        persist_directory=persist_dir,
    )

    async def process_and_index():
        processed = await processor.process_documents(documents)
        await processor.index_documents(processed)

    asyncio.run(process_and_index())
    print("✅ Reprocessing and reindexing complete.")
