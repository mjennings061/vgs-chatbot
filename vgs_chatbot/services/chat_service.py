"""Chat service implementation with LLM integration."""

import time

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from vgs_chatbot.interfaces.chat_service_interface import ChatServiceInterface
from vgs_chatbot.models.chat import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    SourceReference,
)
from vgs_chatbot.models.document import ProcessedDocument
from vgs_chatbot.services.query_expander import QueryExpander


class LLMChatService(ChatServiceInterface):
    """LLM-powered chat service implementation."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4.1-nano") -> None:
        """Initialize chat service.

        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model, temperature=0.1)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.query_expander = QueryExpander(openai_api_key, model)

    async def generate_response_with_two_stage_rag(
        self, messages: list[ChatMessage], document_processor
    ) -> ChatResponse:
        """Generate response using two-stage RAG approach.

        Args:
            messages: Chat history
            document_processor: Document processor for retrieval

        Returns:
            Generated chat response with improved retrieval
        """
        start_time = time.time()

        # Get the latest user message
        latest_message = messages[-1]
        if latest_message.role != MessageRole.USER:
            raise ValueError("Latest message must be from user")

        try:
            # Stage 1: Query expansion
            print(f"Original query: {latest_message.content}")
            expanded_queries = await self.query_expander.expand_query(latest_message.content)
            print(f"Expanded to {len(expanded_queries)} queries: {expanded_queries}")

            # Stage 2: Multi-query retrieval
            all_results = {}
            for query in expanded_queries:
                try:
                    results = await document_processor.search_documents(query, top_k=3)
                    if results:
                        all_results[query] = results
                        print(f"Query '{query}' found {len(results)} documents")
                except Exception as e:
                    print(f"Error searching for '{query}': {e}")
                    continue

            if not all_results:
                print("No results found for any expanded query")
                return ChatResponse(
                    message="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                )

            # Combine and deduplicate results
            combined_docs = self._combine_search_results(all_results)
            print(f"Combined into {len(combined_docs)} unique documents")

            # Build enhanced context
            context = await self._build_context_from_documents(combined_docs)

            # Generate response using standard method with enhanced context
            prompt, ieee_references = self._create_prompt_with_context(
                latest_message.content,
                context,
                messages[:-1],
                combined_docs,
            )

            response = await self.llm.ainvoke(prompt)
            response_text = str(response.content).strip() if hasattr(response.content, 'strip') else str(response.content)

            # Create source references
            sources = [doc.original_document.name for doc in combined_docs]
            source_references = []

            processed_docs = set()
            for doc in combined_docs:
                if doc.original_document.name not in processed_docs:
                    best_section = self._extract_best_section_title(doc)
                    page_range = self._extract_document_page_range(doc)

                    source_ref = SourceReference(
                        document_name=doc.original_document.name,
                        section_title=best_section,
                        page_number=page_range,
                    )
                    source_references.append(source_ref)
                    processed_docs.add(doc.original_document.name)

            processing_time = time.time() - start_time
            print(f"Two-stage RAG completed in {processing_time:.2f}s")

            return ChatResponse(
                message=response_text,
                sources=sources,
                source_references=source_references,
                confidence=0.8,
                processing_time=processing_time,
                ieee_references=ieee_references,
            )

        except Exception as e:
            return ChatResponse(
                message=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
            )

    async def generate_response(
        self, messages: list[ChatMessage], context_documents: list[ProcessedDocument]
    ) -> ChatResponse:
        """Generate chat response using LLM and document context.

        Args:
            messages: Chat history
            context_documents: Relevant documents for context

        Returns:
            Generated chat response
        """
        start_time = time.time()

        # Get the latest user message
        latest_message = messages[-1]
        if latest_message.role != MessageRole.USER:
            raise ValueError("Latest message must be from user")

        # Build context from documents
        context = await self._build_context_from_documents(context_documents)

        # Create prompt with context and get IEEE references
        prompt, ieee_references = self._create_prompt_with_context(
            latest_message.content,
            context,
            messages[:-1],  # Exclude latest message from history
            context_documents,
        )

        try:
            # Generate response using LLM
            # Use invoke instead of agenerate for simpler string-to-string handling
            response = await self.llm.ainvoke(prompt)
            response_text = response.content.strip()

            # Create IEEE-style source references from context documents
            sources = [doc.original_document.name for doc in context_documents]
            source_references = []

            # Create source references for each document with proper section extraction
            processed_docs = set()
            for doc in context_documents:
                if doc.original_document.name not in processed_docs:
                    # Extract the best section title and page numbers from all chunks
                    best_section = self._extract_best_section_title(doc)
                    page_range = self._extract_document_page_range(doc)

                    source_ref = SourceReference(
                        document_name=doc.original_document.name,
                        section_title=best_section,
                        page_number=page_range,
                    )
                    source_references.append(source_ref)
                    processed_docs.add(doc.original_document.name)

            processing_time = time.time() - start_time

            # Create response with IEEE reference mapping
            response_with_refs = ChatResponse(
                message=response_text,
                sources=sources,
                source_references=source_references,
                confidence=0.8,  # Placeholder confidence score
                processing_time=processing_time,
                ieee_references=ieee_references,
            )

            return response_with_refs

        except Exception as e:
            return ChatResponse(
                message=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
            )

    async def summarize_documents(self, documents: list[ProcessedDocument]) -> str:
        """Summarize documents for context.

        Args:
            documents: Documents to summarize

        Returns:
            Summary text
        """
        if not documents:
            return "No documents available for context."

        # Combine document content
        combined_content = ""
        for doc in documents:
            combined_content += f"Document: {doc.original_document.name}\n"
            combined_content += f"Content: {doc.content[:500]}...\n\n"

        # Generate summary
        summary_prompt = f"""
        Please provide a concise summary of the following VGS (Volunteer Gliding Squadron) documents, focusing on key operational information, procedures, and limitations relevant to RAF Air Cadet gliding training.

        Context: These documents relate to VGS operations using Viking gliders, where DHOs (Duty Holder Orders) are more restrictive than GASOs (Group Air Staff Orders).

        {combined_content}

        Summary:"""

        try:
            response = await self.llm.ainvoke(summary_prompt)
            return response.content.strip()
        except Exception:
            return "Unable to generate document summary."

    async def _build_context_from_documents(
        self, documents: list[ProcessedDocument]
    ) -> str:
        """Build context string from relevant documents with DHO prioritization.

        Args:
            documents: Documents to use for context

        Returns:
            Context string with DHO content prioritized
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []

        # Prioritize DHO documents over GASO documents
        dho_docs = [doc for doc in documents if "DHO" in doc.original_document.name]
        gaso_docs = [doc for doc in documents if "GASO" in doc.original_document.name or "Gp Air Staff" in doc.original_document.name]
        other_docs = [doc for doc in documents if doc not in dho_docs and doc not in gaso_docs]

        # First, prioritize any DHO chunks that contain procedural requirements
        critical_dho_chunks = []
        other_dho_chunks = []

        for doc in dho_docs:
            for chunk in doc.chunks:
                if self._is_procedural_requirement_content(chunk):
                    critical_dho_chunks.append((doc, chunk))
                else:
                    other_dho_chunks.append((doc, chunk))

        # Build context with critical DHO content first
        if critical_dho_chunks:
            context_parts.append("CRITICAL OPERATIONAL REQUIREMENTS (DHO):")
            context_parts.append("=" * 50)

            for i, (doc, chunk) in enumerate(critical_dho_chunks[:3], 1):  # Top 3 critical chunks
                section_title = self._extract_chunk_section_info(chunk)
                context_parts.append(f"DHO Requirement {i}: {doc.original_document.name}")
                context_parts.append(f"Section: {section_title}")
                context_parts.append(f"Content: {chunk}")  # Full chunk, no truncation
                context_parts.append("-" * 40)

        # Then add other DHO content
        if other_dho_chunks:
            context_parts.append("\nADDITIONAL DHO GUIDANCE:")
            context_parts.append("=" * 30)

            for i, (doc, chunk) in enumerate(other_dho_chunks[:2], 1):  # Top 2 additional chunks
                section_title = self._extract_chunk_section_info(chunk)
                context_parts.append(f"DHO Info {i}: {doc.original_document.name}")
                context_parts.append(f"Section: {section_title}")
                # Truncate non-critical content
                if len(chunk) > 800:
                    sentences = chunk.split(". ")
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) < 800:
                            truncated += sentence + ". "
                        else:
                            break
                    chunk = truncated.strip() or chunk[:800]
                context_parts.append(f"Content: {chunk}")
                context_parts.append("-" * 40)

        # Only add GASO content if no critical DHO content exists
        if not critical_dho_chunks and gaso_docs:
            context_parts.append("\nGENERAL POLICY (GASO - Use only if DHO guidance unavailable):")
            context_parts.append("=" * 60)

            for doc in gaso_docs[:1]:  # Only 1 GASO doc for fallback
                for i, chunk in enumerate(doc.chunks[:2], 1):  # Max 2 chunks
                    section_title = self._extract_chunk_section_info(chunk)
                    context_parts.append(f"GASO Reference {i}: {doc.original_document.name}")
                    context_parts.append(f"Section: {section_title}")
                    # Heavily truncate GASO content
                    if len(chunk) > 400:
                        chunk = chunk[:400] + "..."
                    context_parts.append(f"Content: {chunk}")
                    context_parts.append("-" * 40)

        return "\n".join(context_parts)

    def _create_prompt_with_context(
        self,
        user_question: str,
        context: str,
        chat_history: list[ChatMessage],
        context_documents: list[ProcessedDocument] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Create prompt with context and chat history.

        Args:
            user_question: Current user question
            context: Document context
            chat_history: Previous chat messages
            context_documents: List of context documents for IEEE references

        Returns:
            Tuple of (formatted prompt, IEEE reference mapping)
        """
        # Build chat history string (reduced to save tokens)
        history_str = ""
        for msg in chat_history[-3:]:  # Limit to last 3 messages only
            role = "Human" if msg.role == MessageRole.USER else "Assistant"
            history_str += f"{role}: {msg.content}\n"

        # Create IEEE-style reference mapping at document level
        ieee_references = {}
        ref_counter = 1

        if context_documents:
            processed_docs = set()
            for doc in context_documents:
                if doc.original_document.name not in processed_docs:
                    ieee_references[doc.original_document.name] = ref_counter
                    processed_docs.add(doc.original_document.name)
                    ref_counter += 1

        # Build reference instruction for the LLM
        ref_instruction = "Available references for citation:\n"
        for doc_name, ref_num in ieee_references.items():
            ref_instruction += f"[{ref_num}] {doc_name}\n"

        prompt = f"""You are a knowledgeable assistant for the Volunteer Gliding Squadron (VGS) that answers questions about aviation operations, safety procedures, and regulations. Use the provided documentation context to give accurate, detailed answers.

VGS OPERATIONAL CONTEXT:
- A Volunteer Gliding Squadron (VGS) consists of volunteer staff and flight staff cadets (FSCs) who provide basic gliding training to RAF Air Cadets
- VGS services include: Glider Instruction Flights (GIF), Gliding Scholarship (G/S), Advanced Glider Training (AGT), G2 pilot training, and G1 pilot training
- Central Gliding School (CGS) oversees VGS operations and provides instructor training for G2, G1, B2, B1, A2 categories
- 2FTS OC Ops is the senior operator and head of VGS operations
- VGSs operate Viking gliders exclusively
- VGSs can service aircraft but cannot carry out maintenance - only the AMO (Aircraft Maintenance Organisation) can perform maintenance

PILOT CATEGORISATION:
- **GS (Gliding Scholarship) cadets** are classified as "Solo Cadets & Trainees" when flying solo
- **GS cadets with an instructor** fall under "All Cadet Dual Flying" category
- **U/T (Under Training) pilots** including GS and AGT cadets are in the "Solo Cadets & Trainees" category. Everything below G2 is U/T.
- **G2** are graded pilots
- **G1 pilots** are qualified to take passengers flying and have higher wind limits
- **B2, B1, A2** are qualified gliding instructor (QGI) categories
- **Flying supervisor** is a qualification awarded to A category instructors which enables additional responsibilities and oversight e.g. B1* or A2*

DOCUMENTATION HIERARCHY:
- Duty Holder Orders (DHOs) are the PRIMARY SOURCE for specific VGS operational requirements
- DHOs contain the actual procedural requirements including specific launch numbers and qualifications
- Group Air Staff Orders (GASOs) are general policy documents - use ONLY when DHO guidance is unavailable
- CRITICAL: DHO content takes absolute precedence over GASO content for operational questions
- When DHO content is available (marked as "CRITICAL OPERATIONAL REQUIREMENTS"), use ONLY that information

ANALYSIS APPROACH:
When asked questions for specific pilot categories:
1. **First** identify which category the pilot fits into e.g. U/T, G2, G1, B2, B1, A category (including flying supervisor)
2. **Then** extract information relevant to the query for applicable category
3. For questions about limits (e.g. wind), provide LAYERED answers showing both solo and dual flying limits where applicable
4. CRITICAL: When asked about pilot limits without specifying solo/dual, show BOTH: "Solo: [limits] | Dual: [limits]"
5. **Finally** answer the question and explain the reasoning behind the answer, with sources.

IMPORTANT: The context contains relevant information from official VGS documents. Read it carefully and extract specific details to answer the question. Do not say you cannot find information unless you have thoroughly reviewed all the context.

{ref_instruction}
Context from VGS documents:
{context}

Previous conversation:
{history_str}

Current question: {user_question}

Instructions:
1. **CRITICAL**: If "CRITICAL OPERATIONAL REQUIREMENTS (DHO)" section exists, use ONLY that information for your answer
2. DHO content contains the actual procedural requirements - prioritize it completely over GASO content
3. Search the context thoroughly for information related to the question
4. For pilot qualification questions, look specifically for launch numbers, requirements, and progression criteria in DHO sections
5. For wind limits questions, provide LAYERED answers showing both solo and dual flying limits where applicable
6. Clearly identify which pilot category the person falls into e.g. U/T, G2, G1, B2, B1, A category (including flying supervisor). Assume dual flying unless solo is mentioned.
7. CRITICAL: When interpreting tables, read column headers carefully and match the pilot category to the correct column. Pay close attention to table structure and ensure you're reading values from the appropriate row/column intersection. Double-check that limits are from the correct category column, not adjacent columns
8. Provide specific details, numbers, and requirements when available - especially launch numbers from DHO content
9. Use IEEE citation format - add ONLY numbered references like [1], [2], etc. after statements from specific documents
10. Only cite documents that you actually reference in your answer
11. CRITICAL: Your response MUST END immediately after your final sentence. Do NOT add "References:", "Summary:", or any list of references at the end
12. If you find conflicting information between DHO and GASO, use ONLY the DHO information
13. Consider the specific VGS operational context when interpreting general aviation guidance
14. If you find partial information, provide what you can and note what might be incomplete
15. Only say information is not available if you genuinely cannot find any relevant details in the DHO content
16. IMPORTANT: End your response with your conclusion. Do NOT add any additional sections, references, or summaries after your main answer

EXAMPLE FORMAT:
"Maximum wind speed is 25 knots [1]. This applies to dual flying operations."

DO NOT FORMAT LIKE THIS:
"Maximum wind speed is 25 knots [1]. This applies to dual flying operations.

References:
[1] Document name..."

Answer:"""

        return prompt, ieee_references

    def _extract_section_title(self, doc: ProcessedDocument) -> str:
        """Extract the most relevant section title from document content."""
        content = doc.content

        # Look for common section patterns in the content
        lines = content.split("\n")
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line:
                # Check for section headers
                if (
                    ("WEATHER" in line.upper() and "LIMIT" in line.upper())
                    or ("WIND" in line.upper() and "LIMIT" in line.upper())
                    or ("OPERATIONAL" in line.upper() and "LIMIT" in line.upper())
                ):
                    return line
                # Check for common document section patterns
                if line.isupper() and len(line) > 10 and len(line) < 80:
                    if any(
                        word in line
                        for word in ["WEATHER", "OPERATIONS", "LIMITS", "PROCEDURES"]
                    ):
                        return line

        # Fallback to metadata or generic title
        if hasattr(doc, "metadata") and doc.metadata.get("sections"):
            sections = doc.metadata["sections"]
            if sections:
                return sections[0]

        return "Operational Guidelines"

    def _extract_page_numbers(self, doc: ProcessedDocument) -> str:
        """Extract page numbers from document content."""
        content = doc.content
        page_numbers = set()

        # Look for page markers in the content
        lines = content.split("\n")
        for line in lines:
            if "--- PAGE" in line:
                # Extract page number from markers like "--- PAGE 107 ---"
                import re

                match = re.search(r"PAGE\s+(\d+)", line)
                if match:
                    page_numbers.add(match.group(1))

        if page_numbers:
            sorted_pages = sorted(page_numbers, key=int)
            if len(sorted_pages) == 1:
                return sorted_pages[0]
            else:
                return f"{sorted_pages[0]}-{sorted_pages[-1]}"

        # Fallback to metadata
        if hasattr(doc, "metadata") and doc.metadata.get("total_pages"):
            return f"1-{doc.metadata['total_pages']}"

        return "1"

    def _combine_search_results(self, all_results: dict[str, list]) -> list:
        """Combine and deduplicate search results from multiple queries.

        Args:
            all_results: Dictionary mapping queries to their search results

        Returns:
            Combined list of unique documents
        """
        seen_docs = set()
        combined_docs = []

        # Process results in order of query importance
        for _query, results in all_results.items():
            for doc in results:
                doc_key = f"{doc.original_document.name}_{id(doc)}"
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    combined_docs.append(doc)

        return combined_docs

    def _is_procedural_requirement_content(self, chunk: str) -> bool:
        """Determine if a chunk contains procedural requirements or standards.

        Args:
            chunk: Text chunk to analyze

        Returns:
            True if chunk contains procedural requirements
        """
        import re

        chunk_lower = chunk.lower()

        # Look for formal requirement language patterns
        requirement_patterns = [
            r"minimum of \d+",  # "minimum of X"
            r"shall be",        # mandatory language
            r"must be",         # mandatory language
            r"should be",       # requirement language
            r"required to",     # requirement language
            r"requirements?:",  # section headers about requirements
            r"qualification",   # qualification standards
            r"competence",      # competency requirements
            r"authorized to",   # authorization requirements
            r"may be authorized", # authorization language
            r"mandatory",       # mandatory requirements
            r"pass the",        # test/exam requirements
            r"complete[d]? \d+", # completion requirements with numbers
        ]

        # Count matches of requirement patterns
        pattern_matches = sum(1 for pattern in requirement_patterns
                            if re.search(pattern, chunk_lower))

        # Look for structured requirement lists (numbered/lettered items)
        structured_list_patterns = [
            r"\(\d+\)",         # (1), (2), etc.
            r"\([a-z]\)",       # (a), (b), etc.
            r"^\s*\d+\.",       # 1., 2., etc. at line start
            r"^\s*[a-z]\.",     # a., b., etc. at line start
        ]

        lines = chunk.split('\n')
        structured_items = sum(1 for line in lines
                             for pattern in structured_list_patterns
                             if re.search(pattern, line.strip()))

        # Consider it procedural if:
        # - Multiple requirement patterns found, OR
        # - Structured list with some requirement language, OR
        # - High density of requirement language
        return (pattern_matches >= 3 or
                (structured_items >= 2 and pattern_matches >= 1) or
                (len(chunk) > 0 and pattern_matches / len(chunk.split()) > 0.02))  # Fallback

    def _extract_enhanced_section_title(self, doc: ProcessedDocument) -> str:
        """Extract enhanced section title using chunk metadata.

        Args:
            doc: ProcessedDocument with chunks

        Returns:
            Most relevant section title
        """
        # Try to find weather/operational sections in chunks
        best_section = "General Content"
        priority_keywords = ["weather", "limit", "wind", "operational", "procedure"]

        for chunk in doc.chunks[:3]:  # Check first 3 chunks
            lines = chunk.split("\n")
            for line in lines[:5]:  # Check first 5 lines of each chunk
                line = line.strip()
                if line:
                    # Check for enhanced structure headings
                    if line.startswith("===") and line.endswith("==="):
                        section_title = line.replace("===", "").strip()
                        # Prioritize weather/operational sections
                        if any(keyword in section_title.lower() for keyword in priority_keywords):
                            return section_title
                        elif best_section == "General Content":
                            best_section = section_title

                    # Check for section title patterns (length-based filtering)
                    elif len(line) < 80 and len(line) > 10:  # Reasonable section title length
                        return line

        return best_section

    def _extract_enhanced_page_numbers(self, doc: ProcessedDocument) -> str:
        """Extract enhanced page numbers using chunk metadata.

        Args:
            doc: ProcessedDocument with chunks

        Returns:
            Page number range or single page
        """
        page_numbers = set()

        # Look through chunks for page markers
        for chunk in doc.chunks:
            lines = chunk.split("\n")
            for line in lines:
                if "--- PAGE" in line:
                    import re
                    match = re.search(r"PAGE\s+(\d+)", line)
                    if match:
                        page_numbers.add(match.group(1))

        if page_numbers:
            sorted_pages = sorted(page_numbers, key=int)
            if len(sorted_pages) == 1:
                return sorted_pages[0]
            else:
                return f"{sorted_pages[0]}-{sorted_pages[-1]}"

        # Fallback to original method
        return self._extract_page_numbers(doc)

    def _extract_chunk_section_info(self, chunk: str) -> str:
        """Extract section information from a specific chunk with proper reference numbers.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Section information string with proper DHO/GASO format
        """
        lines = chunk.split("\n")
        import re

        # Look for enhanced section headers first (=== ... ===)
        for line in lines[:5]:
            line_stripped = line.strip()
            if line_stripped.startswith("===") and line_stripped.endswith("==="):
                section_title = line_stripped.replace("===", "").strip()
                if section_title:
                    return section_title

        # Look for DHO and GASO number patterns throughout the content
        for line in lines[:15]:  # Check more lines for references
            line_stripped = line.strip()

            # DHO number patterns (DHO followed by 4 digits)
            dho_match = re.search(r"DHO\s*(\d{4})", line_stripped.upper())
            if dho_match:
                return f"DHO {dho_match.group(1)}"

            # GASO number patterns (GASO followed by digits)
            gaso_match = re.search(r"GASO\s*(\d+)", line_stripped.upper())
            if gaso_match:
                return f"GASO {gaso_match.group(1)}"

            # FTP references (often part of DHO content)
            ftp_match = re.search(r"(FTP\d{4})", line_stripped.upper())
            if ftp_match:
                return f"Training Reference: {ftp_match.group(1)}"

        # Look for general document structure markers
        for line in lines[:5]:
            line_stripped = line.strip()
            if re.match(
                r"^(ANNEX|APPENDIX|SECTION|CHAPTER|PART)\s+[A-Z0-9]",
                line_stripped.upper(),
            ):
                return line_stripped

        # Fallback to first meaningful line that looks like a title
        for line in lines[:3]:
            line_stripped = line.strip()
            if (line_stripped and
                len(line_stripped) > 5 and
                len(line_stripped) < 100 and
                not line_stripped.startswith("(")):
                return line_stripped

        return "General Content"

    def _extract_chunk_page_numbers(self, chunk: str) -> str:
        """Extract page numbers from a specific chunk.

        Args:
            chunk: Text chunk to analyze

        Returns:
            Page number range or single page
        """
        import re

        page_numbers = set()

        # Look for page markers in the chunk
        lines = chunk.split("\n")
        for line in lines:
            if "--- PAGE" in line:
                match = re.search(r"PAGE\s+(\d+)", line)
                if match:
                    page_numbers.add(match.group(1))

        if page_numbers:
            sorted_pages = sorted(page_numbers, key=int)
            if len(sorted_pages) == 1:
                return sorted_pages[0]
            else:
                return f"{sorted_pages[0]}-{sorted_pages[-1]}"

        return "Unknown"  # More specific than generic "1"

    def _extract_best_section_title(self, doc: ProcessedDocument) -> str:
        """Extract the best section title from a document's chunks.

        Args:
            doc: ProcessedDocument with chunks

        Returns:
            Most relevant section title
        """
        # Look through chunks for the best section title
        section_titles = []

        for chunk in doc.chunks[:3]:  # Check first 3 chunks
            chunk_section = self._extract_chunk_section_info(chunk)
            if chunk_section != "General Content":
                section_titles.append(chunk_section)

        # Return the most specific section title found
        if section_titles:
            return section_titles[0]

        # Fallback to metadata or generic title
        if hasattr(doc, "metadata") and doc.metadata.get("sections"):
            sections = doc.metadata["sections"]
            if sections:
                return sections[0]

        return "General Content"

    def _extract_document_page_range(self, doc: ProcessedDocument) -> str:
        """Extract page range from a document's chunks.

        Args:
            doc: ProcessedDocument with chunks

        Returns:
            Page range or single page
        """
        page_numbers = set()

        # Collect page numbers from all chunks
        for chunk in doc.chunks:
            chunk_pages = self._extract_chunk_page_numbers(chunk)
            if chunk_pages != "Unknown":
                # Handle page ranges
                if "-" in chunk_pages:
                    start, end = chunk_pages.split("-", 1)
                    page_numbers.add(start)
                    page_numbers.add(end)
                else:
                    page_numbers.add(chunk_pages)

        if page_numbers:
            sorted_pages = sorted(page_numbers, key=lambda x: int(x) if x.isdigit() else 0)
            if len(sorted_pages) == 1:
                return sorted_pages[0]
            else:
                return f"{sorted_pages[0]}-{sorted_pages[-1]}"

        # Fallback to metadata
        if hasattr(doc, "metadata") and doc.metadata.get("total_pages"):
            return f"1-{doc.metadata['total_pages']}"

        return "1"
