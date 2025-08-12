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

            # Create IEEE-style source references with detailed information
            sources = [doc.original_document.name for doc in context_documents]
            source_references = []

            # Create source references for each unique document
            processed_docs = set()
            for doc in context_documents:
                if doc.original_document.name not in processed_docs:
                    # Extract section title and page number from document content
                    section_title = self._extract_section_title(doc)
                    page_numbers = self._extract_page_numbers(doc)

                    source_ref = SourceReference(
                        document_name=doc.original_document.name,
                        section_title=section_title,
                        page_number=page_numbers,
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
        """Build context string from relevant documents.

        Args:
            documents: Documents to use for context

        Returns:
            Context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for doc in documents:
            context_parts.append(f"Document: {doc.original_document.name}")
            context_parts.append(f"Source: {doc.original_document.directory_path}")

            # Include metadata information if available
            if doc.metadata.get("key_terms"):
                context_parts.append(
                    f"Key terms: {', '.join(doc.metadata['key_terms'][:5])}"
                )
            if doc.metadata.get("total_pages"):
                context_parts.append(f"Total pages: {doc.metadata['total_pages']}")

            # Use more chunks with better size management for better context
            for i, chunk in enumerate(doc.chunks[:4]):  # Increased to 4 chunks
                # Use larger chunks but with smart truncation
                if len(chunk) > 800:
                    # Try to truncate at sentence boundary
                    sentences = chunk.split(". ")
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) < 800:
                            truncated += sentence + ". "
                        else:
                            break
                    chunk = truncated.strip() or chunk[:800]
                context_parts.append(f"Content {i+1}: {chunk}")

            context_parts.append("---")

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

        # Create IEEE-style reference mapping
        doc_references = {}
        ref_counter = 1

        if context_documents:
            for doc in context_documents:
                if doc.original_document.name not in doc_references:
                    doc_references[doc.original_document.name] = ref_counter
                    ref_counter += 1

        # Build reference instruction for the LLM
        ref_instruction = "Available references for citation:\n"
        for doc_name, ref_num in doc_references.items():
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
- Duty Holder Orders (DHOs) are more restrictive and specialised for VGS operations
- Group Air Staff Orders (GASOs) are senior policy documents, but are less specific than DHOs
- CRITICAL: When conflicts arise, prioritize DHO guidance as it is more specific and restrictive for VGS operations
- Always check DHOs first for VGS-specific procedures and limitations

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
1. Search the context thoroughly for information related to the question
2. Prioritize DHO (Duty Holder Orders) information over GASO (Group Air Staff Orders) when both are available
3. For wind limits questions, provide LAYERED answers showing both solo and dual flying limits where applicable
4. Clearly identify which pilot category the person falls into e.g. U/T, G2, G1, B2, B1, A category (including flying supervisor). Assume dual flying unless solo is mentioned.
5. CRITICAL: When interpreting tables, read column headers carefully and match the pilot category to the correct column. Pay close attention to table structure and ensure you're reading values from the appropriate row/column intersection. Double-check that limits are from the correct category column, not adjacent columns
6. Provide specific details, numbers, and requirements when available
7. Use IEEE citation format - add ONLY numbered references like [1], [2], etc. after statements from specific documents
8. Only cite documents that you actually reference in your answer
9. CRITICAL: Your response MUST END immediately after your final sentence. Do NOT add "References:", "Summary:", or any list of references at the end
10. If you find conflicting information, explain the hierarchy (DHOs take precedence over GASOs)
11. Consider the specific VGS operational context when interpreting general aviation guidance
12. If you find partial information, provide what you can and note what might be incomplete
13. Only say information is not available if you genuinely cannot find any relevant details in the context
14. IMPORTANT: End your response with your conclusion. Do NOT add any additional sections, references, or summaries after your main answer

EXAMPLE FORMAT:
"Maximum wind speed is 25 knots [1]. This applies to dual flying operations."

DO NOT FORMAT LIKE THIS:
"Maximum wind speed is 25 knots [1]. This applies to dual flying operations.

References:
[1] Document name..."

Answer:"""

        return prompt, doc_references

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

        return "1"  # Fallback
