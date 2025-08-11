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

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini") -> None:
        """Initialize chat service.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    async def generate_response(
        self,
        messages: list[ChatMessage],
        context_documents: list[ProcessedDocument]
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

        # Create prompt with context
        prompt = self._create_prompt_with_context(
            latest_message.content,
            context,
            messages[:-1]  # Exclude latest message from history
        )

        try:
            # Generate response using LLM
            # Use invoke instead of agenerate for simpler string-to-string handling
            response = await self.llm.ainvoke(prompt)
            response_text = response.content.strip()

            # Extract source references with section titles and page numbers
            sources = [doc.original_document.name for doc in context_documents]
            source_references = []

            for doc in context_documents:
                # Extract section title and page number from metadata if available
                section_title = doc.metadata.get('section_title')
                page_number = doc.metadata.get('page_number')

                source_ref = SourceReference(
                    document_name=doc.original_document.name,
                    section_title=section_title,
                    page_number=page_number
                )
                source_references.append(source_ref)

            processing_time = time.time() - start_time

            return ChatResponse(
                message=response_text,
                sources=sources,
                source_references=source_references,
                confidence=0.8,  # Placeholder confidence score
                processing_time=processing_time
            )

        except Exception as e:
            return ChatResponse(
                message=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
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
        Please provide a concise summary of the following documents:

        {combined_content}

        Summary:"""

        try:
            response = await self.llm.ainvoke(summary_prompt)
            return response.content.strip()
        except Exception:
            return "Unable to generate document summary."

    async def _build_context_from_documents(
        self,
        documents: list[ProcessedDocument]
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

            # Include section titles and page numbers if available
            if doc.metadata.get('section_title'):
                context_parts.append(f"Section: {doc.metadata['section_title']}")
            if doc.metadata.get('page_number'):
                context_parts.append(f"Page: {doc.metadata['page_number']}")

            # Use fewer chunks and limit their size to stay within token limits
            for i, chunk in enumerate(doc.chunks[:2]):  # Limit to first 2 chunks only
                # Further truncate chunks if they're too long
                truncated_chunk = chunk[:300] if len(chunk) > 300 else chunk
                context_parts.append(f"Content {i+1}: {truncated_chunk}")

            context_parts.append("---")

        return "\n".join(context_parts)

    def _create_prompt_with_context(
        self,
        user_question: str,
        context: str,
        chat_history: list[ChatMessage]
    ) -> str:
        """Create prompt with context and chat history.
        
        Args:
            user_question: Current user question
            context: Document context
            chat_history: Previous chat messages
            
        Returns:
            Formatted prompt
        """
        # Build chat history string (reduced to save tokens)
        history_str = ""
        for msg in chat_history[-3:]:  # Limit to last 3 messages only
            role = "Human" if msg.role == MessageRole.USER else "Assistant"
            history_str += f"{role}: {msg.content}\n"

        prompt = f"""You are a helpful assistant for the Volunteer Gliding Squadron (VGS) that answers questions based on provided documentation. Use the following context to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context from documents:
{context}

Previous conversation:
{history_str}

Current question: {user_question}

Please provide a helpful and accurate answer based on the available information. Include references to specific documents when possible."""

        return prompt
