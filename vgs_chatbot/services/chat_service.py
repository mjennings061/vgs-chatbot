"""Chat service implementation with LLM integration."""

import time
from typing import List

from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from vgs_chatbot.interfaces.chat_service_interface import ChatServiceInterface
from vgs_chatbot.models.chat import ChatMessage, ChatResponse, MessageRole
from vgs_chatbot.models.document import ProcessedDocument


class LLMChatService(ChatServiceInterface):
    """LLM-powered chat service implementation."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo") -> None:
        """Initialize chat service.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    async def generate_response(
        self,
        messages: List[ChatMessage],
        context_documents: List[ProcessedDocument]
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
            response = await self.llm.agenerate([prompt])
            response_text = response.generations[0][0].text.strip()
            
            # Extract source references
            sources = [doc.original_document.name for doc in context_documents]
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                message=response_text,
                sources=sources,
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
    
    async def summarize_documents(self, documents: List[ProcessedDocument]) -> str:
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
            response = await self.llm.agenerate([summary_prompt])
            return response.generations[0][0].text.strip()
        except Exception:
            return "Unable to generate document summary."
    
    async def _build_context_from_documents(
        self, 
        documents: List[ProcessedDocument]
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
            
            # Use first few chunks for context
            for i, chunk in enumerate(doc.chunks[:3]):  # Limit to first 3 chunks
                context_parts.append(f"Content {i+1}: {chunk}")
            
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _create_prompt_with_context(
        self,
        user_question: str,
        context: str,
        chat_history: List[ChatMessage]
    ) -> str:
        """Create prompt with context and chat history.
        
        Args:
            user_question: Current user question
            context: Document context
            chat_history: Previous chat messages
            
        Returns:
            Formatted prompt
        """
        # Build chat history string
        history_str = ""
        for msg in chat_history[-10:]:  # Limit to last 10 messages
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