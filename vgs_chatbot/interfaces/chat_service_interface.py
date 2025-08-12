"""Chat service interface for LLM integration."""

from abc import ABC, abstractmethod

from vgs_chatbot.models.chat import ChatMessage, ChatResponse
from vgs_chatbot.models.document import ProcessedDocument


class ChatServiceInterface(ABC):
    """Abstract interface for chat services."""

    @abstractmethod
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
        pass

    @abstractmethod
    async def summarize_documents(self, documents: list[ProcessedDocument]) -> str:
        """Summarize documents for context.

        Args:
            documents: Documents to summarize

        Returns:
            Summary text
        """
        pass
