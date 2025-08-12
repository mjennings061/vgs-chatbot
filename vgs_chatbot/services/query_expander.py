"""Query expansion service for improved RAG retrieval."""

from typing import Any

from langchain_openai import ChatOpenAI


class QueryExpander:
    """Service to expand user queries for better document retrieval."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini") -> None:
        """Initialize query expander.

        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use for query expansion
        """
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model, temperature=0.1)

    async def expand_query(self, user_query: str) -> list[str]:
        """Expand a user query into multiple search variations.

        Args:
            user_query: Original user query

        Returns:
            List of expanded query variations
        """
        expansion_prompt = f"""You are an expert at analyzing user queries about aviation procedures and pilot qualifications. Your task is to generate multiple search query variations that will help find the most relevant content in official aviation documents.

Given the user query, generate 4-6 alternative search queries that would help find the specific procedural information being requested. Focus on:

1. Formal procedural language used in official documents
2. Specific terminology and abbreviations 
3. Related concepts and requirements
4. Different ways the information might be expressed

Original query: "{user_query}"

Guidelines:
- Include both natural language and formal procedural variants
- Add relevant aviation terminology and abbreviations
- Consider how requirements are typically structured in official documents
- Include related concepts that might contain the answer
- Focus on actionable, specific search terms

Generate 4-6 search query variations, one per line, without numbering or bullets:"""

        try:
            response = await self.llm.ainvoke(expansion_prompt)
            expanded_queries = [
                line.strip()
                for line in response.content.strip().split('\n')
                if line.strip()
            ]

            # Always include the original query
            if user_query not in expanded_queries:
                expanded_queries.insert(0, user_query)

            return expanded_queries[:6]  # Limit to 6 queries max

        except Exception as e:
            print(f"Error expanding query: {e}")
            # Fallback to original query
            return [user_query]

    async def synthesize_results(
        self,
        user_query: str,
        search_results: dict[str, Any]
    ) -> str:
        """Synthesize results from multiple search queries.

        Args:
            user_query: Original user query
            search_results: Dictionary mapping queries to their results

        Returns:
            Synthesized context string
        """
        if not search_results:
            return "No relevant documents found."

        # Combine all unique results while preserving order by relevance
        seen_chunks = set()
        combined_results = []

        for query, results in search_results.items():
            for doc in results:
                for chunk in doc.chunks:
                    chunk_key = f"{doc.original_document.name}_{hash(chunk)}"
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        combined_results.append({
                            'chunk': chunk,
                            'document': doc.original_document.name,
                            'query': query
                        })

        # Build synthesized context
        context_parts = [f"Search results for: '{user_query}'"]
        context_parts.append("=" * 50)

        for i, result in enumerate(combined_results[:10], 1):  # Limit to top 10 results
            doc_type = "DHO" if "DHO" in result['document'] else "GASO" if "GASO" in result['document'] or "Gp Air Staff" in result['document'] else "OTHER"
            context_parts.append(f"\nResult {i} [{doc_type}]: {result['document']}")
            context_parts.append(f"Found via query: '{result['query']}'")
            context_parts.append(f"Content: {result['chunk']}")
            context_parts.append("-" * 40)

        return "\n".join(context_parts)
