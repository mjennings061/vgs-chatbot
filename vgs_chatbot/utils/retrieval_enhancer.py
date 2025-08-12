"""Retrieval quality enhancement utilities for hybrid search and synonym mapping."""

import re
from typing import Any


class RetrievalEnhancer:
    """Enhances document retrieval with keyword matching and synonym expansion."""

    def __init__(self) -> None:
        """Initialize retrieval enhancer with aviation-specific synonym mapping."""
        # Aviation/gliding synonym mapping
        self.synonym_map = {
            "gs": "gliding scholarship",
            "g/s": "gliding scholarship",
            "u/t": "under training",
            "qgi": "qualified gliding instructor",
            "agt": "advanced glider training",
            "gif": "glider instruction flight",
            "cgs": "central gliding school",
            "amo": "aircraft maintenance organisation",
            "dho": "duty holder orders",
            "gaso": "group air staff orders",
            "fsc": "flight staff cadet",
            "vgs": "volunteer gliding squadron",
            "2fts": "2 flying training school",
            "kt": "knots",
            "kts": "knots",
            "x-wind": "crosswind",
            "wind limit": "wind limits",
            "limit": "limits",
            "max": "maximum",
            "min": "minimum",
        }

        # Aviation terms for context weighting
        self.aviation_terms = {
            "wind", "weather", "limit", "glider", "pilot", "cadet", "training",
            "instructor", "flying", "aircraft", "operation", "safety", "procedure",
            "knots", "crosswind", "gust", "visibility", "cloud", "ceiling"
        }

    def expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with aviation synonyms for better matching.

        Args:
            query: Original search query

        Returns:
            Expanded query with synonyms
        """
        expanded_query = query.lower()

        # Replace synonyms in query
        for abbreviation, full_form in self.synonym_map.items():
            # Match whole words only to avoid partial replacements
            pattern = r'\b' + re.escape(abbreviation) + r'\b'
            expanded_query = re.sub(pattern, f"{abbreviation} {full_form}", expanded_query)

        return expanded_query

    def calculate_keyword_boost(self, query: str, document_content: str) -> float:
        """Calculate keyword overlap boost for hybrid search.

        Args:
            query: Search query
            document_content: Document content to match against

        Returns:
            Keyword boost score (0.0 to 1.0)
        """
        query_lower = query.lower()
        content_lower = document_content.lower()

        # Extract meaningful terms from query (remove stop words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "how", "when", "where", "why"}
        query_terms = [term.strip(".,!?;:()[]{}") for term in query_lower.split() if term not in stop_words and len(term) > 2]

        if not query_terms:
            return 0.0

        # Count exact matches
        exact_matches = sum(1 for term in query_terms if term in content_lower)

        # Count partial matches (for compound terms)
        partial_matches = 0
        for term in query_terms:
            if len(term) > 4:  # Only check partial matches for longer terms
                # Check for term as part of other words
                if any(term in word for word in content_lower.split() if term != word):
                    partial_matches += 0.5

        # Boost for aviation-specific terms
        aviation_boost = sum(0.2 for term in query_terms if term in self.aviation_terms)

        # Calculate final score
        match_ratio = (exact_matches + partial_matches) / len(query_terms)
        keyword_score = min(1.0, match_ratio + aviation_boost)

        return keyword_score

    def extract_key_terms(self, text: str) -> list[str]:
        """Extract key aviation terms from text for indexing.

        Args:
            text: Input text

        Returns:
            List of key terms found
        """
        text_lower = text.lower()
        found_terms = []

        # Find aviation terms
        for term in self.aviation_terms:
            if term in text_lower:
                found_terms.append(term)

        # Find numeric patterns (wind speeds, etc.)
        numeric_patterns = re.findall(
            r'\d+\s*(?:knots|kt|kts|mph|m/s|feet|ft|meters|m)\b',
            text_lower
        )
        found_terms.extend(numeric_patterns[:5])  # Limit to 5 most relevant

        # Find specific aviation abbreviations
        aviation_abbrevs = re.findall(
            r'\b(?:gs|g/s|u/t|qgi|agt|gif|cgs|amo|dho|gaso|fsc|vgs|2fts)\b',
            text_lower
        )
        found_terms.extend(aviation_abbrevs)

        return list(set(found_terms))  # Remove duplicates

    def hybrid_score_documents(
        self,
        query: str,
        semantic_results: list[dict[str, Any]],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> list[dict[str, Any]]:
        """Combine semantic similarity with keyword matching for hybrid search.

        Args:
            query: Search query
            semantic_results: Results from semantic search with 'content' and 'distance' keys
            semantic_weight: Weight for semantic similarity (default 0.7)
            keyword_weight: Weight for keyword matching (default 0.3)

        Returns:
            Reordered results with hybrid scores
        """
        if not semantic_results:
            return []

        # Expand query for better keyword matching
        expanded_query = self.expand_query_with_synonyms(query)

        # Calculate hybrid scores
        enhanced_results = []
        for result in semantic_results:
            content = result.get('content', '')
            semantic_distance = result.get('distance', 1.0)

            # Convert distance to similarity (lower distance = higher similarity)
            semantic_similarity = 1.0 - min(semantic_distance, 1.0)

            # Calculate keyword boost
            keyword_boost = self.calculate_keyword_boost(expanded_query, content)

            # Calculate hybrid score
            hybrid_score = (semantic_weight * semantic_similarity) + (keyword_weight * keyword_boost)

            enhanced_result = result.copy()
            enhanced_result['hybrid_score'] = hybrid_score
            enhanced_result['keyword_boost'] = keyword_boost
            enhanced_result['semantic_similarity'] = semantic_similarity

            enhanced_results.append(enhanced_result)

        # Sort by hybrid score (highest first)
        enhanced_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return enhanced_results

    def preprocess_query_for_search(self, query: str) -> str:
        """Preprocess query for optimal search performance.

        Args:
            query: Raw user query

        Returns:
            Preprocessed query optimized for search
        """
        # Expand with synonyms
        expanded = self.expand_query_with_synonyms(query)

        # Add context hints for aviation queries
        if any(term in query.lower() for term in ["wind", "limit", "cadet", "pilot"]):
            expanded += " aviation gliding weather operational"

        if any(term in query.lower() for term in ["gs", "gliding scholarship"]):
            expanded += " cadet training under training solo dual"

        return expanded.strip()
