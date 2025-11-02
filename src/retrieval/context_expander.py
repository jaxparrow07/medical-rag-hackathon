"""
Context Window Expansion Module
Retrieves surrounding chunks to provide better context for answers
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ContextExpander:
    """
    Expands retrieved chunks with surrounding context

    For each retrieved chunk, also fetches:
    - Previous chunk (same document, previous position)
    - Next chunk (same document, next position)

    This provides better context for answer generation.
    """

    def __init__(self, vector_store, window_size: int = 1):
        """
        Initialize context expander

        Args:
            vector_store: VectorStore instance
            window_size: Number of chunks before/after to include (default: 1)
        """
        self.vector_store = vector_store
        self.window_size = window_size

    def expand_results(self, results: List[Dict], include_context: bool = True) -> List[Dict]:
        """
        Expand search results with surrounding context

        Args:
            results: List of search results with metadata
            include_context: Whether to include surrounding chunks

        Returns:
            List of expanded results with 'context_before' and 'context_after' keys
        """
        if not include_context:
            return results

        expanded_results = []

        for result in results:
            # Get surrounding chunks
            context_before, context_after = self._get_surrounding_chunks(result)

            expanded_result = {
                **result,
                'context_before': context_before,
                'context_after': context_after,
                'has_context': len(context_before) > 0 or len(context_after) > 0
            }

            expanded_results.append(expanded_result)

        return expanded_results

    def _get_surrounding_chunks(self, result: Dict) -> tuple:
        """
        Get surrounding chunks for a given result

        Args:
            result: Search result with metadata

        Returns:
            (context_before, context_after) - Lists of chunk texts
        """
        metadata = result.get('metadata', {})

        # Extract source and page information
        source = metadata.get('source', '')
        page = metadata.get('page', None)

        if not source:
            return [], []

        # Get all chunks from the collection
        try:
            # Query for chunks from the same source
            all_chunks = self._get_chunks_from_source(source, page)

            # Find current chunk position
            current_text = result.get('text', '')
            current_idx = self._find_chunk_index(all_chunks, current_text)

            if current_idx == -1:
                return [], []

            # Get surrounding chunks
            context_before = []
            context_after = []

            # Get previous chunks
            for i in range(1, self.window_size + 1):
                idx = current_idx - i
                if idx >= 0:
                    context_before.insert(0, all_chunks[idx])

            # Get next chunks
            for i in range(1, self.window_size + 1):
                idx = current_idx + i
                if idx < len(all_chunks):
                    context_after.append(all_chunks[idx])

            return context_before, context_after

        except Exception as e:
            logger.warning(f"Error getting surrounding chunks: {e}")
            return [], []

    def _get_chunks_from_source(self, source: str, page: Optional[int] = None) -> List[str]:
        """
        Get all chunks from a specific source (and optionally page)

        Args:
            source: Source document name
            page: Optional page number

        Returns:
            List of chunk texts in order
        """
        # This is a simplified version - in production, you'd want to:
        # 1. Store chunk position/index in metadata during ingestion
        # 2. Use that to efficiently retrieve surrounding chunks

        # For now, we'll use a workaround by querying the collection
        # Note: This requires ChromaDB to support metadata filtering

        try:
            # Get all chunks (limited to reasonable amount)
            where_filter = {"source": source}
            if page is not None:
                where_filter["page"] = page

            # ChromaDB get with where filter
            results = self.vector_store.collection.get(
                where=where_filter,
                limit=1000,  # Reasonable limit
                include=['documents', 'metadatas']
            )

            # Sort chunks by some ordering (if available)
            chunks = []
            if results and results.get('documents'):
                documents = results['documents']
                metadatas = results.get('metadatas', [])

                # Try to sort by chunk_id or position if available
                combined = list(zip(documents, metadatas))

                # Sort by chunk_id if available
                try:
                    combined.sort(key=lambda x: x[1].get('chunk_id', 0))
                except:
                    pass  # Keep original order if sorting fails

                chunks = [doc for doc, _ in combined]

            return chunks

        except Exception as e:
            logger.warning(f"Error querying chunks from source {source}: {e}")
            return []

    def _find_chunk_index(self, chunks: List[str], target_text: str) -> int:
        """
        Find the index of a chunk in a list

        Args:
            chunks: List of chunk texts
            target_text: Text to find

        Returns:
            Index of chunk, or -1 if not found
        """
        # Exact match first
        try:
            return chunks.index(target_text)
        except ValueError:
            pass

        # Fuzzy match (first 100 chars)
        target_prefix = target_text[:100]
        for i, chunk in enumerate(chunks):
            if chunk[:100] == target_prefix:
                return i

        return -1

    def format_with_context(self, result: Dict) -> str:
        """
        Format a result with its context as a single text block

        Args:
            result: Expanded result with context_before and context_after

        Returns:
            Formatted text with context markers
        """
        parts = []

        # Add context before
        context_before = result.get('context_before', [])
        if context_before:
            parts.append("--- CONTEXT BEFORE ---")
            for chunk in context_before:
                parts.append(chunk)

        # Add main chunk
        parts.append("--- MAIN CONTENT ---")
        parts.append(result.get('text', ''))

        # Add context after
        context_after = result.get('context_after', [])
        if context_after:
            parts.append("--- CONTEXT AFTER ---")
            for chunk in context_after:
                parts.append(chunk)

        return '\n\n'.join(parts)

    def merge_contexts_for_generation(self, results: List[Dict]) -> List[str]:
        """
        Merge main chunks with their contexts for LLM generation

        Args:
            results: List of expanded results

        Returns:
            List of context strings for LLM, with surrounding chunks integrated
        """
        merged_contexts = []

        for result in results:
            # Simple merge: context_before + main + context_after
            parts = []

            context_before = result.get('context_before', [])
            if context_before:
                # Take last chunk before (most relevant)
                parts.append(context_before[-1])

            # Main chunk
            parts.append(result.get('text', ''))

            context_after = result.get('context_after', [])
            if context_after:
                # Take first chunk after (most relevant)
                parts.append(context_after[0])

            merged_text = ' '.join(parts)
            merged_contexts.append(merged_text)

        return merged_contexts


class PositionAwareVectorStore:
    """
    Vector store that tracks chunk positions for context expansion

    This wrapper adds position tracking to chunk metadata during ingestion
    """

    def __init__(self, vector_store):
        """
        Initialize position-aware vector store

        Args:
            vector_store: Base VectorStore instance
        """
        self.vector_store = vector_store

    def add_documents_with_positions(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Add documents with position tracking

        Enhances metadata with:
        - chunk_position: Position within document
        - prev_chunk_id: ID of previous chunk (if exists)
        - next_chunk_id: ID of next chunk (if exists)
        """
        # Group chunks by source and page
        grouped_chunks = {}

        for idx, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', '')
            page = metadata.get('page', 0)

            key = (source, page)
            if key not in grouped_chunks:
                grouped_chunks[key] = []

            grouped_chunks[key].append((idx, chunk))

        # Add position information
        c_chunks = chunks.copy()

        for key, group in grouped_chunks.items():
            # Sort by existing chunk_id if available
            try:
                group.sort(key=lambda x: x[1].get('metadata', {}).get('chunk_id', 0))
            except:
                pass

            # Add position metadata
            for position, (original_idx, chunk) in enumerate(group):
                metadata = c_chunks[original_idx].get('metadata', {})

                metadata['chunk_position'] = position
                metadata['total_chunks_in_page'] = len(group)

                # Add prev/next chunk IDs
                if position > 0:
                    prev_idx = group[position - 1][0]
                    metadata['prev_chunk_idx'] = prev_idx

                if position < len(group) - 1:
                    next_idx = group[position + 1][0]
                    metadata['next_chunk_idx'] = next_idx

                c_chunks[original_idx]['metadata'] = metadata

        # Add to vector store
        self.vector_store.add_documents(c_chunks, embeddings)

    def get_chunk_by_index(self, chunk_idx: int) -> Optional[Dict]:
        """
        Get a chunk by its index

        This is a helper method for retrieving chunks by their stored index
        """
        # This would need to be implemented based on how you store chunk indices
        # For now, it's a placeholder
        pass

    def search_with_context(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        window_size: int = 1
    ) -> List[Dict]:
        """
        Search with automatic context expansion

        Args:
            query_embedding: Query vector
            top_k: Number of results
            window_size: Number of surrounding chunks to include

        Returns:
            List of results with context_before and context_after
        """
        # Perform base search
        results = self.vector_store.search(query_embedding, top_k)

        # Expand with context
        expander = ContextExpander(self.vector_store, window_size)

        # Convert results to list of dicts
        results_list = []
        for i in range(len(results['documents'][0])):
            results_list.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if results['distances'] else None,
                'similarity': self.vector_store.distance_to_similarity(results['distances'][0][i]) if results['distances'] else None
            })

        expanded_results = expander.expand_results(results_list)

        return expanded_results


def test_context_expander():
    """Test the context expander"""
    print("\n" + "="*60)
    print("Context Expander Module Test")
    print("="*60 + "\n")

    # Mock test data
    mock_result = {
        'text': 'This is the main chunk about diabetes treatment.',
        'metadata': {
            'source': 'diabetes_guide.pdf',
            'page': 5,
            'chunk_id': 2
        },
        'similarity': 0.85
    }

    print("Mock Result:")
    print(f"  Text: {mock_result['text']}")
    print(f"  Source: {mock_result['metadata']['source']}")
    print(f"  Page: {mock_result['metadata']['page']}")

    print("\nNote: Full testing requires an initialized VectorStore")
    print("Context expansion will retrieve ±1 chunks from the same document")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    test_context_expander()
