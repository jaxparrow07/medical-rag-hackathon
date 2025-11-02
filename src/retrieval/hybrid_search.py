"""
Hybrid Search (BM25 + Vector) for Medical RAG

This module combines keyword-based search (BM25) with semantic vector search
to handle diverse query types, especially:
- Medical abbreviations (COPD, MI, ACE inhibitors)
- Drug names (metformin, atorvastatin)
- Specific medical terms that need exact matching

Benefits:
- BM25: Handles exact term matching (keywords, abbreviations)
- Vector: Handles semantic similarity (synonyms, paraphrasing)
- Hybrid: Best of both worlds
"""

from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Optional

try:
    from .config import DEBUG_CONFIG
except ImportError:
    from src.config import DEBUG_CONFIG


class HybridSearcher:
    """
    Hybrid search combining BM25 (keyword) and vector (semantic) search

    Process:
    1. BM25 search → Get keyword relevance scores
    2. Vector search → Get semantic similarity scores
    3. Normalize both scores to [0, 1]
    4. Combine with weighted sum: α*BM25 + (1-α)*Vector
    5. Return top-k results
    """

    def __init__(self, documents: List[Dict], vector_store, embedder):
        """
        Initialize hybrid search

        Args:
            documents: All documents in corpus (with 'text' and 'id' fields)
            vector_store: VectorStore instance for vector search
            embedder: EmbeddingModel instance for query embedding
        """
        self.documents = documents
        self.vector_store = vector_store
        self.embedder = embedder

        # Build BM25 index
        if DEBUG_CONFIG.get('verbose', True):
            print(f"🔨 Building BM25 index for {len(documents)} documents...")

        tokenized_docs = [self._tokenize(doc['text']) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Create document ID mapping
        self.doc_id_to_index = {}
        for i, doc in enumerate(documents):
            doc_id = doc.get('id') or hash(doc['text'][:100])
            self.doc_id_to_index[doc_id] = i

        if DEBUG_CONFIG.get('verbose', True):
            print(f"✅ BM25 index built successfully")

    def search(
        self,
        query: str,
        top_k: int = 6,
        alpha: float = 0.5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 + Vector

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for BM25 (0=pure vector, 1=pure BM25, 0.5=balanced)
            return_scores: Include individual scores in results

        Returns:
            Top-k documents ranked by hybrid score

        Recommended alpha values:
        - 0.7: For queries with medical abbreviations (COPD, MI)
        - 0.5: Balanced (default)
        - 0.3: For natural language questions
        """
        # 1. BM25 scores (keyword matching)
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # 2. Vector scores (semantic matching)
        query_embedding = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=len(self.documents)  # Get all for proper fusion
        )

        # Create score mapping by document ID
        vector_score_map = {}
        for result in vector_results:
            doc_id = result.get('id') or hash(result['text'][:100])
            vector_score_map[doc_id] = result.get('similarity', 0)

        # 3. Normalize scores to [0, 1]
        bm25_norm = self._normalize_scores(bm25_scores)

        # Get vector scores for all documents
        vector_scores = np.array([
            vector_score_map.get(doc.get('id') or hash(doc['text'][:100]), 0)
            for doc in self.documents
        ])
        vector_norm = self._normalize_scores(vector_scores)

        # 4. Combine scores
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * vector_norm

        # 5. Create results with scores
        results = []
        for i, doc in enumerate(self.documents):
            result = {
                **doc,
                'hybrid_score': float(hybrid_scores[i])
            }

            if return_scores:
                result['bm25_score'] = float(bm25_norm[i])
                result['vector_score'] = float(vector_norm[i])

            results.append(result)

        # 6. Sort by hybrid score and return top-k
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Log results
        if DEBUG_CONFIG.get('log_retrievals', True):
            print(f"\n🔍 Hybrid Search (α={alpha}):")
            print(f"   Query: {query}")
            print(f"   Results (top-{top_k}):")
            for i, result in enumerate(results[:top_k]):
                print(f"   {i+1}. Hybrid: {result['hybrid_score']:.3f} "
                      f"(BM25: {result.get('bm25_score', 0):.3f}, "
                      f"Vector: {result.get('vector_score', 0):.3f}) | "
                      f"{result['text'][:50]}...")

        return results[:top_k]

    def adaptive_search(self, query: str, top_k: int = 6) -> List[Dict]:
        """
        Adaptive hybrid search that automatically adjusts alpha based on query type

        Query type detection:
        - Contains abbreviations (COPD, MI, ACE) → Higher BM25 weight (α=0.7)
        - Contains drug names → Higher BM25 weight (α=0.7)
        - Natural language question → Balanced (α=0.5)
        - Complex semantic query → Higher vector weight (α=0.3)

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Top-k results with adaptive weighting
        """
        # Detect query type
        alpha = self._detect_optimal_alpha(query)

        if DEBUG_CONFIG.get('verbose', True):
            query_type = self._classify_query(query)
            print(f"📊 Adaptive Search: Detected '{query_type}' query, using α={alpha}")

        return self.search(query, top_k=top_k, alpha=alpha)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        # Simple tokenization (lowercase + split)
        # Could be improved with medical tokenizer
        return text.lower().split()

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] using min-max normalization"""
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return np.zeros_like(scores)

    def _detect_optimal_alpha(self, query: str) -> float:
        """
        Detect optimal alpha based on query characteristics

        Returns:
            Alpha value (0.0-1.0)
        """
        query_lower = query.lower()

        # Check for medical abbreviations (favor BM25)
        abbreviations = [
            'copd', 'mi', 'ace', 'ace inhibitor', 'chf', 'cva', 'uti', 'dm',
            'htn', 'bp', 'hr', 'ecg', 'ct', 'mri', 'icd', 'copd'
        ]
        if any(abbr in query_lower for abbr in abbreviations):
            return 0.7  # High BM25 weight for abbreviations

        # Check for drug names (favor BM25)
        drug_patterns = [
            'metformin', 'insulin', 'aspirin', 'statin', 'atorvastatin',
            'lisinopril', 'warfarin', 'heparin', 'penicillin', 'amoxicillin'
        ]
        if any(drug in query_lower for drug in drug_patterns):
            return 0.7  # High BM25 weight for drug names

        # Check for natural language questions (balanced)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(q in query_lower for q in question_words):
            return 0.5  # Balanced

        # Default: slightly favor vector for semantic queries
        return 0.4

    def _classify_query(self, query: str) -> str:
        """Classify query type for logging"""
        alpha = self._detect_optimal_alpha(query)

        if alpha >= 0.7:
            return "abbreviation/drug"
        elif alpha >= 0.5:
            return "natural language"
        else:
            return "semantic"


if __name__ == "__main__":
    # Test hybrid search
    print("Testing Hybrid Search\n")

    # Mock documents
    documents = [
        {
            'text': "COPD (Chronic Obstructive Pulmonary Disease) is a progressive lung disease.",
            'id': 1
        },
        {
            'text': "Chronic obstructive pulmonary disease causes breathing difficulty.",
            'id': 2
        },
        {
            'text': "Metformin is the first-line medication for type 2 diabetes.",
            'id': 3
        },
        {
            'text': "ACE inhibitors are used to treat high blood pressure.",
            'id': 4
        }
    ]

    # Test queries
    test_queries = [
        "COPD treatment",  # Abbreviation → High BM25
        "How to manage chronic lung disease?",  # Natural language → Balanced
        "What causes difficulty breathing?",  # Semantic → High vector
    ]

    print("Test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")

    # Note: Full hybrid search requires vector_store and embedder
    # This is just a structural test
