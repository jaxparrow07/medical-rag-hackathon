"""
Cross-Encoder Reranking for Medical RAG

This module implements cross-encoder reranking to improve precision
of retrieved contexts. Cross-encoders consider query-document interaction,
providing more accurate relevance scores than bi-encoder similarity alone.

Benefits:
- Higher precision (removes false positives from initial retrieval)
- Better ranking of retrieved documents
- Considers query-document interaction
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np

try:
    from .config import RETRIEVAL_CONFIG, DEBUG_CONFIG
except ImportError:
    from src.config import RETRIEVAL_CONFIG, DEBUG_CONFIG


class MedicalReranker:
    """
    Cross-encoder reranking for medical document retrieval

    Reranks initially retrieved documents using a cross-encoder model
    that processes query-document pairs jointly for better relevance scoring.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder for reranking

        Args:
            model_name: Cross-encoder model name (defaults to MS-MARCO)

        Recommended models:
        - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, 80MB, general)
        - 'cross-encoder/ms-marco-MedLM-L-12-v2' (medical-specific if available)
        - 'cross-encoder/nli-deberta-v3-base' (high quality, larger)
        """
        # Default to MS-MARCO if not specified
        if model_name is None:
            model_name = RETRIEVAL_CONFIG.get(
                'reranker_model',
                'cross-encoder/ms-marco-MiniLM-L-6-v2'
            )

        self.model_name = model_name

        try:
            self.reranker = CrossEncoder(model_name, max_length=512)

            if DEBUG_CONFIG.get('verbose', True):
                print(f"✅ Cross-encoder reranker initialized: {model_name}")
        except Exception as e:
            if DEBUG_CONFIG.get('verbose', True):
                print(f"⚠️  Could not load cross-encoder: {e}")
                print("   Reranking will be disabled")
            self.reranker = None

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 6
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Original user query
            documents: Retrieved documents from vector search
            top_k: Number of documents to return after reranking

        Returns:
            Top-k reranked documents, sorted by cross-encoder score

        Example:
            Initial retrieval: 10 documents (some may be false positives)
            Reranking: Score each with cross-encoder
            Output: Top-6 highest quality documents
        """
        if not documents:
            return []

        # If reranker not available, return original ranking
        if self.reranker is None:
            if DEBUG_CONFIG.get('verbose', True):
                print("⚠️  Reranker not available, returning original ranking")
            return documents[:top_k]

        # Create query-document pairs
        pairs = [[query, doc['text']] for doc in documents]

        # Score each pair
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)

            # Add rerank scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                # Keep original vector similarity for comparison
                doc['original_similarity'] = doc.get('similarity', 0)

            # Sort by rerank score (descending)
            reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

            # Log reranking results
            if DEBUG_CONFIG.get('log_retrievals', True):
                print(f"\n🔄 Cross-Encoder Reranking:")
                print(f"   Input: {len(documents)} documents")
                print(f"   Output: Top-{top_k} documents")
                print(f"\n   Reranked Results:")
                for i, doc in enumerate(reranked[:top_k]):
                    print(f"   {i+1}. Rerank: {doc['rerank_score']:.3f} | "
                          f"Original: {doc.get('original_similarity', 0):.3f} | "
                          f"{doc['text'][:60]}...")

            return reranked[:top_k]

        except Exception as e:
            if DEBUG_CONFIG.get('verbose', True):
                print(f"⚠️  Reranking failed: {e}")
                print("   Returning original ranking")
            return documents[:top_k]

    def rerank_with_score_fusion(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 6,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Rerank using weighted combination of vector similarity and cross-encoder score

        Args:
            query: Original user query
            documents: Retrieved documents
            top_k: Number of results to return
            alpha: Weight for cross-encoder score (1-alpha for vector similarity)

        Returns:
            Top-k documents ranked by fused score
        """
        if not documents or self.reranker is None:
            return documents[:top_k]

        # Get cross-encoder scores
        pairs = [[query, doc['text']] for doc in documents]
        cross_scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Normalize scores to [0, 1]
        vector_scores = np.array([doc.get('similarity', 0) for doc in documents])
        cross_scores_array = np.array(cross_scores)

        # Min-max normalization
        if cross_scores_array.max() > cross_scores_array.min():
            cross_norm = (cross_scores_array - cross_scores_array.min()) / \
                         (cross_scores_array.max() - cross_scores_array.min())
        else:
            cross_norm = cross_scores_array

        if vector_scores.max() > vector_scores.min():
            vector_norm = (vector_scores - vector_scores.min()) / \
                          (vector_scores.max() - vector_scores.min())
        else:
            vector_norm = vector_scores

        # Fused score
        fused_scores = alpha * cross_norm + (1 - alpha) * vector_norm

        # Add scores to documents
        for doc, cross_score, fused_score in zip(documents, cross_scores, fused_scores):
            doc['cross_encoder_score'] = float(cross_score)
            doc['vector_score'] = doc.get('similarity', 0)
            doc['fused_score'] = float(fused_score)

        # Sort by fused score
        reranked = sorted(documents, key=lambda x: x['fused_score'], reverse=True)

        if DEBUG_CONFIG.get('log_retrievals', True):
            print(f"\n🔀 Score Fusion Reranking (α={alpha}):")
            for i, doc in enumerate(reranked[:top_k]):
                print(f"   {i+1}. Fused: {doc['fused_score']:.3f} "
                      f"(Cross: {doc['cross_encoder_score']:.3f}, "
                      f"Vector: {doc['vector_score']:.3f})")

        return reranked[:top_k]


if __name__ == "__main__":
    # Test reranker
    print("Testing Medical Reranker\n")

    reranker = MedicalReranker()

    # Test query and documents
    query = "What causes diabetes?"

    documents = [
        {
            'text': "Diabetes mellitus is a metabolic disorder characterized by elevated blood glucose levels.",
            'similarity': 0.75
        },
        {
            'text': "The pancreas produces insulin, which regulates blood sugar levels.",
            'similarity': 0.70
        },
        {
            'text': "Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells.",
            'similarity': 0.68
        },
        {
            'text': "Cardiovascular disease is a common complication of diabetes.",
            'similarity': 0.65
        },
        {
            'text': "The weather today is sunny and warm.",
            'similarity': 0.40  # False positive
        }
    ]

    print(f"Query: {query}\n")
    print("Original ranking (by vector similarity):")
    for i, doc in enumerate(documents):
        print(f"{i+1}. Similarity: {doc['similarity']:.2f} - {doc['text'][:60]}...")

    # Rerank
    reranked = reranker.rerank(query, documents, top_k=3)

    print("\n" + "="*60)
    print("After cross-encoder reranking:")
    for i, doc in enumerate(reranked):
        print(f"{i+1}. Rerank Score: {doc['rerank_score']:.2f} - {doc['text'][:60]}...")
