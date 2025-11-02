"""
Evaluation Metrics for RAG System
Tracks retrieval quality, answer faithfulness, and system performance
"""

import time
from typing import List, Dict, Optional
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class RAGEvaluator:
    """
    Comprehensive evaluation metrics for RAG system

    Tracks:
    - Retrieval metrics (MRR, Recall@K, NDCG)
    - Answer quality (faithfulness, relevance)
    - User feedback
    - Performance metrics (latency, throughput)
    """

    def __init__(self, save_results: bool = True, results_dir: str = "./data/evaluation"):
        """
        Initialize RAG evaluator

        Args:
            save_results: Whether to save evaluation results to disk
            results_dir: Directory to save results
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)

        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # Query log
        self.query_log = []

        # Feedback log
        self.feedback_log = []

        # Performance metrics
        self.performance_metrics = {
            'total_queries': 0,
            'total_latency': 0.0,
            'retrieval_latency': 0.0,
            'generation_latency': 0.0
        }

    def log_query(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: str,
        retrieval_time: float,
        generation_time: float,
        metadata: Optional[Dict] = None
    ):
        """
        Log a query and its results

        Args:
            query: User query
            retrieved_docs: Retrieved documents with scores
            answer: Generated answer
            retrieval_time: Time taken for retrieval (seconds)
            generation_time: Time taken for generation (seconds)
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'retrieval_scores': [d.get('similarity', 0) for d in retrieved_docs],
            'answer': answer,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            'metadata': metadata or {}
        }

        self.query_log.append(log_entry)

        # Update performance metrics
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['total_latency'] += log_entry['total_time']
        self.performance_metrics['retrieval_latency'] += retrieval_time
        self.performance_metrics['generation_latency'] += generation_time

    def log_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        feedback_text: Optional[str] = None
    ):
        """
        Log user feedback

        Args:
            query: Original query
            answer: Generated answer
            rating: User rating (1-5 or thumbs up/down as 0/1)
            feedback_text: Optional text feedback
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'rating': rating,
            'feedback_text': feedback_text
        }

        self.feedback_log.append(feedback_entry)

    def calculate_mrr(self, retrieved_docs: List[Dict], relevant_doc_id: str) -> float:
        """
        Calculate Mean Reciprocal Rank

        MRR = 1 / rank of first relevant document

        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_id: ID of the relevant document

        Returns:
            MRR score (0-1)
        """
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc.get('id') == relevant_doc_id or doc.get('metadata', {}).get('citation') == relevant_doc_id:
                return 1.0 / rank

        return 0.0

    def calculate_recall_at_k(
        self,
        retrieved_docs: List[Dict],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K

        Recall@K = (# relevant docs in top K) / (total # relevant docs)

        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0

        top_k_docs = retrieved_docs[:k]
        retrieved_ids = {
            doc.get('id') or doc.get('metadata', {}).get('citation')
            for doc in top_k_docs
        }

        relevant_retrieved = sum(
            1 for doc_id in relevant_doc_ids
            if doc_id in retrieved_ids
        )

        return relevant_retrieved / len(relevant_doc_ids)

    def calculate_ndcg(
        self,
        retrieved_docs: List[Dict],
        relevance_scores: List[float],
        k: int = 5
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain

        NDCG@K measures ranking quality considering position

        Args:
            retrieved_docs: List of retrieved documents
            relevance_scores: Ground truth relevance scores (same order as retrieved_docs)
            k: Number of top results to consider

        Returns:
            NDCG@K score (0-1)
        """
        if not relevance_scores or len(relevance_scores) != len(retrieved_docs):
            return 0.0

        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            rel = relevance_scores[i]
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because rank starts at 1

        # IDCG: Ideal DCG (if docs were perfectly ranked)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_scores))):
            rel = ideal_scores[i]
            idcg += (2**rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_precision_at_k(
        self,
        retrieved_docs: List[Dict],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K

        Precision@K = (# relevant docs in top K) / K

        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@K score (0-1)
        """
        top_k_docs = retrieved_docs[:k]
        retrieved_ids = {
            doc.get('id') or doc.get('metadata', {}).get('citation')
            for doc in top_k_docs
        }

        relevant_retrieved = sum(
            1 for doc_id in relevant_doc_ids
            if doc_id in retrieved_ids
        )

        return relevant_retrieved / k

    def check_answer_faithfulness(self, answer: str, contexts: List[str]) -> Dict:
        """
        Check if answer is faithful to retrieved contexts

        Simple heuristic-based approach:
        - Check for hallucination indicators ("I believe", "probably", etc.)
        - Check if answer content appears in contexts

        Args:
            answer: Generated answer
            contexts: Retrieved context texts

        Returns:
            {
                'is_faithful': bool,
                'confidence': float,
                'issues': List[str]
            }
        """
        issues = []

        # Check for uncertainty phrases (may indicate hallucination)
        uncertainty_phrases = [
            'i believe', 'i think', 'probably', 'maybe', 'might be',
            'could be', 'not sure', 'unclear', 'cannot determine'
        ]

        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                issues.append(f"Contains uncertainty phrase: '{phrase}'")

        # Check if answer mentions sources that weren't retrieved
        # (Simple check: answer shouldn't contain "source X" citations)
        if 'source ' in answer_lower and any(f'source {i}' in answer_lower for i in range(1, 10)):
            issues.append("Answer contains source citations (should be removed)")

        # Check for content overlap
        # Simple heuristic: significant words in answer should appear in contexts
        combined_context = ' '.join(contexts).lower()
        answer_words = set(answer_lower.split())

        # Filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        significant_words = {w for w in answer_words if len(w) > 4 and w not in stopwords}

        if significant_words:
            words_in_context = sum(1 for w in significant_words if w in combined_context)
            overlap_ratio = words_in_context / len(significant_words)

            if overlap_ratio < 0.3:
                issues.append(f"Low content overlap with contexts ({overlap_ratio:.2%})")

        # Determine faithfulness
        is_faithful = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.2)  # Reduce confidence for each issue

        return {
            'is_faithful': is_faithful,
            'confidence': max(0.0, confidence),
            'issues': issues
        }

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary statistics

        Returns:
            Dictionary of performance metrics
        """
        total = self.performance_metrics['total_queries']

        if total == 0:
            return {
                'total_queries': 0,
                'avg_total_latency': 0.0,
                'avg_retrieval_latency': 0.0,
                'avg_generation_latency': 0.0
            }

        return {
            'total_queries': total,
            'avg_total_latency': self.performance_metrics['total_latency'] / total,
            'avg_retrieval_latency': self.performance_metrics['retrieval_latency'] / total,
            'avg_generation_latency': self.performance_metrics['generation_latency'] / total,
            'queries_per_second': total / self.performance_metrics['total_latency'] if self.performance_metrics['total_latency'] > 0 else 0
        }

    def get_feedback_summary(self) -> Dict:
        """
        Get user feedback summary

        Returns:
            Dictionary of feedback statistics
        """
        if not self.feedback_log:
            return {
                'total_feedback': 0,
                'avg_rating': 0.0,
                'positive_feedback_rate': 0.0
            }

        ratings = [f['rating'] for f in self.feedback_log]

        return {
            'total_feedback': len(self.feedback_log),
            'avg_rating': np.mean(ratings),
            'positive_feedback_rate': sum(1 for r in ratings if r >= 4) / len(ratings),
            'negative_feedback_rate': sum(1 for r in ratings if r <= 2) / len(ratings)
        }

    def save_evaluation_report(self, filename: Optional[str] = None):
        """
        Save evaluation report to disk

        Args:
            filename: Optional filename (default: evaluation_TIMESTAMP.json)
        """
        if not self.save_results:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': self.get_performance_summary(),
            'feedback': self.get_feedback_summary(),
            'total_queries_logged': len(self.query_log),
            'recent_queries': self.query_log[-10:]  # Last 10 queries
        }

        report_path = self.results_dir / filename

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Evaluation report saved to: {report_path}")

    def generate_report(self) -> str:
        """
        Generate a human-readable evaluation report

        Returns:
            Formatted report string
        """
        perf = self.get_performance_summary()
        feedback = self.get_feedback_summary()

        report = f"""
{'='*80}
RAG System Evaluation Report
{'='*80}

PERFORMANCE METRICS
-------------------
Total Queries: {perf['total_queries']}
Avg Total Latency: {perf['avg_total_latency']:.3f}s
Avg Retrieval Latency: {perf['avg_retrieval_latency']:.3f}s
Avg Generation Latency: {perf['avg_generation_latency']:.3f}s
Queries/Second: {perf['queries_per_second']:.2f}

USER FEEDBACK
-------------
Total Feedback: {feedback['total_feedback']}
Avg Rating: {feedback['avg_rating']:.2f}/5.0
Positive Rate: {feedback.get('positive_feedback_rate', 0):.1%}
Negative Rate: {feedback.get('negative_feedback_rate', 0):.1%}

RECENT QUERIES
--------------
"""
        for i, query in enumerate(self.query_log[-5:], 1):
            report += f"\n{i}. Query: {query['query'][:60]}..."
            report += f"\n   Retrieved: {query['num_retrieved']} docs"
            report += f"\n   Total Time: {query['total_time']:.3f}s\n"

        report += f"\n{'='*80}\n"

        return report


def test_evaluator():
    """Test the evaluation module"""
    print("\n" + "="*80)
    print("RAG Evaluator Test")
    print("="*80 + "\n")

    evaluator = RAGEvaluator(save_results=False)

    # Simulate some queries
    mock_retrieved = [
        {'id': 'doc1', 'text': 'Diabetes is a metabolic disorder', 'similarity': 0.92},
        {'id': 'doc2', 'text': 'Treatment includes insulin', 'similarity': 0.85},
        {'id': 'doc3', 'text': 'Risk factors include obesity', 'similarity': 0.78},
    ]

    mock_answer = "Diabetes is a metabolic disorder characterized by high blood sugar."

    # Log query
    evaluator.log_query(
        query="What is diabetes?",
        retrieved_docs=mock_retrieved,
        answer=mock_answer,
        retrieval_time=0.123,
        generation_time=0.456
    )

    # Test faithfulness
    contexts = [d['text'] for d in mock_retrieved]
    faithfulness = evaluator.check_answer_faithfulness(mock_answer, contexts)

    print(f"Faithfulness Check:")
    print(f"  Is Faithful: {faithfulness['is_faithful']}")
    print(f"  Confidence: {faithfulness['confidence']:.2f}")
    if faithfulness['issues']:
        print(f"  Issues: {faithfulness['issues']}")

    # Test retrieval metrics
    print(f"\nRetrieval Metrics:")

    mrr = evaluator.calculate_mrr(mock_retrieved, 'doc1')
    print(f"  MRR: {mrr:.3f}")

    recall = evaluator.calculate_recall_at_k(mock_retrieved, ['doc1', 'doc2'], k=5)
    print(f"  Recall@5: {recall:.3f}")

    precision = evaluator.calculate_precision_at_k(mock_retrieved, ['doc1', 'doc2'], k=3)
    print(f"  Precision@3: {precision:.3f}")

    # Generate report
    print("\n" + evaluator.generate_report())


if __name__ == '__main__':
    test_evaluator()
