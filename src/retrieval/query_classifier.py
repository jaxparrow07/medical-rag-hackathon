"""
Query Classification Module
Classifies queries to route them to optimal retrieval strategies
"""

import re
from typing import Dict, List, Literal
from enum import Enum


class QueryType(Enum):
    """Types of medical queries"""
    DEFINITION = "definition"  # "What is X?"
    SYMPTOM = "symptom"  # "What are symptoms of X?"
    TREATMENT = "treatment"  # "How to treat X?"
    DIAGNOSIS = "diagnosis"  # "How to diagnose X?"
    DRUG_INFO = "drug_info"  # Questions about specific drugs
    ABBREVIATION = "abbreviation"  # Questions with medical abbreviations
    PROCEDURAL = "procedural"  # "How to perform X?"
    COMPARISON = "comparison"  # "X vs Y"
    MECHANISM = "mechanism"  # "How does X work?"
    GENERAL = "general"  # General medical knowledge


class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    SEMANTIC = "semantic"  # Pure vector search
    HYBRID = "hybrid"  # BM25 + Vector
    MULTI_QUERY = "multi_query"  # Query reformulation + vector
    CONTEXTUAL = "contextual"  # Vector + context expansion


class QueryClassifier:
    """
    Classifies medical queries and recommends optimal retrieval strategy

    Routes queries to the best retrieval method based on query characteristics
    """

    def __init__(self):
        """Initialize query classifier"""
        # Medical abbreviations that should use hybrid search
        self.medical_abbreviations = {
            'MI', 'COPD', 'HTN', 'DM', 'CHF', 'CVA', 'PE', 'DVT', 'UTI',
            'CAD', 'ACE', 'ARB', 'NSAID', 'CBC', 'CXR', 'CT', 'MRI',
            'ECG', 'EKG', 'BP', 'HR', 'RR', 'ABG', 'BUN', 'GFR',
            'ALT', 'AST', 'LDL', 'HDL', 'TSH', 'T3', 'T4', 'HbA1c'
        }

        # Common drug name patterns
        self.drug_suffixes = [
            'cillin', 'statin', 'prazole', 'olol', 'sartan', 'pril',
            'mycin', 'cycline', 'floxacin', 'azole', 'tidine'
        ]

    def classify(self, query: str) -> Dict:
        """
        Classify a query and recommend retrieval strategy

        Args:
            query: User query text

        Returns:
            {
                'query_type': QueryType,
                'recommended_strategy': RetrievalStrategy,
                'confidence': float,
                'features': Dict of detected features,
                'reasoning': str explanation
            }
        """
        query_lower = query.lower()
        features = self._extract_features(query)

        # Classify query type
        query_type = self._classify_query_type(query_lower, features)

        # Recommend strategy based on type and features
        strategy, confidence, reasoning = self._recommend_strategy(query_type, features)

        return {
            'query_type': query_type,
            'recommended_strategy': strategy,
            'confidence': confidence,
            'features': features,
            'reasoning': reasoning
        }

    def _extract_features(self, query: str) -> Dict:
        """
        Extract features from query

        Returns:
            Dictionary of detected features
        """
        query_lower = query.lower()

        # Check for medical abbreviations
        has_abbreviations = any(
            re.search(r'\b' + abbr + r'\b', query)
            for abbr in self.medical_abbreviations
        )

        # Check for drug names
        has_drug_names = any(
            suffix in query_lower
            for suffix in self.drug_suffixes
        )

        # Check for specific patterns
        is_definition = bool(re.match(r'what is |define |definition of ', query_lower))
        is_how_to = bool(re.match(r'how to |how do |how can ', query_lower))
        is_symptom_query = 'symptom' in query_lower or 'signs of' in query_lower
        is_treatment_query = 'treatment' in query_lower or 'treat ' in query_lower or 'therapy' in query_lower
        is_diagnosis_query = 'diagnose' in query_lower or 'diagnosis' in query_lower
        is_comparison = ' vs ' in query_lower or ' versus ' in query_lower or ' compared to ' in query_lower
        is_mechanism = 'mechanism' in query_lower or 'how does' in query_lower or 'why does' in query_lower

        # Query length
        word_count = len(query.split())
        is_short = word_count <= 5
        is_long = word_count > 15

        # Check for specific medical entities
        has_disease_terms = any(
            term in query_lower for term in [
                'disease', 'syndrome', 'disorder', 'condition',
                'infection', 'cancer', 'tumor'
            ]
        )

        return {
            'has_abbreviations': has_abbreviations,
            'has_drug_names': has_drug_names,
            'is_definition': is_definition,
            'is_how_to': is_how_to,
            'is_symptom_query': is_symptom_query,
            'is_treatment_query': is_treatment_query,
            'is_diagnosis_query': is_diagnosis_query,
            'is_comparison': is_comparison,
            'is_mechanism': is_mechanism,
            'is_short': is_short,
            'is_long': is_long,
            'word_count': word_count,
            'has_disease_terms': has_disease_terms
        }

    def _classify_query_type(self, query_lower: str, features: Dict) -> QueryType:
        """
        Classify the type of query

        Args:
            query_lower: Lowercase query
            features: Extracted features

        Returns:
            QueryType enum
        """
        # Rule-based classification
        if features['is_definition']:
            return QueryType.DEFINITION

        if features['is_symptom_query']:
            return QueryType.SYMPTOM

        if features['is_treatment_query']:
            return QueryType.TREATMENT

        if features['is_diagnosis_query']:
            return QueryType.DIAGNOSIS

        if features['has_drug_names']:
            return QueryType.DRUG_INFO

        if features['has_abbreviations']:
            return QueryType.ABBREVIATION

        if features['is_how_to']:
            return QueryType.PROCEDURAL

        if features['is_comparison']:
            return QueryType.COMPARISON

        if features['is_mechanism']:
            return QueryType.MECHANISM

        return QueryType.GENERAL

    def _recommend_strategy(self, query_type: QueryType, features: Dict) -> tuple:
        """
        Recommend retrieval strategy based on query type and features

        Args:
            query_type: Classified query type
            features: Extracted features

        Returns:
            (strategy, confidence, reasoning)
        """
        # Strategy recommendations based on query type
        if query_type == QueryType.ABBREVIATION or features['has_abbreviations']:
            return (
                RetrievalStrategy.HYBRID,
                0.9,
                "Abbreviations benefit from keyword matching (BM25) + semantic search"
            )

        if query_type == QueryType.DRUG_INFO or features['has_drug_names']:
            return (
                RetrievalStrategy.HYBRID,
                0.85,
                "Drug names require exact keyword matching combined with semantic understanding"
            )

        if query_type == QueryType.DEFINITION:
            return (
                RetrievalStrategy.SEMANTIC,
                0.8,
                "Definition queries work well with pure semantic search"
            )

        if query_type == QueryType.COMPARISON:
            return (
                RetrievalStrategy.MULTI_QUERY,
                0.85,
                "Comparisons benefit from reformulating to search for both concepts separately"
            )

        if query_type in [QueryType.MECHANISM, QueryType.PROCEDURAL]:
            return (
                RetrievalStrategy.CONTEXTUAL,
                0.8,
                "Mechanism/procedural queries need surrounding context for complete understanding"
            )

        if query_type in [QueryType.TREATMENT, QueryType.DIAGNOSIS]:
            return (
                RetrievalStrategy.MULTI_QUERY,
                0.75,
                "Treatment/diagnosis queries benefit from multiple query perspectives"
            )

        if features['is_long']:
            return (
                RetrievalStrategy.MULTI_QUERY,
                0.7,
                "Long queries benefit from reformulation to extract key concepts"
            )

        if features['is_short']:
            return (
                RetrievalStrategy.HYBRID,
                0.7,
                "Short queries need both keyword precision and semantic expansion"
            )

        # Default to multi-query for general queries
        return (
            RetrievalStrategy.MULTI_QUERY,
            0.6,
            "General queries use multi-query reformulation for comprehensive coverage"
        )

    def get_hybrid_alpha(self, query_type: QueryType, features: Dict) -> float:
        """
        Get recommended BM25 weight (alpha) for hybrid search

        Args:
            query_type: Query type
            features: Query features

        Returns:
            Alpha value (0.0-1.0)
            - 0.0 = pure vector
            - 0.5 = balanced
            - 1.0 = pure BM25
        """
        # Higher alpha = more BM25 weight
        if query_type == QueryType.ABBREVIATION or features['has_abbreviations']:
            return 0.7  # Favor keyword matching

        if query_type == QueryType.DRUG_INFO or features['has_drug_names']:
            return 0.7  # Favor exact matches

        if query_type == QueryType.DEFINITION:
            return 0.3  # Favor semantic

        if features['is_short']:
            return 0.6  # More keyword weight

        # Default balanced
        return 0.5

    def should_expand_context(self, query_type: QueryType, features: Dict) -> bool:
        """
        Determine if context expansion should be used

        Args:
            query_type: Query type
            features: Query features

        Returns:
            True if context expansion recommended
        """
        # Queries that benefit from surrounding context
        context_beneficial_types = {
            QueryType.MECHANISM,
            QueryType.PROCEDURAL,
            QueryType.TREATMENT,
            QueryType.DIAGNOSIS
        }

        return query_type in context_beneficial_types or features['is_long']

    def should_reformulate(self, query_type: QueryType, features: Dict) -> bool:
        """
        Determine if query reformulation should be used

        Args:
            query_type: Query type
            features: Query features

        Returns:
            True if reformulation recommended
        """
        # Most queries benefit from reformulation except very specific ones
        no_reformulation_types = {
            QueryType.ABBREVIATION,  # Already specific
        }

        if query_type in no_reformulation_types:
            return False

        # Short drug name queries don't need reformulation
        if features['is_short'] and features['has_drug_names']:
            return False

        return True


def test_query_classifier():
    """Test the query classifier"""
    classifier = QueryClassifier()

    test_queries = [
        "What is myocardial infarction?",
        "How to treat COPD?",
        "Symptoms of diabetes",
        "Aspirin vs clopidogrel for MI prevention",
        "How does metformin work?",
        "MI treatment guidelines",
        "What are the side effects of atorvastatin?",
        "Diagnosis of pulmonary embolism",
        "ACE inhibitors mechanism of action",
        "Difference between Type 1 and Type 2 diabetes mellitus and their pathophysiology"
    ]

    print("\n" + "="*80)
    print("Query Classification Test")
    print("="*80 + "\n")

    for query in test_queries:
        result = classifier.classify(query)

        print(f"Query: {query}")
        print(f"  Type: {result['query_type'].value}")
        print(f"  Strategy: {result['recommended_strategy'].value}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reasoning: {result['reasoning']}")

        if result['recommended_strategy'] == RetrievalStrategy.HYBRID:
            alpha = classifier.get_hybrid_alpha(result['query_type'], result['features'])
            print(f"  Hybrid Alpha: {alpha:.2f} (BM25 weight)")

        expand_context = classifier.should_expand_context(result['query_type'], result['features'])
        print(f"  Expand Context: {expand_context}")

        reformulate = classifier.should_reformulate(result['query_type'], result['features'])
        print(f"  Reformulate: {reformulate}")

        print()

    print("="*80 + "\n")


if __name__ == '__main__':
    test_query_classifier()
