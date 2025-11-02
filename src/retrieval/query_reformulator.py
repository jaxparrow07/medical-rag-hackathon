"""
MiniMax-powered Query Reformulation for Medical RAG

This module handles dynamic query reformulation using MiniMax LLM via OpenRouter
to bridge the semantic gap between layman terminology and medical jargon.

Example:
    "What causes a lump in the breast?"
    → "What is the etiology of breast masses and neoplasms?"
    → "Breast tumor mass nodule causes and risk factors"
"""

from openai import OpenAI
from typing import List, Dict, Optional
import os

try:
    from .config import DEBUG_CONFIG, QUERY_REFORMULATION_CONFIG
except ImportError:
    from src.config import DEBUG_CONFIG, QUERY_REFORMULATION_CONFIG


class QueryReformulator:
    """
    Uses MiniMax LLM (via OpenRouter) to dynamically reformulate medical queries

    Benefits over static dictionaries:
    - Adapts to any medical terminology
    - Context-aware reformulation
    - Handles complex multi-term queries
    - No maintenance of static synonym lists
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize MiniMax client for query reformulation

        Args:
            api_key: OpenRouter API key (uses env var if None)
            model: Model to use (uses config default if None)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please set it in .env file or pass as parameter."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/jaxparrow07/rag-model",
                "X-Title": "Medical RAG Query Reformulation"
            }
        )

        # Use config default or provided model
        self.model = model or QUERY_REFORMULATION_CONFIG.get('model', 'minimax/minimax-01')

        if DEBUG_CONFIG.get('verbose', True):
            print(f"✅ Query Reformulator initialized with model: {self.model}")

    def reformulate_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple medical query variants using MiniMax

        Args:
            query: Original user query (may use layman terminology)
            num_variants: Number of reformulated variants to generate

        Returns:
            List of query strings: [original, medical_version, comprehensive_version, ...]

        Example:
            Input: "What causes a lump in the breast?"
            Output: [
                "What causes a lump in the breast?",
                "What is the etiology of breast masses and neoplasms?",
                "Breast tumor mass nodule lump causes risk factors",
                "Pathophysiology of palpable breast lesions"
            ]
        """

        prompt = self._build_reformulation_prompt(query, num_variants)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical information specialist expert at reformulating patient questions into optimal medical literature search queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=QUERY_REFORMULATION_CONFIG.get('temperature', 0.4),
                max_tokens=300
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            variants = self._parse_reformulated_queries(content)

            # Always include original query first
            all_queries = [query] + variants

            # Log reformulation
            if DEBUG_CONFIG.get('verbose', True):
                print(f"\n🔄 Query Reformulation:")
                print(f"   Original: {query}")
                for i, variant in enumerate(variants, 1):
                    print(f"   Variant {i}: {variant}")

            return all_queries

        except Exception as e:
            print(f"⚠️  Query reformulation failed: {e}")
            print(f"   Falling back to original query only")
            return [query]  # Fallback to original query

    def reformulate_with_reasoning(self, query: str) -> Dict[str, any]:
        """
        Use MiniMax's reasoning capabilities to understand query intent
        and generate targeted reformulations with explanations

        Args:
            query: Original user query

        Returns:
            Dict with original query, variants, and reasoning

        Example:
            {
                'original': "What causes a lump in the breast?",
                'variants': {
                    'medical': "What is the etiology of breast masses?",
                    'comprehensive': "Breast tumor mass nodule causes",
                    'focused': "Breast neoplasm pathophysiology"
                },
                'reasoning': "Query is asking about breast lesions...",
                'all_queries': [original, medical, comprehensive, focused]
            }
        """

        prompt = f"""Analyze this medical question and reformulate it for optimal medical database retrieval.

Question: "{query}"

Think through:
1. What medical concepts/entities are mentioned? (diseases, symptoms, anatomy, drugs)
2. Is this layman terminology or medical terminology?
3. What are the medical synonyms for key terms?
4. What related concepts should be included for comprehensive retrieval?
5. Expand the abbreviations if any.
6. Include searchable queries since this will be searched in a vector database of medical literature.

Then provide:
REASONING: [Your analysis of the query]
MEDICAL: [Medical terminology version]
COMPREHENSIVE: [Version with all relevant synonyms]
FOCUSED: [Targeted version for specific medical literature]

Keep each reformulated query concise (1-2 sentences)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical information specialist with deep knowledge of clinical terminology and medical literature indexing."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            content = response.choices[0].message.content

            # Parse structured output
            variants = {}
            reasoning = ""

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('MEDICAL:'):
                    variants['medical'] = line.replace('MEDICAL:', '').strip()
                elif line.startswith('COMPREHENSIVE:'):
                    variants['comprehensive'] = line.replace('COMPREHENSIVE:', '').strip()
                elif line.startswith('FOCUSED:'):
                    variants['focused'] = line.replace('FOCUSED:', '').strip()

            all_queries = [query] + list(variants.values())

            if DEBUG_CONFIG.get('verbose', True):
                print(f"\n🧠 Query Reasoning:")
                print(f"   {reasoning}")
                print(f"\n🔄 Reformulated Queries:")
                for key, variant in variants.items():
                    print(f"   {key.upper()}: {variant}")

            return {
                'original': query,
                'variants': variants,
                'reasoning': reasoning,
                'all_queries': all_queries
            }

        except Exception as e:
            print(f"⚠️  Reasoning reformulation failed: {e}")
            return {
                'original': query,
                'variants': {},
                'reasoning': '',
                'all_queries': [query]
            }

    def _build_reformulation_prompt(self, query: str, num_variants: int) -> str:
        """Build the reformulation prompt"""

        return f"""You are a medical information specialist. Given a medical question, generate {num_variants} reformulated versions to improve medical literature search.

Original Question: "{query}"

Generate {num_variants} reformulated versions:
1. **Medical Terminology Version**: Rewrite using clinical/medical terminology. Convert layman terms to medical terms:
   - "lump" → "mass", "neoplasm", "tumor", "nodule"
   - "heart attack" → "myocardial infarction", "MI", "acute coronary syndrome"
   - "pain" → "discomfort", "tenderness", "ache"
   - "high blood sugar" → "hyperglycemia", "elevated glucose"

2. **Expanded Synonym Version**: Include medical synonyms and related concepts as keywords (e.g., "heart attack myocardial infarction MI cardiac event acute coronary syndrome")

3. **Anatomical/Pathophysiology Version**: Focus on underlying mechanisms and anatomical terms (e.g., "cardiac ischemia pathophysiology")

Rules:
- Keep questions concise (1-2 sentences each)
- Only output the reformulated questions, one per line
- No numbering, no explanations, no extra text
- Preserve the core intent of the original question
- Output only the reformulated queries

Reformulated Questions:"""

    def _parse_reformulated_queries(self, content: str) -> List[str]:
        """Parse LLM response into list of queries"""

        # Split by newlines and clean
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Remove any numbering (1., 2., etc.)
        queries = []
        for line in lines:
            # Remove common prefixes
            cleaned = line
            for prefix in ['1.', '2.', '3.', '1)', '2)', '3)', '-', '*', '•']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()

            # Remove markdown bold
            cleaned = cleaned.replace('**', '')

            # Skip empty or very short lines
            if len(cleaned) > 10:
                queries.append(cleaned)

        return queries


class MultiQueryRetrieval:
    """
    Retrieve contexts for multiple query variants and merge results

    This class orchestrates:
    1. Query reformulation (via MiniMax)
    2. Embedding multiple query variants
    3. Vector search for each variant
    4. Merging and deduplicating results
    """

    def __init__(self, reformulator: QueryReformulator, vector_store, embedder):
        """
        Initialize multi-query retrieval

        Args:
            reformulator: QueryReformulator instance
            vector_store: VectorStore instance
            embedder: EmbeddingModel instance
        """
        self.reformulator = reformulator
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        top_k_per_query: int = 10,
        use_reasoning: bool = False
    ) -> List[Dict]:
        """
        Generate query variants and retrieve for each, then merge results

        Args:
            query: Original user query
            top_k: Final number of contexts to return
            top_k_per_query: Number of results to retrieve per query variant
            use_reasoning: Use reasoning-based reformulation (slower but better)

        Returns:
            List of top-k unique contexts, sorted by similarity

        Process:
            1. Query → MiniMax → [Q1, Q2, Q3, Q4]
            2. Each Q → Embedding → Vector
            3. Each Vector → Search → Top-N results
            4. Merge all results, deduplicate, sort by score
            5. Return top-k
        """

        # Generate query variants
        if use_reasoning:
            result = self.reformulator.reformulate_with_reasoning(query)
            query_variants = result['all_queries']
        else:
            query_variants = self.reformulator.reformulate_query(query, num_variants=3)

        # Retrieve for each variant
        all_results = []
        seen_texts = set()  # Deduplication based on text hash

        for variant in query_variants:
            # Embed variant query
            query_embedding = self.embedder.encode_query(variant)

            # Retrieve contexts
            search_results = self.vector_store.search(query_embedding, top_k=top_k_per_query)

            # Convert search results to list of dicts (handle Chroma format)
            if search_results['documents'] and len(search_results['documents']) > 0:
                for doc, meta, dist in zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
                ):
                    # Create hash from first 100 characters for deduplication
                    text_snippet = doc[:100] if len(doc) >= 100 else doc
                    text_hash = hash(text_snippet)

                    if text_hash not in seen_texts:
                        seen_texts.add(text_hash)
                        # Convert ChromaDB cosine distance to similarity
                        # ChromaDB uses: distance = 1 - cosine_similarity
                        # So: similarity = 1 - distance
                        similarity = self.vector_store.distance_to_similarity(dist)

                        all_results.append({
                            'text': doc,
                            'metadata': meta,
                            'similarity': similarity,
                            'source_query': variant  # Track which variant found this
                        })

        # Sort by similarity score and return top-k
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        if DEBUG_CONFIG.get('log_retrievals', True):
            print(f"\n📚 Multi-Query Retrieval Results:")
            print(f"   Total unique documents: {len(all_results)}")
            print(f"   Returning top-{top_k}")
            for i, result in enumerate(all_results[:top_k]):
                print(f"   {i+1}. Similarity: {result.get('similarity', 0):.3f} | Source: {result.get('source_query', 'unknown')}...")

        return all_results[:top_k]


if __name__ == "__main__":
    # Test the reformulator
    print("Testing Query Reformulator\n")

    reformulator = QueryReformulator()

    test_queries = [
        "What causes a lump in the breast?",
        "How to treat high blood pressure?",
        "What are the symptoms of a heart attack?",
        "Why do I have pain in my chest?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print('='*60)

        # Simple reformulation
        variants = reformulator.reformulate_query(query)

        # Reasoning-based reformulation
        # result = reformulator.reformulate_with_reasoning(query)
