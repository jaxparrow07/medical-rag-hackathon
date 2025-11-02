"""
Main file to test the RAG system with queries
Run this after setting up the database with setup_database.py
"""

import os
from dotenv import load_dotenv
from src.retrieval import LocalEmbedder
from src.retrieval import VectorStore
from src.retrieval import QueryProcessor
from src.generation import CitationGenerator
from src.retrieval import QueryReformulator, MultiQueryRetrieval
from src.retrieval import MedicalReranker
from src.config import (
    QUERY_REFORMULATION_CONFIG,
    RETRIEVAL_CONFIG,
    EMBEDDING_CONFIG,
    DEBUG_CONFIG
)

# Load environment variables
load_dotenv()

def query_rag_system(query: str, top_k: int = None, use_reformulation: bool = None, use_reranking: bool = None):
    """
    Query the RAG system with pipeline

    Args:
        query: User question
        top_k: Number of results (uses config default if None)
        use_reformulation: Enable MiniMax query reformulation (uses config if None)
        use_reranking: Enable cross-encoder reranking (uses config if None)

    Returns:
        dict with answer, contexts, and metadata
    """

    # Use config defaults if not specified
    if top_k is None:
        top_k = RETRIEVAL_CONFIG.get('final_top_k', 6)
    if use_reformulation is None:
        use_reformulation = QUERY_REFORMULATION_CONFIG.get('enabled', True)
    if use_reranking is None:
        use_reranking = RETRIEVAL_CONFIG.get('rerank', True)

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # 1. Initialize components
    print("🔧 Loading components...")

    # Embedding model (uses MedCPT from config)
    embedder = LocalEmbedder()
    vector_store = VectorStore(persist_dir="./data/vectordb")

    # Optional: Query reformulation with MiniMax
    if use_reformulation:
        try:
            reformulator = QueryReformulator(
                model=QUERY_REFORMULATION_CONFIG['model']
            )
            multi_retriever = MultiQueryRetrieval(reformulator, vector_store, embedder)
            print("✅ MiniMax query reformulation enabled")
        except Exception as e:
            print(f"⚠️  Query reformulation not available: {e}")
            print("   Falling back to single-query retrieval")
            use_reformulation = False
    else:
        print("ℹ️  Query reformulation disabled")

    # Optional: Cross-encoder reranking
    if use_reranking:
        try:
            reranker = MedicalReranker()
            print("✅ Cross-encoder reranking enabled")
        except Exception as e:
            print(f"⚠️  Reranking not available: {e}")
            use_reranking = False
    else:
        print("ℹ️  Reranking disabled")

    # LLM for answer generation (DeepSeek R1 via OpenRouter)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found! Set it in .env")

    citation_gen = CitationGenerator(api_key=api_key, provider="openrouter")

    # 2. Query Processing & Retrieval
    print("\n📚 Retrieving relevant contexts...")

    if use_reformulation and 'multi_retriever' in locals():
        # Multi-query retrieval with MiniMax reformulation
        top_k_per_query = RETRIEVAL_CONFIG.get('top_k_per_query', 10)
        contexts = multi_retriever.retrieve(
            query,
            top_k=top_k * 2,  # Get more for reranking
            top_k_per_query=top_k_per_query
        )
    else:
        # Single-query retrieval (legacy)
        query_embedding = embedder.embed_query(query)
        search_results = vector_store.search(query_embedding, top_k * 2)

        contexts = []
        if search_results['documents'] and len(search_results['documents']) > 0:
            for doc, meta, dist in zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            ):
                # Use proper distance-to-similarity conversion
                similarity = vector_store.distance_to_similarity(dist)
                contexts.append({
                    'text': doc,
                    'metadata': meta,
                    'similarity': similarity
                })

    print(f"   Retrieved {len(contexts)} initial results")

    # 3. Reranking (optional)
    if use_reranking and len(contexts) > 0 and 'reranker' in locals():
        print("\n🔄 Reranking results with cross-encoder...")
        contexts = reranker.rerank(query, contexts, top_k=top_k)
        print(f"   Reranked to top-{top_k} results")
    else:
        # Just take top-k without reranking
        contexts = contexts[:top_k]

    # Filter by similarity threshold
    similarity_threshold = RETRIEVAL_CONFIG.get('similarity_threshold', 0.4)

    # Debug: Print similarity scores before filtering
    if DEBUG_CONFIG.get('verbose', True):
        print(f"\n📊 Similarity scores before filtering (threshold: {similarity_threshold}):")
        for i, ctx in enumerate(contexts[:10], 1):  # Show first 10
            sim = ctx.get('similarity', 0)
            print(f"   {i}. Similarity: {sim:.4f}")

    filtered_contexts = [c for c in contexts if c.get('similarity', 0) >= similarity_threshold]

    # Fallback: If no contexts pass threshold, use top contexts anyway with warning
    if len(filtered_contexts) == 0 and len(contexts) > 0:
        print(f"\n⚠️  Warning: No contexts meet similarity threshold {similarity_threshold}")
        print(f"   Using top {min(3, len(contexts))} contexts anyway (best available)")
        contexts = contexts[:min(3, len(contexts))]
    else:
        contexts = filtered_contexts

    print(f"\n✅ Using {len(contexts)} contexts (similarity >= {similarity_threshold})\n")

    # 4. Generate answer
    print("🤖 Generating answer with LLM...\n")
    result = citation_gen.generate_answer(query, contexts)

    # 5. Display results
    print(f"ANSWER:\n{'-'*80}")
    print(result['answer'])
    print(f"\n{'-'*80}")

    if 'confidence' in result:
        confidence_emoji = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.4 else "🔴"
        print(f"\nConfidence: {confidence_emoji} {result['confidence']:.1%}")

    print(f"\nCONTEXTS USED ({len(contexts)}):")
    for i, ctx in enumerate(contexts, 1):
        metadata = ctx.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', '?')
        similarity = ctx.get('similarity', 0)
        rerank_score = ctx.get('rerank_score', None)

        if rerank_score is not None:
            print(f"  [{i}] {source}, Page {page} (Sim: {similarity:.2f}, Rerank: {rerank_score:.2f})\n {ctx['text']}\n")
        else:
            print(f"  [{i}] {source}, Page {page} (Similarity: {similarity:.2f})\n {ctx['text']}\n")

    print(f"\n{'='*80}\n")

    return result


def interactive_mode():
    """Run in interactive mode to ask multiple questions"""
    print("\n" + "="*80)
    print("Medical RAG System - Interactive Mode")
    print("Type 'exit' or 'quit' to stop")
    print("="*80 + "\n")
    
    while True:
        query = input("Enter your medical question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid question.\n")
            continue
        
        try:
            query_rag_system(query)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def test_sample_queries():
    """Test with predefined sample queries"""
    sample_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes pneumonia?",
        "What are the risk factors for heart disease?",
        "How to diagnose asthma?"
    ]
    
    print("\n" + "="*80)
    print("Testing RAG System with Sample Queries")
    print("="*80 + "\n")
    
    for query in sample_queries:
        try:
            query_rag_system(query, top_k=3)
            input("\nPress Enter to continue to next query...")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    import sys
    
    # Check database exists
    vector_store_check = VectorStore(persist_dir="./data/vectordb")
    doc_count = vector_store_check.get_count()
    
    if doc_count == 0:
        print("\n❌ Error: Vector database is empty!")
        print("Please run 'python setup_database.py' first to populate the database.\n")
        sys.exit(1)
    
    print(f"\n✓ Vector database loaded: {doc_count} documents found")
    
    # Choose mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_sample_queries()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            # Single query from command line
            query = " ".join(sys.argv[1:])
            query_rag_system(query)
    else:
        # Default: interactive mode
        interactive_mode()
