"""
Main file to test the RAG system with queries
Run this after setting up the database with setup_database.py
"""

import os
from dotenv import load_dotenv
from src.embedding import LocalEmbedder
from src.vectordb import VectorStore
from src.query_processor import QueryProcessor
from src.generation import CitationGenerator

# Load environment variables
load_dotenv()

def query_rag_system(query: str, top_k: int = 5):
    """Query the RAG system and get an answer with citations"""
    
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    # 1. Initialize components
    print("Loading components...")
    embedder = LocalEmbedder(model_name="BAAI/bge-base-en-v1.5")
    vector_store = VectorStore(persist_dir="./data/vectordb")
    query_processor = QueryProcessor()
    
    # Check if API key exists
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables!")
    
    citation_gen = CitationGenerator(api_key=api_key)
    
    # 2. Process and correct query
    print("Processing query...")
    corrected_query = query_processor.correct_query(query)
    if corrected_query != query:
        print(f"Corrected query: {corrected_query}")
    
    expanded_query = query_processor.expand_query(corrected_query)
    
    # 3. Embed query
    print("Generating query embedding...")
    query_embedding = embedder.embed_query(expanded_query)
    
    # 4. Retrieve relevant contexts
    print(f"Searching for top {top_k} relevant documents...")
    search_results = vector_store.search(query_embedding, top_k)
    
    contexts = []
    if search_results['documents'] and len(search_results['documents']) > 0:
        for doc, meta in zip(search_results['documents'][0], search_results['metadatas'][0]):
            contexts.append({
                'text': doc,
                'metadata': meta
            })
    
    print(f"Found {len(contexts)} relevant contexts\n")
    
    # 5. Generate answer with citations
    print("Generating answer with Gemini 2.5 Flash...\n")
    result = citation_gen.generate_answer(query, contexts)
    
    # 6. Display results
    print(f"ANSWER:\n{'-'*80}")
    print(result['answer'])
    print(f"\n{'-'*80}")
    
    print(f"\nSOURCES USED:")
    for i, citation in enumerate(result['citations'], 1):
        print(f"  [{i}] {citation}")
    
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
