"""
Benchmark different configurations to find optimal settings
"""
import time
import os
from src.config import BENCHMARK_CONFIG, PRESETS, EMBEDDING_CONFIG, LLM_CONFIG
from src.embedding import LocalEmbedder
from src.generation import CitationGenerator
from src.vectordb import VectorStore

def run_benchmark():
    """Run benchmarks on different presets"""
    if not BENCHMARK_CONFIG['enabled']:
        print("Benchmarking is disabled. Set BENCHMARK_CONFIG['enabled'] = True in src/config.py")
        return
    
    print("\n" + "="*70)
    print("RAG SYSTEM BENCHMARK")
    print("="*70)
    
    results = []
    
    # Initialize vector store (shared across all tests)
    print("\nInitializing vector store...")
    vector_store = VectorStore()
    
    for test_config in BENCHMARK_CONFIG['test_configs']:
        preset_name = test_config['preset']
        print(f"\n{'='*70}")
        print(f"Testing preset: {test_config['name']} ({preset_name})")
        print(f"{'='*70}")
        
        # Apply preset
        preset = PRESETS[preset_name]
        EMBEDDING_CONFIG.update(preset['embedding'])
        LLM_CONFIG.update(preset['llm'])
        
        print(f"\nConfiguration:")
        print(f"  Embedding dtype: {EMBEDDING_CONFIG['dtype']}")
        print(f"  Batch size: {EMBEDDING_CONFIG['batch_size']}")
        print(f"  Temperature: {LLM_CONFIG['temperature']}")
        print(f"  Top-p: {LLM_CONFIG['top_p']}")
        
        # Initialize components
        embedder = LocalEmbedder()
        citation_gen = CitationGenerator()
        
        # Run test queries
        preset_results = {
            'name': test_config['name'],
            'queries': []
        }
        
        for query in BENCHMARK_CONFIG['test_queries']:
            print(f"\n  Query: {query}")
            
            # Measure total latency
            start_time = time.time()
            
            # Embed query
            embed_start = time.time()
            query_embedding = embedder.embed_query(query)
            embed_time = time.time() - embed_start
            
            # Search
            search_start = time.time()
            search_results = vector_store.search(query_embedding, top_k=5)
            search_time = time.time() - search_start
            
            # Format contexts
            contexts = []
            if search_results['documents'] and len(search_results['documents']) > 0:
                for doc, meta in zip(search_results['documents'][0], search_results['metadatas'][0]):
                    contexts.append({
                        'text': doc,
                        'metadata': meta
                    })
            
            # Generate answer
            gen_start = time.time()
            if contexts:
                result = citation_gen.generate_answer(query, contexts)
            else:
                result = {
                    'answer': 'No relevant documents found',
                    'confidence': 0.0
                }
            gen_time = time.time() - gen_start
            
            total_time = time.time() - start_time
            
            query_result = {
                'query': query,
                'total_latency': total_time,
                'embed_time': embed_time,
                'search_time': search_time,
                'gen_time': gen_time,
                'confidence': result.get('confidence') or 0.0,
                'answer_length': len(result['answer'])
            }
            
            preset_results['queries'].append(query_result)
            
            print(f"    Total: {total_time:.3f}s (embed: {embed_time:.3f}s, search: {search_time:.3f}s, gen: {gen_time:.3f}s)")
            print(f"    Confidence: {query_result['confidence']:.2f}")
            print(f"    Answer: {result['answer'][:100]}...")
        
        # Calculate averages
        avg_latency = sum(q['total_latency'] for q in preset_results['queries']) / len(preset_results['queries'])
        avg_embed = sum(q['embed_time'] for q in preset_results['queries']) / len(preset_results['queries'])
        avg_search = sum(q['search_time'] for q in preset_results['queries']) / len(preset_results['queries'])
        avg_gen = sum(q['gen_time'] for q in preset_results['queries']) / len(preset_results['queries'])
        avg_confidence = sum(q['confidence'] for q in preset_results['queries']) / len(preset_results['queries'])
        
        preset_results['avg_total_latency'] = avg_latency
        preset_results['avg_embed_time'] = avg_embed
        preset_results['avg_search_time'] = avg_search
        preset_results['avg_gen_time'] = avg_gen
        preset_results['avg_confidence'] = avg_confidence
        
        results.append(preset_results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Preset':<20} {'Total (s)':<12} {'Embed (s)':<12} {'Search (s)':<12} {'Gen (s)':<12} {'Confidence':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['avg_total_latency']:<12.3f} "
              f"{result['avg_embed_time']:<12.3f} "
              f"{result['avg_search_time']:<12.3f} "
              f"{result['avg_gen_time']:<12.3f} "
              f"{result['avg_confidence']:<12.2f}")
    
    print("\n" + "="*70)
    print("Recommendation:")
    
    # Find best balance
    best_accuracy = max(results, key=lambda x: x['avg_confidence'])
    fastest = min(results, key=lambda x: x['avg_total_latency'])
    
    print(f"  Most Accurate: {best_accuracy['name']} (confidence: {best_accuracy['avg_confidence']:.2f})")
    print(f"  Fastest: {fastest['name']} (latency: {fastest['avg_total_latency']:.3f}s)")
    
    # Find balanced option
    for result in results:
        if 'balanced' in result['name'].lower():
            print(f"  Balanced: {result['name']} (confidence: {result['avg_confidence']:.2f}, latency: {result['avg_total_latency']:.3f}s)")
            break
    
    print("="*70 + "\n")

if __name__ == "__main__":
    run_benchmark()
