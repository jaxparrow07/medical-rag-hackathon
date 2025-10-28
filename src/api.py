from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from embedding import LocalEmbedder
from vectordb import VectorStore
from query_processor import QueryProcessor
from generation import CitationGenerator
from config import RETRIEVAL_CONFIG, DEBUG_CONFIG

app = FastAPI(title="Medical RAG API")

# Initialize components with config-driven settings
if DEBUG_CONFIG['verbose']:
    print("Initializing Medical RAG API...")
    print("Loading components...")

embedder = LocalEmbedder()  # Uses EMBEDDING_CONFIG
vector_store = VectorStore()
query_processor = QueryProcessor()
citation_gen = CitationGenerator()  # Uses LLM_CONFIG for provider and model

if DEBUG_CONFIG['verbose']:
    print("API initialized successfully!")
    print(f"Default top_k: {RETRIEVAL_CONFIG['top_k']}")
    print(f"Similarity threshold: {RETRIEVAL_CONFIG['similarity_threshold']}")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=RETRIEVAL_CONFIG['top_k'], ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_medical_rag(request: QueryRequest):
    """Main RAG endpoint"""
    try:
        if DEBUG_CONFIG['log_retrievals']:
            print(f"\nReceived query: {request.query}")
        
        # 1. Process and correct query
        corrected_query = query_processor.correct_query(request.query)
        expanded_query = query_processor.expand_query(corrected_query)
        
        if DEBUG_CONFIG['verbose'] and corrected_query != request.query:
            print(f"Corrected query: {corrected_query}")
        
        # 2. Embed query
        query_embedding = embedder.embed_query(expanded_query)
        
        # 3. Retrieve relevant contexts with similarity threshold
        search_results = vector_store.search(query_embedding, request.top_k)
        
        # Filter by similarity threshold if configured
        contexts = []
        if search_results['documents'] and len(search_results['documents']) > 0:
            for i, (doc, meta, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0] if 'distances' in search_results else [0] * len(search_results['documents'][0])
            )):
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - distance
                
                if similarity >= RETRIEVAL_CONFIG['similarity_threshold']:
                    contexts.append({
                        'text': doc,
                        'metadata': meta
                    })
                elif DEBUG_CONFIG['log_retrievals']:
                    print(f"Filtered out result {i+1} (similarity: {similarity:.3f} < threshold: {RETRIEVAL_CONFIG['similarity_threshold']})")
        
        if DEBUG_CONFIG['log_retrievals']:
            print(f"Retrieved {len(contexts)} contexts above threshold")
        
        # 4. Generate answer with citations
        if contexts:
            result = citation_gen.generate_answer(request.query, contexts)
        else:
            result = {
                'answer': "No relevant information found in the database for your query.",
                'contexts': []
            }
        
        return QueryResponse(
            answer=result['answer'],
            contexts=result['contexts']
        )
    
    except Exception as e:
        if DEBUG_CONFIG['verbose']:
            print(f"Error processing query: {e}")
        if DEBUG_CONFIG['save_failed_queries']:
            # Log failed query for debugging
            with open('failed_queries.log', 'a') as f:
                f.write(f"{request.query}\t{str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with config info"""
    from config import ACTIVE_PRESET, LLM_CONFIG, EMBEDDING_CONFIG
    
    return {
        "status": "healthy",
        "config": {
            "preset": ACTIVE_PRESET,
            "provider": LLM_CONFIG['provider'],
            "model": LLM_CONFIG.get(f"{LLM_CONFIG['provider']}_model"),
            "embedding_model": EMBEDDING_CONFIG['model_name'],
            "precision": EMBEDDING_CONFIG['dtype']
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    from config import (
        ACTIVE_PRESET, 
        LLM_CONFIG, 
        EMBEDDING_CONFIG, 
        RETRIEVAL_CONFIG,
        SYSTEM_PROMPTS
    )
    
    return {
        "active_preset": ACTIVE_PRESET,
        "llm": {
            "provider": LLM_CONFIG['provider'],
            "model": LLM_CONFIG.get(f"{LLM_CONFIG['provider']}_model"),
            "temperature": LLM_CONFIG['temperature'],
            "top_p": LLM_CONFIG['top_p'],
            "top_k": LLM_CONFIG['top_k']
        },
        "embedding": {
            "model": EMBEDDING_CONFIG['model_name'],
            "dtype": EMBEDDING_CONFIG['dtype'],
            "batch_size": EMBEDDING_CONFIG['batch_size']
        },
        "retrieval": {
            "top_k": RETRIEVAL_CONFIG['top_k'],
            "similarity_threshold": RETRIEVAL_CONFIG['similarity_threshold']
        },
        "prompt": SYSTEM_PROMPTS['active']
    }

if __name__ == "__main__":
    import uvicorn
    
    if DEBUG_CONFIG['verbose']:
        print("\n" + "="*60)
        print("Starting Medical RAG API Server")
        print("="*60)
        print(f"Access API at: http://localhost:8000")
        print(f"API docs at: http://localhost:8000/docs")
        print(f"Health check: http://localhost:8000/health")
        print(f"Config info: http://localhost:8000/config")
        print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
