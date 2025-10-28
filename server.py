"""
Frontend Server for Medical RAG System
Serves the web interface and integrates with the RAG pipeline
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import LocalEmbedder
from src.vectordb import VectorStore
from src.query_processor import QueryProcessor
from src.generation import CitationGenerator
from src.config import RETRIEVAL_CONFIG, DEBUG_CONFIG, LLM_CONFIG, EMBEDDING_CONFIG

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical RAG Frontend")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize RAG components
print("Initializing Medical RAG components...")
try:
    embedder = LocalEmbedder()
    vector_store = VectorStore()
    query_processor = QueryProcessor()
    citation_gen = CitationGenerator()
    print("✓ All components loaded successfully!")
except Exception as e:
    print(f"❌ Error loading components: {e}")
    sys.exit(1)

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=RETRIEVAL_CONFIG['top_k'], ge=1, le=20)

class ContextItem(BaseModel):
    text: str
    similarity: float
    source: str = ""

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main frontend page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the RAG pipeline
    Returns the generated answer and retrieved context
    """
    try:
        if DEBUG_CONFIG['verbose']:
            print(f"\n{'='*80}")
            print(f"Query: {request.query}")
            print(f"{'='*80}")
        
        # 1. Process and correct query
        if DEBUG_CONFIG['log_retrievals']:
            print("Processing query...")
        
        corrected_query = query_processor.correct_query(request.query)
        expanded_query = query_processor.expand_query(corrected_query)
        
        if DEBUG_CONFIG['verbose'] and corrected_query != request.query:
            print(f"Corrected query: {corrected_query}")
        
        # 2. Embed query
        if DEBUG_CONFIG['log_embeddings']:
            print("Generating query embedding...")
        
        query_embedding = embedder.embed_query(expanded_query)
        
        # 3. Retrieve relevant contexts
        if DEBUG_CONFIG['log_retrievals']:
            print(f"Searching for top {request.top_k} relevant documents...")
        
        search_results = vector_store.search(query_embedding, request.top_k)
        
        # 4. Process search results
        contexts = []
        context_for_display = []
        
        if search_results['documents'] and len(search_results['documents']) > 0:
            for i, (doc, meta, distance) in enumerate(zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0] if 'distances' in search_results else [0] * len(search_results['documents'][0])
            )):
                # Convert distance to similarity (cosine distance)
                similarity = 1 - distance
                
                # Filter by similarity threshold
                if similarity >= RETRIEVAL_CONFIG['similarity_threshold']:
                    # For generation
                    contexts.append({
                        'text': doc,
                        'metadata': meta
                    })
                    
                    # For frontend display
                    context_for_display.append({
                        'text': doc[:500] + ('...' if len(doc) > 500 else ''),  # Truncate for display
                        'similarity': float(similarity),
                        'source': meta.get('source', f'Document {i+1}')
                    })
                elif DEBUG_CONFIG['log_retrievals']:
                    print(f"Filtered out result {i+1} (similarity: {similarity:.3f})")
        
        if DEBUG_CONFIG['log_retrievals']:
            print(f"Found {len(contexts)} relevant contexts")
        
        # 5. Generate answer
        if contexts:
            if DEBUG_CONFIG['log_llm_calls']:
                print("Generating answer...")
            
            result = citation_gen.generate_answer(request.query, contexts)
            answer = result['answer']
        else:
            answer = "I couldn't find relevant information in the database to answer your question. Please try rephrasing or ask a different question."
            if DEBUG_CONFIG['save_failed_queries']:
                with open('failed_queries.log', 'a') as f:
                    f.write(f"{request.query}\tNo relevant contexts found\n")
        
        if DEBUG_CONFIG['verbose']:
            print("\nAnswer generated successfully!")
            print(f"{'='*80}\n")
        
        return QueryResponse(
            answer=answer,
            context=context_for_display,
            query=request.query
        )
    
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        if DEBUG_CONFIG['save_failed_queries']:
            with open('failed_queries.log', 'a') as f:
                f.write(f"{request.query}\tError: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        doc_count = vector_store.get_count()
        return {
            "status": "healthy",
            "documents": doc_count,
            "config": {
                "provider": LLM_CONFIG['provider'],
                "model": LLM_CONFIG.get(f"{LLM_CONFIG['provider']}_model"),
                "embedding_model": EMBEDDING_CONFIG['model_name'],
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        doc_count = vector_store.get_count()
        return {
            "total_documents": doc_count,
            "retrieval_config": {
                "top_k": RETRIEVAL_CONFIG['top_k'],
                "similarity_threshold": RETRIEVAL_CONFIG['similarity_threshold'],
                "rerank": RETRIEVAL_CONFIG['rerank']
            },
            "llm_config": {
                "provider": LLM_CONFIG['provider'],
                "model": LLM_CONFIG.get(f"{LLM_CONFIG['provider']}_model"),
                "temperature": LLM_CONFIG['temperature']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Check if vector database has documents
    try:
        doc_count = vector_store.get_count()
        if doc_count == 0:
            print("\n⚠️  Warning: Vector database is empty!")
            print("Please run 'python setup_database.py' first to populate the database.\n")
    except Exception as e:
        print(f"\n❌ Error checking database: {e}\n")
    
    print("\n" + "="*80)
    print("🏥 Medical RAG Frontend Server")
    print("="*80)
    print("Frontend URL: http://localhost:8000")
    print("API Health:   http://localhost:8000/api/health")
    print("API Stats:    http://localhost:8000/api/stats")
    print("="*80)
    print(f"\nProvider: {LLM_CONFIG['provider']}")
    model_key = f"{LLM_CONFIG['provider']}_model"
    print(f"Model: {LLM_CONFIG.get(model_key)}")
    doc_count_display = doc_count if 'doc_count' in locals() else 'Unknown'
    print(f"Documents in database: {doc_count_display}")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
