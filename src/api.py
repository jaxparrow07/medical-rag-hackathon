from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

from embedding import LocalEmbedder
from vectordb import VectorStore
from query_processor import QueryProcessor
from generation import CitationGenerator

app = FastAPI(title="Medical RAG API")

# Initialize components
embedder = LocalEmbedder()
vector_store = VectorStore()
query_processor = QueryProcessor()
citation_gen = CitationGenerator(api_key=os.getenv("GEMINI_API_KEY"))

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_medical_rag(request: QueryRequest):
    """Main RAG endpoint"""
    try:
        # 1. Process and correct query
        corrected_query = query_processor.correct_query(request.query)
        expanded_query = query_processor.expand_query(corrected_query)
        
        # 2. Embed query
        query_embedding = embedder.embed_query(expanded_query)
        
        # 3. Retrieve relevant contexts
        search_results = vector_store.search(query_embedding, request.top_k)
        
        contexts = [
            {
                'text': doc,
                'metadata': meta
            }
            for doc, meta in zip(search_results['documents'], search_results['metadatas'])
        ]
        
        # 4. Generate answer with citations
        result = citation_gen.generate_answer(request.query, contexts)
        
        return QueryResponse(
            answer=result['answer'],
            contexts=result['contexts']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
