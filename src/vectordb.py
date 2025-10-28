import chromadb
from typing import List, Dict
from pathlib import Path

class VectorStore:
    def __init__(self, persist_dir: str = "./data/vectordb"):
        """Initialize ChromaDB with persistence"""
        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Use PersistentClient for automatic persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add documents with embeddings to vector store"""
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> Dict:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Return the full results structure (not unpacked)
        # This ensures consistency with ChromaDB's return format
        return {
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'distances': results['distances']
        }
    
    def get_count(self) -> int:
        """Get total number of documents in the collection"""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name="medical_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
