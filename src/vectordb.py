import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_dir: str = "./data/vectordb"):
        """Initialize ChromaDB with persistence"""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
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
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
