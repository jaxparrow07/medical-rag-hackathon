from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Best models for medical domain:
        - BAAI/bge-base-en-v1.5 (335MB, fast)
        - sentence-transformers/all-MiniLM-L6-v2 (80MB, faster)
        - pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (medical-specific)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model on {device}...")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed documents in batches for GPU efficiency with normalization"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better similarity
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed single query with normalization and query prefix"""
        # Add query instruction for BGE models (improves retrieval)
        if "bge" in self.model._model_card_data.model_id.lower():
            query = f"Represent this query for searching medical information: {query}"
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
    
    def compute_similarity(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """Compute cosine similarity scores"""
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Cosine similarity (vectors already normalized)
        similarities = np.dot(doc_vecs, query_vec)
        
        return similarities.tolist()
