from sentence_transformers import SentenceTransformer
import torch
from typing import List

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
        """Embed documents in batches for GPU efficiency"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed single query"""
        return self.model.encode(query, convert_to_numpy=True).tolist()
