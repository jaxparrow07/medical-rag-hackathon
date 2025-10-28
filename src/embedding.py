from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np

try:
    from .config import EMBEDDING_CONFIG, HARDWARE_CONFIG, DEBUG_CONFIG
except ImportError:
    # Fallback for direct imports
    from config import EMBEDDING_CONFIG, HARDWARE_CONFIG, DEBUG_CONFIG

class LocalEmbedder:
    def __init__(self, model_name: str = None):
        """
        Initialize embedder with config-driven settings
        
        Best models for medical domain:
        - BAAI/bge-base-en-v1.5 (335MB, fast)
        - sentence-transformers/all-MiniLM-L6-v2 (80MB, faster)
        - pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (medical-specific)
        """
        # Use config if model_name not provided
        if model_name is None:
            model_name = EMBEDDING_CONFIG['model_name']
        
        device = HARDWARE_CONFIG['device']
        
        if DEBUG_CONFIG['verbose']:
            print(f"Loading embedding model '{model_name}' on {device}...")
            print(f"Precision: {EMBEDDING_CONFIG['dtype']}")
            print(f"Batch size: {EMBEDDING_CONFIG['batch_size']}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = EMBEDDING_CONFIG['max_seq_length']
        self.model_name = model_name
        
        # Apply GPU configuration if available
        if device == 'cuda':
            self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings based on config"""
        # Set precision
        if EMBEDDING_CONFIG['dtype'] == 'float16':
            self.model.half()
        
        # cuDNN settings
        torch.backends.cudnn.benchmark = EMBEDDING_CONFIG['enable_cudnn_benchmark']
        torch.backends.cudnn.deterministic = EMBEDDING_CONFIG['deterministic']
        
        # TF32 settings (RTX 30/40 series)
        torch.backends.cuda.matmul.allow_tf32 = EMBEDDING_CONFIG['allow_tf32']
        torch.backends.cudnn.allow_tf32 = EMBEDDING_CONFIG['allow_tf32']
        
        # Matrix multiplication precision
        torch.set_float32_matmul_precision(EMBEDDING_CONFIG['matmul_precision'])
        
        # Clear cache
        torch.cuda.empty_cache()
        
        if DEBUG_CONFIG['verbose']:
            print(f"GPU configured:")
            print(f"  - cuDNN benchmark: {EMBEDDING_CONFIG['enable_cudnn_benchmark']}")
            print(f"  - Deterministic: {EMBEDDING_CONFIG['deterministic']}")
            print(f"  - TF32: {EMBEDDING_CONFIG['allow_tf32']}")
            print(f"  - Matmul precision: {EMBEDDING_CONFIG['matmul_precision']}")
    
    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Embed documents with config-driven settings"""
        if batch_size is None:
            batch_size = EMBEDDING_CONFIG['batch_size']
        
        # Use mixed precision if enabled
        if EMBEDDING_CONFIG['use_mixed_precision'] and HARDWARE_CONFIG['device'] == 'cuda':
            with torch.cuda.amp.autocast():
                embeddings = self._encode(texts, batch_size)
        else:
            embeddings = self._encode(texts, batch_size)
        
        if DEBUG_CONFIG['log_embeddings']:
            print(f"Embedded {len(texts)} documents in batches of {batch_size}")
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed single query with optional prefix"""
        # Add query prefix if enabled
        if EMBEDDING_CONFIG['add_query_prefix'] and "bge" in self.model_name.lower():
            query = f"Represent this query for searching medical information: {query}"
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_CONFIG['normalize_embeddings']
        )
        
        if DEBUG_CONFIG['log_embeddings']:
            print(f"Embedded query: {query[:50]}...")
        
        return embedding.tolist()
    
    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Internal encoding method"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=DEBUG_CONFIG['verbose'],
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_CONFIG['normalize_embeddings']
        )
    
    def compute_similarity(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """Compute cosine similarity scores"""
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Cosine similarity (vectors already normalized if config enabled)
        similarities = np.dot(doc_vecs, query_vec)
        
        return similarities.tolist()
