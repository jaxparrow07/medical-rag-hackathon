from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np

try:
    from .config import EMBEDDING_CONFIG, HARDWARE_CONFIG, DEBUG_CONFIG
except ImportError:
    # Fallback for direct imports
    from src.config import EMBEDDING_CONFIG, HARDWARE_CONFIG, DEBUG_CONFIG

class LocalEmbedder:
    def __init__(self, model_name: str = None, article_encoder: str = None):
        """
        Initialize embedder with config-driven settings

        Supports both single-encoder and dual-encoder architectures:

        Single-encoder (symmetric):
        - BAAI/bge-base-en-v1.5 (335MB, fast)
        - sentence-transformers/all-MiniLM-L6-v2 (80MB, faster)
        - pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (medical-specific)
        - FremyCompany/BioLORD-2023-M (medical, handles synonyms)

        Dual-encoder (asymmetric - better for retrieval):
        - ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder (BEST for medical)

        Args:
            model_name: Query encoder model name (or single model for symmetric)
            article_encoder: Article encoder model name (None for symmetric encoding)
        """
        # Use config if model_name not provided
        if model_name is None:
            model_name = EMBEDDING_CONFIG['model_name']

        # Check if using dual encoders (MedCPT style)
        if article_encoder is None:
            article_encoder = EMBEDDING_CONFIG.get('article_encoder', None)

        device = HARDWARE_CONFIG['device']

        self.use_dual_encoders = article_encoder is not None
        self.model_name = model_name
        self.article_encoder_name = article_encoder if self.use_dual_encoders else None

        if DEBUG_CONFIG['verbose']:
            if self.use_dual_encoders:
                print(f"🔧 Loading DUAL-ENCODER model on {device}...")
                print(f"   Query Encoder: {model_name}")
                print(f"   Article Encoder: {article_encoder}")
            else:
                print(f"🔧 Loading SINGLE-ENCODER model on {device}...")
                print(f"   Model: {model_name}")
            print(f"   Precision: {EMBEDDING_CONFIG['dtype']}")
            print(f"   Batch size: {EMBEDDING_CONFIG['batch_size']}")

        # Load query encoder (or single model)
        self.query_encoder = SentenceTransformer(model_name, device=device)
        self.query_encoder.max_seq_length = EMBEDDING_CONFIG['max_seq_length']
        self.embedding_dim = self.query_encoder.get_sentence_embedding_dimension()

        # Load article encoder if dual-encoder
        if self.use_dual_encoders:
            self.article_encoder = SentenceTransformer(article_encoder, device=device)
            self.article_encoder.max_seq_length = EMBEDDING_CONFIG['max_seq_length']
        else:
            self.article_encoder = self.query_encoder  # Use same model

        # Legacy: keep self.model for backward compatibility
        self.model = self.query_encoder

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
            print("GPU configured:")
            print(f"  - cuDNN benchmark: {EMBEDDING_CONFIG['enable_cudnn_benchmark']}")
            print(f"  - Deterministic: {EMBEDDING_CONFIG['deterministic']}")
            print(f"  - TF32: {EMBEDDING_CONFIG['allow_tf32']}")
            print(f"  - Matmul precision: {EMBEDDING_CONFIG['matmul_precision']}")
    
    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Embed documents using article encoder (for dual-encoder) or main model

        Args:
            texts: List of document texts to embed
            batch_size: Batch size for encoding (uses config default if None)

        Returns:
            List of embeddings (one per document)
        """
        if batch_size is None:
            batch_size = EMBEDDING_CONFIG['batch_size']

        # Use mixed precision if enabled
        if EMBEDDING_CONFIG['use_mixed_precision'] and HARDWARE_CONFIG['device'] == 'cuda':
            with torch.cuda.amp.autocast():
                embeddings = self._encode_documents(texts, batch_size)
        else:
            embeddings = self._encode_documents(texts, batch_size)

        if DEBUG_CONFIG['log_embeddings']:
            encoder_type = "article encoder" if self.use_dual_encoders else "single encoder"
            print(f"📄 Embedded {len(texts)} documents using {encoder_type} (batch size: {batch_size})")

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed query using query encoder

        For dual-encoder models (MedCPT), uses specialized query encoder.
        For single-encoder models, uses main model with optional prefix.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        # Add query prefix for BGE models only (single-encoder optimization)
        # MedCPT and other dual-encoders don't need prefixes
        if EMBEDDING_CONFIG['add_query_prefix'] and "bge" in self.model_name.lower() and not self.use_dual_encoders:
            query = f"Represent this query for searching medical information: {query}"

        embedding = self.query_encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_CONFIG['normalize_embeddings']
        )

        if DEBUG_CONFIG['log_embeddings']:
            encoder_type = "query encoder" if self.use_dual_encoders else "single encoder"
            print(f"🔍 Embedded query using {encoder_type}: {query[:50]}...")

        return embedding.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Alias for embed_query (for consistency with query_reformulator)"""
        return self.embed_query(query)

    def get_embedding_signature(self) -> dict:
        """Return signature describing current embedding configuration."""
        return {
            'model_name': self.model_name,
            'article_encoder': self.article_encoder_name,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': EMBEDDING_CONFIG['normalize_embeddings'],
            'add_query_prefix': EMBEDDING_CONFIG['add_query_prefix'],
            'dtype': EMBEDDING_CONFIG['dtype'],
        }
    
    def _encode_documents(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Internal method for encoding documents using article encoder"""
        return self.article_encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=DEBUG_CONFIG['verbose'],
            convert_to_numpy=True,
            normalize_embeddings=EMBEDDING_CONFIG['normalize_embeddings']
        )

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Legacy internal encoding method (kept for backward compatibility)"""
        return self._encode_documents(texts, batch_size)
    
    def compute_similarity(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """Compute cosine similarity scores"""
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Cosine similarity (vectors already normalized if config enabled)
        similarities = np.dot(doc_vecs, query_vec)
        
        return similarities.tolist()
