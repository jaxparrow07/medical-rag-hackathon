"""
RAG System - Modular Architecture

Structure:
- ingestion/: PDF processing, chunking, metadata extraction
- retrieval/: Embeddings, vector search, query processing
- generation/: LLM-based answer generation
- evaluation/: Metrics and monitoring
- api.py: FastAPI models
- config.py: Configuration management
"""

# Re-export commonly used modules for convenience
from .ingestion import PDFProcessor
from .retrieval import LocalEmbedder, VectorStore, QueryProcessor
from .generation import CitationGenerator
from .evaluation import RAGEvaluator

__version__ = '2.0.0'

__all__ = [
    'PDFProcessor',
    'LocalEmbedder',
    'VectorStore',
    'QueryProcessor',
    'CitationGenerator',
    'RAGEvaluator'
]
