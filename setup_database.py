#!/usr/bin/env python3
"""
Database Setup Script
Uses all new features: table extraction, medical NER, metadata enrichment
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import (
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    ENHANCED_FEATURES_CONFIG,
    PDF_EXTRACTION_CONFIG,
    MEDICAL_NER_CONFIG
)
from src.ingestion import PDFProcessor
from src.retrieval import LocalEmbedder
from src.retrieval import VectorStore


def main():
    """
    Database setup with all new features
    """
    print("\n" + "="*80)
    print("RAG Database Setup")
    print("="*80 + "\n")

    # Configuration summary
    print("Configuration:")
    print(f"  Embedding Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"  Chunking Strategy: {CHUNKING_CONFIG['strategy']}")
    print(f"  Max Tokens: {CHUNKING_CONFIG['max_tokens']}")
    print(f"\Features:")
    print(f"  Table Extraction: {ENHANCED_FEATURES_CONFIG['extract_tables']}")
    print(f"  Medical NER: {ENHANCED_FEATURES_CONFIG['use_medical_ner']}")
    print(f"  Context Expansion: {ENHANCED_FEATURES_CONFIG['use_context_expansion']}")
    print()

    # Step 1: Initialize PDF Processor
    print("Step 1: Initializing PDF Processor")
    print("-" * 80)

    pdf_dir = './data/raw_pdfs'
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        print(f"Please create the directory and add PDF files")
        return

    processor = PDFProcessor(
        pdf_dir=pdf_dir,
        extract_tables=ENHANCED_FEATURES_CONFIG['extract_tables'],
        use_medical_ner=ENHANCED_FEATURES_CONFIG['use_medical_ner'],
        use_semantic_chunking=(CHUNKING_CONFIG['strategy'] == 'semantic'),
        chunk_config={
            'max_tokens': CHUNKING_CONFIG['max_tokens'],
            'similarity_threshold': CHUNKING_CONFIG.get('similarity_threshold', 0.75),
            'min_chunk_tokens': CHUNKING_CONFIG.get('min_chunk_tokens', 100)
        }
    )

    print("PDF processor initialized\n")

    # Step 2: Process PDFs
    print("Step 2: Processing PDFs")
    print("-" * 80)

    result = processor.process_all_pdfs()

    if not result['all_chunks']:
        print("❌ No chunks extracted. Please check your PDFs")
        return

    print(f"\nPDF processing complete")
    print(f"\nGlobal Statistics:")
    stats = result['global_statistics']
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Pages: {stats['total_pages']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Tables: {stats['total_tables']}")
    print(f"  Total Size: {stats['total_file_size_mb']:.2f} MB")

    if stats.get('documents_needing_ocr', 0) > 0:
        print(f"\n⚠️  Warning: {stats['documents_needing_ocr']} documents may need OCR")

    if 'medical_entities' in stats:
        print(f"\nMedical Entities:")
        print(f"  Diseases: {stats['medical_entities']['total_diseases']}")
        print(f"  Drugs: {stats['medical_entities']['total_drugs']}")
        print(f"  Procedures: {stats['medical_entities']['total_procedures']}")

    print()

    # Step 3: Generate Embeddings
    print("Step 3: Generating Embeddings")
    print("-" * 80)

    embedder = LocalEmbedder(
        model_name=EMBEDDING_CONFIG['model_name']
    )

    print(f"Embedding {len(result['all_chunks'])} chunks...")

    embeddings = embedder.embed_documents([chunk['text'] for chunk in result['all_chunks']])

    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"   Dimension: {len(embeddings[0]) if embeddings else 0}\n")

    # Step 4: Store in Vector Database
    print("Step 4: Storing in Vector Database")
    print("-" * 80)

    # Create embedding signature for validation
    embedding_signature = {
        'model_name': EMBEDDING_CONFIG['model_name'],
        'article_encoder': EMBEDDING_CONFIG.get('article_encoder'),
        'embedding_dim': EMBEDDING_CONFIG['embedding_dim'],
        'normalize_embeddings': EMBEDDING_CONFIG['normalize_embeddings']
    }

    # Initialize vector store
    vector_store = VectorStore(
        persist_dir='./data/vectordb',
        embedding_signature=embedding_signature
    )

    # Clear existing data
    print("Clearing existing database...")
    vector_store.clear()

    # Add documents with position tracking for context expansion
    if ENHANCED_FEATURES_CONFIG['use_context_expansion']:
        from src.retrieval.context_expander import PositionAwareVectorStore

        position_aware_store = PositionAwareVectorStore(vector_store)
        position_aware_store.add_documents_with_positions(result['all_chunks'], embeddings)
        print("✅ Documents added with position tracking")
    else:
        vector_store.add_documents(result['all_chunks'], embeddings)
        print("✅ Documents added to vector database")

    # Verify storage
    count = vector_store.get_count()
    print(f"\n✅ Vector database ready")
    print(f"   Total documents: {count}\n")

    # Step 5: Summary
    print("="*80)
    print("Setup Complete!")
    print("="*80)
    print()

    print("Database Statistics:")
    print(f"  Total Documents: {count}")
    print(f"  Embedding Dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"  Model: {EMBEDDING_CONFIG['model_name']}")

    print("\Features Active:")
    if ENHANCED_FEATURES_CONFIG['extract_tables']:
        print(f"  ✅ Table Extraction: {stats['total_tables']} tables extracted")
    if ENHANCED_FEATURES_CONFIG['use_medical_ner']:
        print(f"  ✅ Medical NER: Entities extracted and enriched")
    if ENHANCED_FEATURES_CONFIG['use_context_expansion']:
        print(f"  ✅ Context Expansion: Position tracking enabled")

    print("\nDocument Details:")
    for doc in result['documents']:
        print(f"\n  📄 {doc['file_name']}")
        print(f"     Chunks: {doc['statistics']['total_chunks']}")
        print(f"     Pages: {doc['metadata']['num_pages']}")
        print(f"     Tables: {doc['statistics']['total_tables']}")
        print(f"     Quality: {doc['quality']['quality']}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Test the pipeline: python test_enhancements.py")
    print("  2. Run queries: python main.py")
    print("  3. Start API server: python server.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
