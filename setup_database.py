from src.ingestion import PDFProcessor
from src.embedding import LocalEmbedder
from src.vectordb import VectorStore
from tqdm import tqdm

def setup_rag_database():
    """Complete pipeline to setup RAG database"""
    
    print("Step 1: Processing PDFs...")
    processor = PDFProcessor("./data/raw_pdfs")
    chunks = processor.process_all_pdfs()
    print(f"Total chunks: {len(chunks)}")
    
    print("\nStep 2: Loading embedding model...")
    embedder = LocalEmbedder(model_name="BAAI/bge-base-en-v1.5")
    
    print("\nStep 3: Generating embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.embed_documents(texts, batch_size=32)
    
    print("\nStep 4: Storing in vector database...")
    vector_store = VectorStore(persist_dir="./data/vectordb")
    vector_store.add_documents(chunks, embeddings)
    
    print("\n✓ Database setup complete!")

if __name__ == "__main__":
    setup_rag_database()
