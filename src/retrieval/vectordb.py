import chromadb
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

class VectorStore:
    def __init__(
        self,
        persist_dir: str = "./data/vectordb",
        embedding_signature: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ChromaDB with persistence"""
        # Ensure directory exists
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.persist_dir / "embedding_config.json"
        self.embedding_signature = embedding_signature
        
        # Use PersistentClient for automatic persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

        self.stored_signature = self._load_signature()

        if self.stored_signature and self.embedding_signature:
            if not self._signature_matches(self.stored_signature, self.embedding_signature):
                raise ValueError(
                    "Vector store embeddings were generated with a different model. "
                    "Please rebuild the database with setup_database.py before querying."
                )

        if not self.stored_signature and self.embedding_signature:
            self._save_signature(self.embedding_signature)
            self.stored_signature = self.embedding_signature
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add documents with embeddings to vector store"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        expected_dim = self._expected_embedding_dim()
        if expected_dim and embeddings and len(embeddings[0]) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_dim}, got {len(embeddings[0])}. "
                "Re-run ingestion with the correct embedding model."
            )

        if self.embedding_signature and not self.stored_signature:
            self._save_signature(self.embedding_signature)
            self.stored_signature = self.embedding_signature

        ids = self._generate_ids(chunks)
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
    
    def search(self, query_embedding: List[float], top_k: int = 5, include_distances: bool = True) -> Dict:
        """
        Search for similar documents using ChromaDB

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            include_distances: Whether to include distance scores

        Returns:
            Dict with 'documents', 'metadatas', and 'distances' keys

        Note:
            ChromaDB with cosine distance returns: distance = 1 - cosine_similarity
            So distance of 0 = perfect match, distance of 2 = opposite vectors
        """
        # Ensure we have documents in the collection
        count = self.collection.count()
        if count == 0:
            print("⚠️  Warning: Vector database is empty!")
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

        # Limit top_k to available documents
        actual_k = min(top_k, count)

        try:
            include_fields = ['documents', 'metadatas', 'distances']
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                include=include_fields
            )

            # Debug: Log search results if verbose
            try:
                from src.config import DEBUG_CONFIG
                if DEBUG_CONFIG.get('log_retrievals', False):
                    if results['distances'] and len(results['distances'][0]) > 0:
                        print(f"   Retrieved {len(results['distances'][0])} documents from vector DB")
                        print(f"   Distance range: {min(results['distances'][0]):.4f} - {max(results['distances'][0]):.4f}")
            except Exception:
                pass

            # Return the full results structure
            return {
                'documents': results['documents'],
                'metadatas': results['metadatas'],
                'distances': results['distances'],
                'ids': results.get('ids', [[]])
            }

        except Exception as e:
            print(f"❌ Error during vector search: {e}")
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }
    
    def distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB cosine distance to cosine similarity

        ChromaDB cosine distance formula: distance = 1 - cosine_similarity
        Therefore: cosine_similarity = 1 - distance

        Args:
            distance: Cosine distance from ChromaDB (0 = identical, 2 = opposite)

        Returns:
            Cosine similarity (-1 to 1, where 1 = identical)
        """
        return 1.0 - distance

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
        if self.config_path.exists():
            self.config_path.unlink()
        self.stored_signature = None
        if self.embedding_signature:
            self._save_signature(self.embedding_signature)
            self.stored_signature = self.embedding_signature

    def validate_embeddings(
        self,
        embedder,
        sample_size: int = 3,
        similarity_threshold: float = 0.90
    ) -> None:
        """Validate that stored embeddings match the active embedder."""
        if self.collection.count() == 0:
            return

        try:
            sample = self.collection.peek(limit=min(sample_size, self.collection.count()))
        except Exception:
            # Older Chroma versions might not support peek when empty metadata
            return

        documents = sample.get('documents', [])
        stored_embeddings = sample.get('embeddings', [])

        if len(documents) == 0 or len(stored_embeddings) == 0:
            return

        new_embeddings = embedder.embed_documents(list(documents))

        for idx, (stored_vec, new_vec) in enumerate(zip(stored_embeddings, new_embeddings)):
            stored_arr = np.array(stored_vec)
            new_arr = np.array(new_vec)

            if stored_arr.shape != new_arr.shape:
                raise ValueError(
                    "Stored embeddings dimension does not match the active embedder. "
                    "Please rebuild the vector database with setup_database.py."
                )

            similarity = float(np.dot(stored_arr, new_arr))
            if similarity < similarity_threshold:
                chunk_id = None
                if 'ids' in sample and len(sample['ids']) > idx:
                    chunk_id = sample['ids'][idx]
                raise ValueError(
                    "Detected embedding mismatch between stored vectors and current encoder. "
                    "Re-run setup_database.py to regenerate embeddings. "
                    + (f"Problematic chunk id: {chunk_id}" if chunk_id else "")
                )

    def _load_signature(self) -> Optional[Dict[str, Any]]:
        if self.config_path.exists():
            try:
                with self.config_path.open('r', encoding='utf-8') as handle:
                    return json.load(handle)
            except Exception:
                print("⚠️  Warning: Failed to read embedding signature file; ignoring.")
        return None

    def _save_signature(self, signature: Dict[str, Any]) -> None:
        try:
            with self.config_path.open('w', encoding='utf-8') as handle:
                json.dump(signature, handle, indent=2)
        except Exception as exc:
            print(f"⚠️  Could not persist embedding signature: {exc}")

    def _signature_matches(self, stored: Dict[str, Any], provided: Dict[str, Any]) -> bool:
        keys = ['model_name', 'article_encoder', 'embedding_dim', 'normalize_embeddings']
        for key in keys:
            stored_val = stored.get(key)
            provided_val = provided.get(key)
            if stored_val is None and provided_val is None:
                continue
            if stored_val is None or provided_val is None:
                return False
            if stored_val != provided_val:
                return False
        return True

    def _expected_embedding_dim(self) -> Optional[int]:
        if self.stored_signature and self.stored_signature.get('embedding_dim'):
            return int(self.stored_signature['embedding_dim'])
        if self.embedding_signature and self.embedding_signature.get('embedding_dim'):
            return int(self.embedding_signature['embedding_dim'])
        return None

    def _generate_ids(self, chunks: List[Dict]) -> List[str]:
        ids = []
        seen = set()

        for index, chunk in enumerate(chunks):
            candidate = self._candidate_id(chunk, index)

            if candidate in seen:
                suffix = 1
                base = candidate
                while f"{base}_{suffix}" in seen:
                    suffix += 1
                candidate = f"{base}_{suffix}"

            seen.add(candidate)
            ids.append(candidate)

        return ids

    def _candidate_id(self, chunk: Dict, index: int) -> str:
        if chunk.get('id'):
            return str(chunk['id'])

        metadata = chunk.get('metadata', {}) or {}
        for key in ('chunk_id', 'id', 'citation'):
            value = metadata.get(key)
            if value:
                return str(value)

        text = chunk.get('text') or ''
        if text:
            digest = hashlib.md5(text.encode('utf-8')).hexdigest()
            return f"hash_{digest}"

        return f"doc_{index}"
