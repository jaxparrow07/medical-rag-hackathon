"""
Semantic Chunking for Medical Text

This module implements intelligent chunking strategies that preserve medical concepts
and avoid splitting information mid-sentence or mid-concept.

Key improvements over word-based chunking:
- Respects sentence boundaries (never splits mid-sentence)
- Groups semantically related sentences together
- Preserves medical concepts (e.g., drug mechanisms, disease descriptions)
- Adaptive chunk sizes based on content structure
"""

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
from typing import List, Dict, Optional
import re

try:
    from .config import EMBEDDING_CONFIG, DEBUG_CONFIG
except ImportError:
    from src.config import EMBEDDING_CONFIG, DEBUG_CONFIG


class SemanticChunker:
    """
    Semantic boundary-aware chunking for medical text

    Uses sentence embeddings to group semantically related content together,
    ensuring that chunks preserve complete medical concepts.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        similarity_threshold: float = 0.75,
        min_chunk_tokens: int = 100
    ):
        """
        Initialize semantic chunker

        Args:
            max_tokens: Maximum tokens per chunk (should match embedding model limit)
            similarity_threshold: Minimum cosine similarity to group sentences (0.0-1.0)
            min_chunk_tokens: Minimum tokens per chunk (avoid very small chunks)
        """
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens

        # Load medical-specific spaCy model (or fallback to default)
        try:
            self.nlp = spacy.load("en_core_sci_md")  # Medical NLP model
            if DEBUG_CONFIG.get('verbose', True):
                print("✅ Loaded medical spaCy model: en_core_sci_md")
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")  # Default English model
                if DEBUG_CONFIG.get('verbose', True):
                    print("⚠️  Medical spaCy model not found, using default: en_core_web_sm")
                    print("   Install with: python -m spacy download en_core_sci_md")
            except:
                self.nlp = None
                if DEBUG_CONFIG.get('verbose', True):
                    print("⚠️  No spaCy model found. Install with: python -m spacy download en_core_web_sm")

        # Use article encoder from config for consistency
        embedder_model = EMBEDDING_CONFIG.get('article_encoder') or EMBEDDING_CONFIG['model_name']

        try:
            self.embedder = SentenceTransformer(embedder_model)
            if DEBUG_CONFIG.get('verbose', True):
                print(f"✅ Loaded sentence embedder for chunking: {embedder_model}")
        except Exception as e:
            if DEBUG_CONFIG.get('verbose', True):
                print(f"⚠️  Could not load embedder for semantic chunking: {e}")
                print("   Falling back to simple sentence-based chunking")
            self.embedder = None

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Chunk text based on semantic boundaries

        Process:
        1. Split text into sentences (using spaCy)
        2. Embed each sentence
        3. Group semantically similar sentences
        4. Respect max_tokens limit
        5. Return chunks with metadata

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with 'text', 'metadata', 'num_sentences', 'num_tokens'
        """
        if not text or len(text.strip()) < 50:
            return []

        # Fallback to simple chunking if models not available
        if self.nlp is None or self.embedder is None:
            return self._simple_sentence_chunk(text, metadata)

        # 1. Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # If only one sentence or very few, return as is
        if len(sentences) <= 2:
            return [{
                'text': text.strip(),
                'metadata': metadata or {},
                'num_sentences': len(sentences),
                'num_tokens': len(text.split())
            }]

        # 2. Embed all sentences
        try:
            embeddings = self.embedder.encode(sentences, show_progress_bar=False)
        except Exception as e:
            if DEBUG_CONFIG.get('verbose', True):
                print(f"⚠️  Embedding failed, using simple chunking: {e}")
            return self._simple_sentence_chunk(text, metadata)

        # 3. Group sentences by semantic similarity
        chunks = self._group_sentences(sentences, embeddings, metadata)

        return chunks

    def chunk_by_headings(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Alternative chunking strategy based on document structure (headings, sections)

        Better for structured medical textbooks with clear section divisions.

        Detects headings using heuristics:
        - ALL CAPS lines
        - Short lines (< 80 chars)
        - No period at end

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of chunks (one per section, or sub-chunked if section too large)
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_heading = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect heading (heuristic: ALL CAPS, < 80 chars, no period at end)
            is_heading = self._is_heading(line)

            if is_heading:
                # Save previous section
                if current_section:
                    sections.append({
                        'heading': current_heading,
                        'text': '\n'.join(current_section),
                        'metadata': metadata or {}
                    })

                # Start new section
                current_heading = line
                current_section = []
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            sections.append({
                'heading': current_heading,
                'text': '\n'.join(current_section),
                'metadata': metadata or {}
            })

        # Now chunk each section semantically if too large
        final_chunks = []
        for section in sections:
            section_tokens = len(section['text'].split())

            if section_tokens > self.max_tokens:
                # Section too large, chunk semantically
                sub_chunks = self.chunk_text(section['text'], metadata)
                for chunk in sub_chunks:
                    # Add section heading to metadata
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}
                    chunk['metadata']['section_heading'] = section['heading']
                final_chunks.extend(sub_chunks)
            else:
                # Section fits, keep as is
                final_chunks.append({
                    'text': section['text'],
                    'metadata': {**(metadata or {}), 'section_heading': section['heading']},
                    'num_sentences': len(section['text'].split('.')),
                    'num_tokens': section_tokens
                })

        return final_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        return sentences

    def _group_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        metadata: Optional[Dict]
    ) -> List[Dict]:
        """
        Group sentences by semantic similarity

        Algorithm:
        1. Start with first sentence
        2. For each subsequent sentence:
           - Check if similar to current chunk (cosine similarity)
           - Check if adding would exceed max_tokens
           - If both OK, add to current chunk
           - Otherwise, start new chunk
        """
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_tokens = len(sentences[0].split())
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            sent_tokens = len(sentences[i].split())

            # Compute similarity to current chunk
            similarity = sklearn_cosine_similarity(
                current_embedding.reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            # Check if can add to current chunk
            can_add = (
                similarity >= self.similarity_threshold and
                current_chunk_tokens + sent_tokens <= self.max_tokens
            )

            if can_add:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_tokens += sent_tokens
                # Update chunk embedding (running average)
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                # Save current chunk (if meets minimum size)
                if current_chunk_tokens >= self.min_chunk_tokens:
                    chunks.append({
                        'text': ' '.join(current_chunk_sentences),
                        'metadata': metadata or {},
                        'num_sentences': len(current_chunk_sentences),
                        'num_tokens': current_chunk_tokens
                    })

                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_tokens = sent_tokens
                current_embedding = embeddings[i]

        # Add final chunk
        if current_chunk_sentences:
            chunks.append({
                'text': ' '.join(current_chunk_sentences),
                'metadata': metadata or {},
                'num_sentences': len(current_chunk_sentences),
                'num_tokens': current_chunk_tokens
            })

        return chunks

    def _simple_sentence_chunk(self, text: str, metadata: Optional[Dict]) -> List[Dict]:
        """
        Simple sentence-based chunking (fallback when embeddings unavailable)

        Splits text into sentences and groups by token count only.
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = len(sentence.split())

            if current_tokens + sent_tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            else:
                # Save current chunk
                if current_chunk and current_tokens >= self.min_chunk_tokens:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': metadata or {},
                        'num_sentences': len(current_chunk),
                        'num_tokens': current_tokens
                    })

                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sent_tokens

        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': metadata or {},
                'num_sentences': len(current_chunk),
                'num_tokens': current_tokens
            })

        return chunks

    def _is_heading(self, line: str) -> bool:
        """
        Detect if a line is a heading using heuristics

        Heading characteristics:
        - ALL CAPS or Title Case
        - Short (< 80 characters)
        - No period at end
        - May contain numbers (e.g., "Chapter 5: Cardiology")
        """
        if len(line) > 80:
            return False

        if line.endswith('.'):
            return False

        # Check if mostly uppercase or title case
        uppercase_ratio = sum(1 for c in line if c.isupper()) / max(len([c for c in line if c.isalpha()]), 1)

        return uppercase_ratio > 0.6  # At least 60% uppercase


if __name__ == "__main__":
    # Test semantic chunker
    print("Testing Semantic Chunker\n")

    chunker = SemanticChunker(max_tokens=100, similarity_threshold=0.75)

    # Test medical text
    test_text = """
    Diabetes mellitus is a metabolic disorder characterized by elevated blood glucose levels.
    The condition results from defects in insulin secretion, insulin action, or both.
    Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells.
    Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency.
    Treatment options for diabetes include lifestyle modifications, oral medications, and insulin therapy.
    Metformin is the first-line medication for type 2 diabetes.
    It works by reducing hepatic glucose production and improving insulin sensitivity.
    Alternative medications include sulfonylureas, which stimulate insulin secretion from pancreatic beta cells.
    """

    chunks = chunker.chunk_text(test_text, metadata={'source': 'test', 'page': 1})

    print(f"Created {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Sentences: {chunk['num_sentences']}")
        print(f"  Tokens: {chunk['num_tokens']}")
        print(f"  Text: {chunk['text'][:100]}...")
        print()
