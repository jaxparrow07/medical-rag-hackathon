"""
Ingestion Pipeline
Integrates all improvements: table extraction, medical NER, metadata enrichment
"""

from pathlib import Path
from typing import List, Dict
import logging
import time
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.medical_ner import MedicalNER
from src.ingestion.semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    PDF processing pipeline with:
    - Table extraction
    - Medical entity recognition
    - Metadata enrichment
    - Structure preservation
    """

    def __init__(
        self,
        pdf_dir: str,
        extract_tables: bool = True,
        use_medical_ner: bool = True,
        use_semantic_chunking: bool = True,
        chunk_config: Dict = None,
        num_workers: int = None
    ):
        """
        Initialize PDF processor

        Args:
            pdf_dir: Directory containing PDF files
            extract_tables: Whether to extract tables
            use_medical_ner: Whether to use medical NER for metadata enrichment
            use_semantic_chunking: Whether to use semantic chunking
            chunk_config: Configuration for chunking
            num_workers: Number of parallel workers (None = sequential, -1 = auto-detect CPUs)
        """
        self.pdf_dir = Path(pdf_dir)
        self.extract_tables = extract_tables
        self.use_medical_ner = use_medical_ner
        self.use_semantic_chunking = use_semantic_chunking

        # Set number of workers for parallel processing
        if num_workers == -1:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = num_workers

        # Initialize components
        self.pdf_extractor = PDFExtractor(
            extract_tables=extract_tables,
            extract_images=False  # OCR not implemented yet
        )

        # Initialize medical NER if enabled
        if use_medical_ner:
            try:
                self.medical_ner = MedicalNER()
                logger.info("✅ Medical NER initialized")
            except Exception as e:
                logger.warning(f"⚠️  Medical NER initialization failed: {e}")
                self.medical_ner = None
        else:
            self.medical_ner = None

        # Initialize semantic chunker if enabled
        if use_semantic_chunking:
            chunk_config = chunk_config or {
                'max_tokens': 512,
                'similarity_threshold': 0.75,
                'min_chunk_tokens': 100
            }
            self.semantic_chunker = SemanticChunker(**chunk_config)
            logger.info("✅ Semantic chunking enabled")
        else:
            self.semantic_chunker = None
            logger.info("⚠️  Using legacy word-based chunking")

        if self.num_workers:
            logger.info(f"🚀 Parallel processing enabled with {self.num_workers} workers")

    def process_pdf(self, pdf_path: Path, show_progress: bool = True) -> Dict:
        """
        Process a single PDF with all enhancements

        Args:
            pdf_path: Path to PDF file
            show_progress: Whether to show detailed progress bars

        Returns:
            {
                'document_metadata': {...},
                'quality': {...},
                'chunks': [...],
                'statistics': {...}
            }
        """
        start_time = time.time()

        print(f"\nProcessing: {pdf_path.name}")
        logger.info(f"Processing: {pdf_path.name}")

        # Extract full document with structure
        if show_progress:
            print("  [1/4] Extracting PDF content and structure...")
        extraction_start = time.time()
        doc_data = self.pdf_extractor.extract_full_document(pdf_path, show_progress=show_progress)
        extraction_time = time.time() - extraction_start
        if show_progress:
            print(f"  ✓ Extracted {doc_data['metadata']['num_pages']} pages in {extraction_time:.2f}s")

        # Process each page and create chunks
        all_chunks = []
        total_tables = 0

        if show_progress:
            print(f"  [2/4] Chunking text from {doc_data['metadata']['num_pages']} pages...")
        chunking_start = time.time()

        page_progress = tqdm(
            doc_data['pages'],
            desc="    Pages",
            unit="page",
            disable=not show_progress,
            leave=False
        )

        for page_data in page_progress:
            # Get cleaned text (with tables integrated)
            text = page_data['text_clean']
            page_num = page_data['page_num']

            # Count tables
            total_tables += page_data['num_tables']

            # Create chunks from this page
            if self.use_semantic_chunking and self.semantic_chunker:
                # Semantic chunking
                page_chunks = self.semantic_chunker.chunk_text(
                    text,
                    metadata={
                        'source': doc_data['file_name'],
                        'page': page_num,
                        'citation': f"{doc_data['file_name']}, Page {page_num}",
                        'has_tables': page_data['has_tables'],
                        'num_tables': page_data['num_tables']
                    }
                )
            else:
                # Legacy chunking
                page_chunks = self._legacy_chunk(
                    text,
                    metadata={
                        'source': doc_data['file_name'],
                        'page': page_num,
                        'citation': f"{doc_data['file_name']}, Page {page_num}",
                        'has_tables': page_data['has_tables'],
                        'num_tables': page_data['num_tables']
                    }
                )

            # Enrich chunks with medical NER
            if self.medical_ner:
                for chunk in page_chunks:
                    chunk['metadata'] = self.medical_ner.enrich_chunk_metadata(
                        chunk['text'],
                        chunk.get('metadata', {})
                    )

            # Add tables information to metadata
            # Flatten for ChromaDB compatibility (only str, int, float, bool, None allowed)
            if page_data['has_tables']:
                for chunk in page_chunks:
                    # Store as JSON string for full table info
                    chunk['metadata']['table_info_json'] = json.dumps([
                        {
                            'table_index': t['table_index'],
                            'rows': t['num_rows'],
                            'cols': t['num_cols']
                        }
                        for t in page_data['tables']
                    ])
                    # Also store simple counts for easier filtering
                    chunk['metadata']['table_count'] = len(page_data['tables'])

            all_chunks.extend(page_chunks)
            page_progress.set_postfix({'chunks': len(all_chunks)})

        chunking_time = time.time() - chunking_start
        if show_progress:
            print(f"  ✓ Created {len(all_chunks)} chunks in {chunking_time:.2f}s")

        # Add document-level metadata to all chunks
        if show_progress:
            print("  [3/4] Enriching chunks with document-level metadata...")
        metadata_start = time.time()

        for chunk in all_chunks:
            chunk['metadata'].update({
                'document_title': doc_data['metadata']['title'],
                'document_author': doc_data['metadata']['author'],
                'total_pages': doc_data['metadata']['num_pages'],
                'file_size_mb': doc_data['metadata']['file_size_mb']
            })

        metadata_time = time.time() - metadata_start
        if show_progress:
            print(f"  ✓ Enriched {len(all_chunks)} chunks in {metadata_time:.2f}s")

        # Compute statistics
        if show_progress:
            print("  [4/4] Computing statistics...")
        stats_start = time.time()
        statistics = self._compute_statistics(all_chunks, doc_data)
        stats_time = time.time() - stats_start

        total_time = time.time() - start_time
        if show_progress:
            print(f"  ✓ Statistics computed in {stats_time:.2f}s")
            print(f"\n  Total processing time: {total_time:.2f}s ({len(all_chunks)} chunks)")

        return {
            'document_metadata': doc_data['metadata'],
            'quality': doc_data['quality'],
            'chunks': all_chunks,
            'statistics': statistics,
            'acronyms': doc_data['acronyms'],
            'processing_time': total_time
        }

    def _legacy_chunk(self, text: str, metadata: Dict, max_tokens: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Legacy word-based chunking

        Args:
            text: Text to chunk
            metadata: Base metadata
            max_tokens: Max tokens per chunk
            overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_tokens - overlap):
            chunk_words = words[i:i + max_tokens]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text.strip()) > 100:  # Minimum chunk size
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': len(chunks)
                    }
                })

        return chunks

    def _compute_statistics(self, chunks: List[Dict], doc_data: Dict) -> Dict:
        """
        Compute statistics about the processed document

        Args:
            chunks: List of chunks
            doc_data: Document data

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_chunks': len(chunks),
            'total_pages': doc_data['metadata']['num_pages'],
            'total_tables': doc_data['total_tables'],
            'total_acronyms': len(doc_data['acronyms']),
            'avg_chunk_length': 0,
            'chunks_with_tables': 0,
            'chunks_with_medical_entities': 0
        }

        if chunks:
            total_length = sum(len(c['text']) for c in chunks)
            stats['avg_chunk_length'] = round(total_length / len(chunks))

            stats['chunks_with_tables'] = sum(
                1 for c in chunks
                if c.get('metadata', {}).get('has_tables', False)
            )

            if self.medical_ner:
                stats['chunks_with_medical_entities'] = sum(
                    1 for c in chunks
                    if c.get('metadata', {}).get('entity_count', 0) > 0
                )

        return stats

    def _process_single_pdf_wrapper(self, pdf_file: Path) -> Dict:
        """
        Wrapper function for parallel processing of a single PDF

        Args:
            pdf_file: Path to PDF file

        Returns:
            Processed PDF result with metadata
        """
        try:
            result = self.process_pdf(pdf_file, show_progress=False)
            return {
                'success': True,
                'file_name': pdf_file.name,
                'result': result
            }
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            return {
                'success': False,
                'file_name': pdf_file.name,
                'error': str(e)
            }

    def process_all_pdfs(self) -> Dict:
        """
        Process all PDFs in directory

        Returns:
            {
                'all_chunks': [...],
                'documents': [...],
                'global_statistics': {...},
                'total_processing_time': float
            }
        """
        overall_start = time.time()

        pdf_files = list(self.pdf_dir.glob('*.pdf'))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return {
                'all_chunks': [],
                'documents': [],
                'global_statistics': {},
                'total_processing_time': 0
            }

        print(f"\n{'='*80}")
        print("PDF INGESTION PIPELINE")
        print('='*80)
        print(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
        if self.num_workers:
            print(f"Using {self.num_workers} parallel workers")
        print('='*80 + "\n")

        logger.info(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        documents = []
        processing_times = []

        # Choose parallel or sequential processing
        if self.num_workers and self.num_workers > 1:
            # PARALLEL PROCESSING
            print(f"🚀 Processing {len(pdf_files)} PDFs in parallel with {self.num_workers} workers...\n")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_pdf = {
                    executor.submit(self._process_single_pdf_wrapper, pdf_file): pdf_file
                    for pdf_file in pdf_files
                }

                # Progress bar
                file_progress = tqdm(
                    total=len(pdf_files),
                    desc="Processing PDFs",
                    unit="file",
                    colour='green'
                )

                completed = 0
                # Process completed tasks
                for future in as_completed(future_to_pdf):
                    completed += 1
                    pdf_file = future_to_pdf[future]

                    try:
                        result_wrapper = future.result()

                        if result_wrapper['success']:
                            result = result_wrapper['result']
                            all_chunks.extend(result['chunks'])
                            documents.append({
                                'file_name': result_wrapper['file_name'],
                                'metadata': result['document_metadata'],
                                'quality': result['quality'],
                                'statistics': result['statistics'],
                                'acronyms': result['acronyms'],
                                'processing_time': result.get('processing_time', 0)
                            })
                            processing_times.append(result.get('processing_time', 0))

                            file_progress.set_postfix({
                                'chunks': len(all_chunks),
                                'completed': f'{completed}/{len(pdf_files)}'
                            })
                        else:
                            print(f"\n  ❌ Error processing {result_wrapper['file_name']}: {result_wrapper['error']}")

                    except Exception as e:
                        logger.error(f"❌ Error processing {pdf_file.name}: {e}")
                        print(f"\n  ❌ Error processing {pdf_file.name}: {e}")

                    file_progress.update(1)

                file_progress.close()

        else:
            # SEQUENTIAL PROCESSING (original behavior)
            file_progress = tqdm(
                pdf_files,
                desc="Processing PDFs",
                unit="file",
                position=0,
                colour='green'
            )

            for idx, pdf_file in enumerate(file_progress, 1):
                try:
                    file_progress.set_description(f"[{idx}/{len(pdf_files)}] {pdf_file.name[:40]}")

                    result = self.process_pdf(pdf_file, show_progress=True)

                    all_chunks.extend(result['chunks'])
                    documents.append({
                        'file_name': pdf_file.name,
                        'metadata': result['document_metadata'],
                        'quality': result['quality'],
                        'statistics': result['statistics'],
                        'acronyms': result['acronyms'],
                        'processing_time': result.get('processing_time', 0)
                    })

                    processing_times.append(result.get('processing_time', 0))

                    # Calculate and display timing estimates
                    avg_time_per_file = sum(processing_times) / len(processing_times)
                    remaining_files = len(pdf_files) - idx
                    estimated_remaining = avg_time_per_file * remaining_files

                    file_progress.set_postfix({
                        'chunks': len(all_chunks),
                        'avg_time': f'{avg_time_per_file:.1f}s/file',
                        'eta': f'{estimated_remaining:.0f}s'
                    })

                    logger.info(f"  → Extracted {result['statistics']['total_chunks']} chunks")

                except Exception as e:
                    logger.error(f"❌ Error processing {pdf_file.name}: {e}")
                    print(f"\n  ❌ Error processing {pdf_file.name}: {e}")
                    continue

        # Compute global statistics
        print(f"\n{'='*80}")
        print("Computing global statistics...")
        global_stats = self._compute_global_statistics(documents, all_chunks)

        total_time = time.time() - overall_start
        global_stats['total_processing_time'] = total_time
        global_stats['avg_time_per_file'] = total_time / len(pdf_files) if pdf_files else 0

        # Display comprehensive summary
        print("✓ Complete!")
        print('='*80)
        print("\nPROCESSING SUMMARY")
        print('='*80)
        print(f"Total files processed: {len(documents)}/{len(pdf_files)}")
        print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average time per file: {global_stats['avg_time_per_file']:.2f}s")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Total pages processed: {global_stats.get('total_pages', 0)}")
        print(f"Total tables extracted: {global_stats.get('total_tables', 0)}")

        if processing_times:
            print("\nTiming breakdown:")
            print(f"  Fastest file: {min(processing_times):.2f}s")
            print(f"  Slowest file: {max(processing_times):.2f}s")
            print(f"  Median time: {sorted(processing_times)[len(processing_times)//2]:.2f}s")

            # Show speedup estimate for parallel processing
            if self.num_workers and self.num_workers > 1:
                sequential_estimate = sum(processing_times)
                speedup = sequential_estimate / total_time if total_time > 0 else 1
                print(f"\n  Parallel speedup: {speedup:.2f}x (estimated)")
                print(f"  Sequential time estimate: {sequential_estimate:.2f}s ({sequential_estimate/60:.2f} min)")

        print('='*80 + "\n")

        return {
            'all_chunks': all_chunks,
            'documents': documents,
            'global_statistics': global_stats,
            'total_processing_time': total_time
        }

    def _compute_global_statistics(self, documents: List[Dict], all_chunks: List[Dict]) -> Dict:
        """
        Compute global statistics across all documents

        Args:
            documents: List of document info
            all_chunks: All chunks from all documents

        Returns:
            Global statistics
        """
        stats = {
            'total_documents': len(documents),
            'total_chunks': len(all_chunks),
            'total_pages': sum(d['metadata']['num_pages'] for d in documents),
            'total_tables': sum(d['statistics']['total_tables'] for d in documents),
            'total_file_size_mb': sum(d['metadata']['file_size_mb'] for d in documents),
            'avg_chunks_per_document': 0,
            'documents_needing_ocr': 0
        }

        if documents:
            stats['avg_chunks_per_document'] = round(stats['total_chunks'] / stats['total_documents'])

            stats['documents_needing_ocr'] = sum(
                1 for d in documents
                if d['quality']['needs_ocr']
            )

        # Medical entity statistics (if NER enabled)
        if self.medical_ner:
            entity_summary = self.medical_ner.get_entity_summary(all_chunks)
            stats['medical_entities'] = {
                'total_diseases': entity_summary['total_unique_diseases'],
                'total_drugs': entity_summary['total_unique_drugs'],
                'total_procedures': entity_summary['total_unique_procedures'],
                'total_abbreviations': entity_summary['total_abbreviations']
            }

        return stats


def test_ingestion():
    """Test the ingestion pipeline"""
    print("\n" + "="*80)
    print("PDF Ingestion Pipeline Test")
    print("="*80 + "\n")

    # Check if PDFs exist
    pdf_dir = Path('./data/raw_pdfs')
    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("Please create ./data/raw_pdfs/ and add PDF files")
        return

    # Initialize processor with parallel processing
    # num_workers=-1 auto-detects CPU count
    # Set num_workers=4 for 4 parallel workers, or None for sequential
    processor = PDFProcessor(
        pdf_dir=str(pdf_dir),
        extract_tables=True,
        use_medical_ner=True,
        use_semantic_chunking=True,
        num_workers=-1  # Enable parallel processing with auto CPU detection
    )

    # Process all PDFs
    result = processor.process_all_pdfs()

    # Display detailed results
    print(f"\n{'='*80}")
    print("DETAILED PROCESSING RESULTS")
    print(f"{'='*80}\n")

    stats = result['global_statistics']

    print(f"Documents Processed: {stats['total_documents']}")
    print(f"Total Pages: {stats['total_pages']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Tables: {stats['total_tables']}")
    print(f"Total Size: {stats['total_file_size_mb']:.2f} MB")
    print(f"Avg Chunks/Doc: {stats['avg_chunks_per_document']}")

    if stats.get('total_processing_time'):
        print("\nPerformance Metrics:")
        print(f"  Total time: {stats['total_processing_time']:.2f}s ({stats['total_processing_time']/60:.2f} min)")
        print(f"  Avg time per file: {stats.get('avg_time_per_file', 0):.2f}s")
        if stats['total_pages'] > 0:
            time_per_page = stats['total_processing_time'] / stats['total_pages']
            print(f"  Avg time per page: {time_per_page:.2f}s")

    if stats.get('documents_needing_ocr', 0) > 0:
        print(f"\n⚠️  Documents needing OCR: {stats['documents_needing_ocr']}")

    if 'medical_entities' in stats:
        print("\nMedical Entities Extracted:")
        print(f"  Diseases: {stats['medical_entities']['total_diseases']}")
        print(f"  Drugs: {stats['medical_entities']['total_drugs']}")
        print(f"  Procedures: {stats['medical_entities']['total_procedures']}")
        print(f"  Abbreviations: {stats['medical_entities']['total_abbreviations']}")

    # Show sample chunk
    if result['all_chunks']:
        print(f"\n{'='*80}")
        print("Sample Chunk (with metadata)")
        print(f"{'='*80}\n")

        sample_chunk = result['all_chunks'][0]
        print(f"Text: {sample_chunk['text'][:200]}...")
        print(f"\nMetadata:")
        for key, value in sample_chunk['metadata'].items():
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    test_ingestion()
