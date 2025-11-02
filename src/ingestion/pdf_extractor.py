"""
PDF Extraction Module
Extracts text, tables, images, and metadata from medical PDFs
"""

import pymupdf
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Advanced PDF extraction with table detection, metadata enrichment,
    and structure preservation optimized for medical documents
    """

    def __init__(self, extract_tables: bool = True, extract_images: bool = False):
        """
        Initialize PDF extractor

        Args:
            extract_tables: Whether to extract tables from PDFs
            extract_images: Whether to extract images (OCR not implemented yet)
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def extract_pdf_metadata(self, pdf_path: Path) -> Dict:
        """
        Extract comprehensive PDF metadata

        Returns metadata like author, creation date, title, etc.
        """
        try:
            doc = pymupdf.open(pdf_path)
            metadata = doc.metadata or {}

            # Get file system metadata
            file_stats = pdf_path.stat()

            extracted_metadata = {
                'title': metadata.get('title', pdf_path.stem),
                'author': metadata.get('author', 'Unknown'),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'num_pages': len(doc),
                'file_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            }

            doc.close()
            return extracted_metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                'title': pdf_path.stem,
                'author': 'Unknown',
                'num_pages': 0
            }

    def detect_section_header(self, text: str, line: str, font_size: Optional[float] = None) -> bool:
        """
        Detect if a line is a section header

        Heuristics:
        - All caps
        - Short line (< 100 chars)
        - No period at end
        - Contains common header words
        - Larger font size (if available)
        """
        line = line.strip()

        if not line or len(line) > 100:
            return False

        # Check for all caps (allowing numbers and common punctuation)
        if line.isupper() and len(line) > 3:
            return True

        # Check for common medical section headers
        header_patterns = [
            r'^(INTRODUCTION|BACKGROUND|METHODS|RESULTS|DISCUSSION|CONCLUSION|ABSTRACT|SUMMARY)',
            r'^\d+\.\s+[A-Z]',  # Numbered sections like "1. INTRODUCTION"
            r'^Chapter\s+\d+',
            r'^SECTION\s+[IVX\d]+',
        ]

        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    def extract_tables_from_page(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """
        Extract tables from a specific page using pdfplumber

        Returns list of tables with their textual representation
        """
        if not self.extract_tables:
            return []

        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num > len(pdf.pages):
                    return []

                page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing

                # Extract all tables from page
                page_tables = page.extract_tables()

                for table_idx, table_data in enumerate(page_tables):
                    if not table_data:
                        continue

                    # Convert table to natural language text
                    table_text = self._format_table_as_text(table_data)

                    tables.append({
                        'page': page_num,
                        'table_index': table_idx,
                        'data': table_data,
                        'text': table_text,
                        'num_rows': len(table_data),
                        'num_cols': len(table_data[0]) if table_data else 0
                    })

        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num} of {pdf_path.name}: {e}")

        return tables

    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to natural language text

        Example:
        [['Drug', 'Dosage'], ['Aspirin', '100mg']]
        -> "Table: Drug | Dosage\nAspirin | 100mg"
        """
        if not table_data:
            return ""

        # Clean None values
        cleaned_table = []
        for row in table_data:
            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
            cleaned_table.append(cleaned_row)

        # Format as pipe-separated text
        table_lines = []
        for row in cleaned_table:
            row_text = ' | '.join(row)
            if row_text.strip():
                table_lines.append(row_text)

        if table_lines:
            return "TABLE:\n" + '\n'.join(table_lines)

        return ""

    def extract_text_with_structure(self, pdf_path: Path, page_num: int) -> Dict:
        """
        Extract text from a page with structural information

        Returns:
            {
                'text': str,
                'tables': List[Dict],
                'headers': List[str],
                'has_tables': bool,
                'text_with_tables': str  # Text with tables integrated
            }
        """
        # Extract text using PyMuPDF
        doc = pymupdf.open(pdf_path)
        page = doc[page_num - 1]

        text = page.get_text()

        # Extract tables using pdfplumber
        tables = self.extract_tables_from_page(pdf_path, page_num)

        # Detect section headers
        headers = []
        lines = text.split('\n')
        for line in lines:
            if self.detect_section_header(text, line):
                headers.append(line.strip())

        # Integrate tables into text
        text_with_tables = text
        if tables:
            # Append tables at the end with clear markers
            table_texts = [t['text'] for t in tables]
            text_with_tables = text + '\n\n' + '\n\n'.join(table_texts)

        doc.close()

        return {
            'text': text,
            'tables': tables,
            'headers': headers,
            'has_tables': len(tables) > 0,
            'text_with_tables': text_with_tables,
            'num_tables': len(tables)
        }

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving structure

        Improvements:
        - Fix encoding issues
        - Repair hyphenation
        - Remove headers/footers
        - Normalize whitespace
        """
        import ftfy

        # Fix encoding issues
        text = ftfy.fix_text(text)

        # Repair hyphenation (anti-\ninflammatory -> anti-inflammatory)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Remove common header/footer patterns
        # Page numbers
        text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines -> double newline

        return text.strip()

    def extract_acronyms(self, text: str) -> Dict[str, str]:
        """
        Extract acronym definitions from text

        Pattern: "Full Term (ACRONYM)" or "ACRONYM (Full Term)"

        Returns: {acronym: full_term}
        """
        acronyms = {}

        # Pattern 1: "Full Term (ACRONYM)"
        pattern1 = r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+\(([A-Z]{2,})\)'
        matches1 = re.findall(pattern1, text)
        for full_term, acronym in matches1:
            acronyms[acronym] = full_term.strip()

        # Pattern 2: "ACRONYM (Full Term)"
        pattern2 = r'\b([A-Z]{2,})\s+\(([A-Z][a-z]+(?:\s+[a-z]+)*)\)'
        matches2 = re.findall(pattern2, text)
        for acronym, full_term in matches2:
            if acronym not in acronyms:
                acronyms[acronym] = full_term.strip()

        return acronyms

    def validate_pdf_quality(self, pdf_path: Path) -> Dict[str, any]:
        """
        Validate PDF quality for text extraction

        Checks:
        - Is searchable (has text layer) vs scanned image
        - Text extraction quality
        - Average text per page
        """
        doc = pymupdf.open(pdf_path)

        total_text_length = 0
        pages_with_text = 0

        # Sample first 5 pages
        sample_size = min(5, len(doc))

        for i in range(sample_size):
            text = doc[i].get_text()
            text_length = len(text.strip())
            total_text_length += text_length

            if text_length > 50:
                pages_with_text += 1

        avg_chars_per_page = total_text_length / sample_size if sample_size > 0 else 0

        # Determine quality
        is_searchable = pages_with_text > 0
        quality = 'high' if avg_chars_per_page > 1000 else 'medium' if avg_chars_per_page > 300 else 'low'

        needs_ocr = not is_searchable or quality == 'low'

        doc.close()

        return {
            'is_searchable': is_searchable,
            'quality': quality,
            'avg_chars_per_page': round(avg_chars_per_page),
            'needs_ocr': needs_ocr,
            'pages_with_text': pages_with_text,
            'sample_size': sample_size
        }

    def extract_full_document(self, pdf_path: Path, show_progress: bool = False) -> Dict:
        """
        Extract complete document with all features

        Args:
            pdf_path: Path to PDF file
            show_progress: Whether to show page-by-page progress bar

        Returns comprehensive document structure with:
        - Metadata
        - Page-by-page text and tables
        - Document-level acronyms
        - Quality assessment
        """
        pdf_path = Path(pdf_path)

        # Validate quality first
        quality_info = self.validate_pdf_quality(pdf_path)

        if quality_info['needs_ocr']:
            logger.warning(f"PDF {pdf_path.name} may need OCR (quality: {quality_info['quality']})")

        # Extract metadata
        metadata = self.extract_pdf_metadata(pdf_path)

        # Extract content page by page
        doc = pymupdf.open(pdf_path)
        pages_data = []
        all_acronyms = {}
        total_tables_found = 0

        # Create progress bar for page extraction
        page_range = range(1, len(doc) + 1)
        if show_progress:
            page_iterator = tqdm(
                page_range,
                desc="      Extracting pages",
                unit="page",
                leave=False,
                position=1,
                colour='cyan'
            )
        else:
            page_iterator = page_range

        for page_num in page_iterator:
            # Extract text with structure
            page_data = self.extract_text_with_structure(pdf_path, page_num)

            # Clean text
            page_data['text_clean'] = self.clean_text(page_data['text_with_tables'])

            # Extract acronyms from this page
            page_acronyms = self.extract_acronyms(page_data['text_clean'])
            all_acronyms.update(page_acronyms)

            # Track tables
            if page_data['num_tables'] > 0:
                total_tables_found += page_data['num_tables']

            # Update progress bar with current stats
            if show_progress and isinstance(page_iterator, tqdm):
                page_iterator.set_postfix({
                    'tables': total_tables_found,
                    'acronyms': len(all_acronyms)
                })

            # Skip empty pages
            if len(page_data['text_clean'].strip()) < 50:
                continue

            page_data['page_num'] = page_num
            pages_data.append(page_data)

        doc.close()

        return {
            'file_path': str(pdf_path),
            'file_name': pdf_path.name,
            'metadata': metadata,
            'quality': quality_info,
            'pages': pages_data,
            'acronyms': all_acronyms,
            'total_pages': len(pages_data),
            'total_tables': sum(p['num_tables'] for p in pages_data)
        }


def test_extractor():
    """Test the PDF extractor"""
    extractor = PDFExtractor(extract_tables=True)

    # Test on a sample PDF
    pdf_dir = Path('./data/raw_pdfs')
    if not pdf_dir.exists():
        print("No PDFs found in ./data/raw_pdfs")
        return

    pdf_files = list(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        print("No PDF files found")
        return

    # Test first PDF
    test_pdf = pdf_files[0]
    print(f"\n{'='*60}")
    print(f"Testing PDF Extractor on: {test_pdf.name}")
    print(f"{'='*60}\n")

    # Extract full document
    result = extractor.extract_full_document(test_pdf)

    print(f"Metadata:")
    print(f"  Title: {result['metadata']['title']}")
    print(f"  Author: {result['metadata']['author']}")
    print(f"  Pages: {result['metadata']['num_pages']}")
    print(f"  File Size: {result['metadata']['file_size_mb']} MB")

    print(f"\nQuality:")
    print(f"  Searchable: {result['quality']['is_searchable']}")
    print(f"  Quality: {result['quality']['quality']}")
    print(f"  Avg Chars/Page: {result['quality']['avg_chars_per_page']}")

    print(f"\nContent:")
    print(f"  Total Pages: {result['total_pages']}")
    print(f"  Total Tables: {result['total_tables']}")
    print(f"  Acronyms Found: {len(result['acronyms'])}")

    if result['acronyms']:
        print(f"\n  Sample Acronyms:")
        for i, (acronym, full_term) in enumerate(list(result['acronyms'].items())[:5]):
            print(f"    {acronym}: {full_term}")

    if result['total_tables'] > 0:
        print(f"\n  Sample Table (first page with table):")
        for page in result['pages']:
            if page['has_tables']:
                table = page['tables'][0]
                print(f"    Page {page['page_num']}: {table['num_rows']}x{table['num_cols']} table")
                print(f"    {table['text'][:200]}...")
                break

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    test_extractor()
