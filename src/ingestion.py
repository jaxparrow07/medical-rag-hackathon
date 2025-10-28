import pymupdf
import re
from pathlib import Path
from typing import List, Dict

class PDFProcessor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)
    
    def extract_text_with_metadata(self, pdf_path: Path) -> List[Dict]:
        """Extract text with page numbers and book metadata"""
        doc = pymupdf.open(pdf_path)
        chunks = []
        
        book_name = pdf_path.stem  # Filename as book name
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            
            # Clean text
            text = self.clean_text(text)
            
            if len(text.strip()) < 50:  # Skip empty pages
                continue
            
            # Create chunks with metadata
            page_chunks = self.chunk_text(text, max_length=500)
            
            for chunk_id, chunk in enumerate(page_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': book_name,
                        'page': page_num,
                        'chunk_id': chunk_id,
                        'citation': f"{book_name}, Page {page_num}"
                    }
                })
        
        doc.close()
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import ftfy
        # Fix encoding issues
        text = ftfy.fix_text(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove headers/footers (common patterns)
        text = re.sub(r'Page \d+|\d+ Chapter', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text with overlap for context preservation"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            if len(chunk.strip()) > 100:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in directory"""
        all_chunks = []
        
        pdf_files = list(self.pdf_dir.glob('*.pdf'))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            chunks = self.extract_text_with_metadata(pdf_file)
            all_chunks.extend(chunks)
            print(f"  → Extracted {len(chunks)} chunks")
        
        return all_chunks
