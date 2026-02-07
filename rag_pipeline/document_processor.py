# RAG Pipeline - Document Processor
"""
Document processing for legal texts with structure-aware chunking.

Implements best practices for legal document processing:
- PDF text extraction with layout awareness
- Recursive text chunking (256-512 tokens with overlap)
- Legal hierarchy preservation (sections, paragraphs, clauses)
- Metadata extraction for enhanced retrieval
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

import fitz  # PyMuPDF
import tiktoken

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a legal document for embedding"""
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class ProcessedDocument:
    """Represents a fully processed legal document"""
    document_id: str
    source_path: str
    full_text: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_status: str
    error_message: Optional[str] = None


class LegalDocumentProcessor:
    """
    Processes legal documents (PDFs, text) for RAG pipeline ingestion.
    
    Features:
    - PDF text extraction with PyMuPDF
    - Structure-aware chunking for legal texts
    - Token counting with tiktoken
    - Metadata extraction (case info, citations, dates)
    - Legal section detection (Facts, Law, Analysis, Conclusion)
    """
    
    LEGAL_SECTION_PATTERNS = [
        (r'\b(FACTS?|BACKGROUND|FACTUAL BACKGROUND)\b', 'facts'),
        (r'\b(LAW|LEGAL FRAMEWORK|APPLICABLE LAW|RELEVANT LAW)\b', 'law'),
        (r'\b(ANALYSIS|DISCUSSION|REASONING)\b', 'analysis'),
        (r'\b(CONCLUSION|DECISION|ORDER|DISPOSITION)\b', 'conclusion'),
        (r'\b(ISSUES?)\b', 'issues'),
        (r'\b(RELIEF|REMEDY|DAMAGES)\b', 'relief'),
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int, content: str) -> str:
        """Generate unique ID for a chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_chunk_{chunk_index}_{content_hash}"
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF with layout awareness.
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        metadata = {
            "source_file": str(pdf_path),
            "file_name": pdf_path.name,
            "page_count": 0,
        }
        
        try:
            doc = fitz.open(str(pdf_path))
            metadata["page_count"] = len(doc)
            
            # Extract text from all pages
            full_text = []
            for page_num, page in enumerate(doc):
                # Get text with layout preservation
                text = page.get_text("text")
                if text.strip():
                    full_text.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            
            extracted_text = "\n\n".join(full_text)
            
            # Extract additional metadata from text
            metadata.update(self._extract_legal_metadata(extracted_text))
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def _extract_legal_metadata(self, text: str) -> Dict[str, Any]:
        """Extract legal metadata from document text"""
        metadata = {}
        
        # Extract citation patterns (e.g., "2020 ONSC 1234")
        citation_pattern = r'\d{4}\s+[A-Z]{2,6}\s+\d+'
        citations = re.findall(citation_pattern, text[:2000])  # Check first part
        if citations:
            metadata["primary_citation"] = citations[0]
            metadata["all_citations"] = list(set(citations))
        
        # Extract date patterns
        date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        dates = re.findall(date_pattern, text[:2000])
        if dates:
            metadata["decision_date"] = dates[0]
        
        # Detect court level
        court_indicators = {
            'Supreme Court': 'supreme',
            'Court of Appeal': 'appeal',
            'Superior Court': 'superior',
            'Divisional Court': 'divisional',
            'Small Claims': 'small_claims',
            'Tribunal': 'tribunal',
        }
        for indicator, level in court_indicators.items():
            if indicator.lower() in text[:3000].lower():
                metadata["court_level"] = level
                break
        
        # Detect legal sections present
        sections_found = []
        for pattern, section_name in self.LEGAL_SECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                sections_found.append(section_name)
        metadata["sections_present"] = sections_found
        
        return metadata
    
    def _detect_section(self, text: str) -> Optional[str]:
        """Detect which legal section a chunk belongs to"""
        for pattern, section_name in self.LEGAL_SECTION_PATTERNS:
            if re.search(pattern, text[:200], re.IGNORECASE):
                return section_name
        return None
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text using recursive character splitting with legal structure awareness.
        
        Uses a hierarchy of separators to maintain semantic coherence:
        1. Double newlines (paragraph breaks)
        2. Single newlines
        3. Sentences (periods followed by space)
        4. Spaces (last resort)
        """
        if not text.strip():
            return []
        
        base_metadata = base_metadata or {}
        chunks = []
        
        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize multiple newlines
        text = re.sub(r' {2,}', ' ', text)  # Normalize multiple spaces
        
        # Split into initial segments by paragraph
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self._count_tokens(para)
            
            # If paragraph alone exceeds chunk size, split it further
            if para_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, document_id, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_tokens = self._count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens <= self.chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                        current_tokens += sentence_tokens
                    else:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                current_chunk, document_id, chunk_index, base_metadata
                            ))
                            chunk_index += 1
                        
                        # Handle overlap
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = self._get_overlap_text(current_chunk)
                            current_chunk = overlap_text + " " + sentence
                            current_tokens = self._count_tokens(current_chunk)
                        else:
                            current_chunk = sentence
                            current_tokens = sentence_tokens
            
            # Normal paragraph - add to current chunk
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens
            
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, document_id, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + para
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = para
                    current_tokens = para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, document_id, chunk_index, base_metadata
            ))
        
        # Update total_chunks in all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.encoding.decode(overlap_tokens)
    
    def _create_chunk(
        self, 
        content: str, 
        document_id: str, 
        chunk_index: int,
        base_metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata"""
        chunk_metadata = base_metadata.copy()
        
        # Detect section
        section = self._detect_section(content)
        if section:
            chunk_metadata["legal_section"] = section
        
        # Add chunk-specific metadata
        chunk_metadata["char_count"] = len(content)
        
        return DocumentChunk(
            chunk_id=self._generate_chunk_id(document_id, chunk_index, content),
            document_id=document_id,
            content=content,
            token_count=self._count_tokens(content),
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            metadata=chunk_metadata
        )
    
    def process_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> ProcessedDocument:
        """
        Fully process a PDF document for RAG ingestion.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional custom document ID (uses filename if not provided)
            
        Returns:
            ProcessedDocument with chunks ready for embedding
        """
        pdf_path = Path(pdf_path)
        document_id = document_id or pdf_path.stem
        
        try:
            # Extract text
            full_text, metadata = self.extract_text_from_pdf(str(pdf_path))
            
            # Chunk the text
            chunks = self.chunk_text(full_text, document_id, metadata)
            
            return ProcessedDocument(
                document_id=document_id,
                source_path=str(pdf_path),
                full_text=full_text,
                chunks=chunks,
                metadata=metadata,
                processing_status="success"
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return ProcessedDocument(
                document_id=document_id,
                source_path=str(pdf_path),
                full_text="",
                chunks=[],
                metadata={},
                processing_status="failed",
                error_message=str(e)
            )
    
    def process_directory(self, directory: str) -> List[ProcessedDocument]:
        """Process all PDFs in a directory"""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        results = []
        for pdf_path in pdf_files:
            result = self.process_pdf(str(pdf_path))
            results.append(result)
            
            status_emoji = "✅" if result.processing_status == "success" else "❌"
            logger.info(f"{status_emoji} Processed {pdf_path.name}: {len(result.chunks)} chunks")
        
        return results


def main():
    """Test the document processor"""
    processor = LegalDocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Test with sample text
    sample_text = """
    SUPERIOR COURT OF JUSTICE – ONTARIO
    
    BEFORE: Smith J.
    
    BETWEEN:
    John Doe, Plaintiff
    - and -
    ABC Corporation, Defendant
    
    REASONS FOR DECISION
    
    I. BACKGROUND
    
    [1] This case involves a claim for wrongful dismissal. The plaintiff, John Doe, 
    worked for the defendant corporation for 15 years as a senior manager.
    
    [2] On January 15, 2024, the plaintiff was terminated without cause. The defendant 
    provided two weeks' notice, which the plaintiff claims is grossly inadequate given 
    his length of service and position.
    
    II. ISSUES
    
    [3] The following issues arise for determination:
    (a) Was the plaintiff an employee or independent contractor?
    (b) What is the appropriate notice period?
    (c) Is the plaintiff entitled to damages for bad faith?
    
    III. ANALYSIS
    
    [4] In determining whether an individual is an employee or independent contractor, 
    courts must consider the factors set out in 671122 Ontario Ltd. v. Sagaz Industries 
    Canada Inc., 2001 SCC 59:
    
    - Control over work
    - Ownership of tools
    - Chance of profit/risk of loss
    - Integration into the business
    
    [5] Having reviewed the evidence, I find that the control and integration factors 
    strongly support an employment relationship. The plaintiff worked exclusively for 
    the defendant, used company equipment, and was fully integrated into the management 
    structure.
    
    IV. CONCLUSION
    
    [6] For the foregoing reasons, I find that:
    (a) The plaintiff was an employee of the defendant;
    (b) The appropriate notice period is 18 months;
    (c) An additional 2 months is awarded for bad faith conduct.
    
    [7] The defendant shall pay the plaintiff damages in the amount of $180,000.
    """
    
    chunks = processor.chunk_text(sample_text, "test_case_001")
    
    print("\n" + "="*60)
    print("DOCUMENT CHUNKING RESULTS")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Section: {chunk.metadata.get('legal_section', 'N/A')}")
        print(f"Content preview: {chunk.content[:150]}...")


if __name__ == "__main__":
    main()
