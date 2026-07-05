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

try:
    import fitz  # PyMuPDF — optional, only needed for PDF extraction
except ImportError:
    fitz = None
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
            if fitz is None:
                raise ImportError("PyMuPDF (fitz) is required for PDF extraction. Run: pip install PyMuPDF")
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


class SemanticChunker:
    """
    Semantic-aware dynamic chunking for legal documents.
    
    Inspired by ByteDance RAG Guideline §4.1.2:
    "Semantic completeness priority" — analyse paragraph structure,
    headings, and semantic pauses. Legal parameter tables get 128-token
    chunks; narrative text gets 384-token chunks.
    
    Key improvements over fixed-size chunking:
      1. Never splits across legal section boundaries (FACTS, ANALYSIS, etc.)
      2. Keeps numbered paragraphs ([1], [2], etc.) as atomic units
      3. Variable chunk sizes: shorter for statutes/lists, longer for reasoning
      4. Richer metadata: paragraph_range, legal_section, chunk_type
    """
    
    # Legal paragraph numbering patterns
    PARAGRAPH_PATTERN = re.compile(
        r'^\s*\[(\d+)\]',           # [1], [2], ...
        re.MULTILINE
    )
    
    # Section heading patterns
    SECTION_HEADING_PATTERN = re.compile(
        r'^(?:'
        r'(?:[IVXLCDM]+\.?\s+)'                 # Roman numerals: "IV. ANALYSIS"
        r'|(?:[A-Z]\.?\s+)'                      # Letter headings: "A. Background"
        r'|(?:\d+\.?\s+)'                         # Numbered: "3. Issues"
        r')?'
        r'(FACTS?|BACKGROUND|FACTUAL BACKGROUND'
        r'|LAW|LEGAL FRAMEWORK|APPLICABLE LAW|RELEVANT LAW'
        r'|ANALYSIS|DISCUSSION|REASONING'
        r'|CONCLUSION|DECISION|ORDER|DISPOSITION'
        r'|ISSUES?'
        r'|RELIEF|REMEDY|DAMAGES'
        r'|OVERVIEW|INTRODUCTION|SUMMARY'
        r'|BETWEEN|BEFORE'
        r')',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Content type detection patterns
    STATUTE_PATTERNS = [
        re.compile(r'(?:s\.|[Ss]ection)\s*\d+', re.IGNORECASE),
        re.compile(r'\(\d+\)\s*[A-Z]'),           # "(1) Every employer..."
        re.compile(r'\b(?:shall|must|may not|is prohibited)\b', re.IGNORECASE),
    ]
    
    LIST_PATTERN = re.compile(
        r'^\s*(?:[-•*]|\([a-z]\)|\([ivx]+\)|\d+\.)\s+',
        re.MULTILINE
    )
    
    def __init__(
        self,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        narrative_target: int = 384,
        structured_target: int = 128,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.narrative_target = narrative_target
        self.structured_target = structured_target
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def _detect_content_type(self, text: str) -> str:
        """Classify content as 'statute', 'list', or 'narrative'."""
        statute_hits = sum(1 for p in self.STATUTE_PATTERNS if p.search(text))
        list_hits = len(self.LIST_PATTERN.findall(text))
        
        if statute_hits >= 2:
            return "statute"
        if list_hits >= 3:
            return "list"
        return "narrative"
    
    def _target_size_for(self, content_type: str) -> int:
        """Get target chunk size based on content type (ByteDance §4.1.2)."""
        if content_type in ("statute", "list"):
            return self.structured_target
        return self.narrative_target
    
    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split document into major legal sections.
        
        Each section is a dict with:
          - 'name': section name (e.g., 'analysis', 'facts')
          - 'text': section content
          - 'start_pos': character position in original text
        """
        # Find all section headings
        headings = list(self.SECTION_HEADING_PATTERN.finditer(text))
        
        if not headings:
            return [{"name": "body", "text": text, "start_pos": 0}]
        
        sections = []
        
        # Text before first heading
        if headings[0].start() > 0:
            pre_text = text[:headings[0].start()].strip()
            if pre_text:
                sections.append({
                    "name": "preamble",
                    "text": pre_text,
                    "start_pos": 0,
                })
        
        # Each section
        for i, heading in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[heading.start():end].strip()
            section_name = heading.group(1).lower().strip()
            
            # Normalise section names
            section_map = {
                "facts": "facts", "background": "facts",
                "factual background": "facts",
                "law": "law", "legal framework": "law",
                "applicable law": "law", "relevant law": "law",
                "analysis": "analysis", "discussion": "analysis",
                "reasoning": "analysis",
                "conclusion": "conclusion", "decision": "conclusion",
                "order": "conclusion", "disposition": "conclusion",
                "issues": "issues", "issue": "issues",
                "relief": "relief", "remedy": "relief",
                "damages": "relief",
            }
            normalised = section_map.get(section_name, section_name)
            
            sections.append({
                "name": normalised,
                "text": section_text,
                "start_pos": heading.start(),
            })
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Split section text into legal paragraphs.
        
        Respects [N] numbering as atomic units.
        """
        # Try splitting by numbered paragraphs first
        parts = re.split(r'(?=\s*\[\d+\])', text)
        
        paragraphs = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Extract paragraph number if present
            num_match = re.match(r'\[(\d+)\]', part)
            para_num = int(num_match.group(1)) if num_match else None
            
            paragraphs.append({
                "text": part,
                "paragraph_number": para_num,
            })
        
        # If no numbered paragraphs found, fall back to double-newline split
        if len(paragraphs) <= 1 and '\n\n' in text:
            paragraphs = []
            for part in text.split('\n\n'):
                part = part.strip()
                if part:
                    paragraphs.append({"text": part, "paragraph_number": None})
        
        return paragraphs
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Semantically chunk a legal document.
        
        Pipeline:
          1. Split into sections (FACTS, ANALYSIS, CONCLUSION, ...)
          2. Within each section, split into paragraphs (respect [N] numbering)
          3. Group paragraphs into chunks respecting content-type target sizes
          4. Never split across section or paragraph boundaries
          5. Attach rich metadata
        """
        if not text.strip():
            return []
        
        base_metadata = base_metadata or {}
        
        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        
        sections = self._split_into_sections(text)
        
        for section in sections:
            section_name = section["name"]
            section_text = section["text"]
            
            paragraphs = self._split_into_paragraphs(section_text)
            
            # Determine content type and target chunk size
            content_type = self._detect_content_type(section_text)
            target_size = self._target_size_for(content_type)
            
            current_text = ""
            current_tokens = 0
            current_para_start = None
            current_para_end = None
            
            for para in paragraphs:
                para_text = para["text"]
                para_tokens = self._count_tokens(para_text)
                para_num = para["paragraph_number"]
                
                # If single paragraph exceeds max, it becomes its own chunk
                if para_tokens > self.max_chunk_tokens:
                    # Flush current buffer first
                    if current_text:
                        chunks.append(self._make_chunk(
                            content=current_text,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            base_metadata=base_metadata,
                            section=section_name,
                            content_type=content_type,
                            para_start=current_para_start,
                            para_end=current_para_end,
                        ))
                        chunk_index += 1
                        current_text = ""
                        current_tokens = 0
                    
                    # Store oversized paragraph as its own chunk
                    chunks.append(self._make_chunk(
                        content=para_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        section=section_name,
                        content_type=content_type,
                        para_start=para_num,
                        para_end=para_num,
                    ))
                    chunk_index += 1
                    current_para_start = None
                    current_para_end = None
                    continue
                
                # Would adding this paragraph exceed the target?
                if current_tokens + para_tokens > target_size and current_text:
                    chunks.append(self._make_chunk(
                        content=current_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        section=section_name,
                        content_type=content_type,
                        para_start=current_para_start,
                        para_end=current_para_end,
                    ))
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0
                    current_para_start = None
                    current_para_end = None
                
                # Accumulate
                separator = "\n\n" if current_text else ""
                current_text += separator + para_text
                current_tokens += para_tokens
                
                if para_num is not None:
                    if current_para_start is None:
                        current_para_start = para_num
                    current_para_end = para_num
            
            # Flush remaining text in this section
            if current_text and self._count_tokens(current_text) >= self.min_chunk_tokens:
                chunks.append(self._make_chunk(
                    content=current_text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    base_metadata=base_metadata,
                    section=section_name,
                    content_type=content_type,
                    para_start=current_para_start,
                    para_end=current_para_end,
                ))
                chunk_index += 1
                current_text = ""
                current_tokens = 0
            elif current_text and chunks:
                # Too small — merge with previous chunk if same section
                prev = chunks[-1]
                if prev.metadata.get("legal_section") == section_name:
                    prev.content += "\n\n" + current_text
                    prev.token_count = self._count_tokens(prev.content)
                    if current_para_end is not None:
                        prev.metadata["paragraph_range_end"] = current_para_end
                else:
                    # Different section — keep as small chunk
                    chunks.append(self._make_chunk(
                        content=current_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        section=section_name,
                        content_type=content_type,
                        para_start=current_para_start,
                        para_end=current_para_end,
                    ))
                    chunk_index += 1
        
        # Update total_chunks
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total
        
        return chunks
    
    def _make_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        base_metadata: Dict[str, Any],
        section: str,
        content_type: str,
        para_start: Optional[int],
        para_end: Optional[int],
    ) -> DocumentChunk:
        """Create a DocumentChunk with rich semantic metadata."""
        metadata = base_metadata.copy()
        metadata["legal_section"] = section
        metadata["chunk_type"] = content_type  # "narrative", "statute", "list"
        metadata["char_count"] = len(content)
        
        if para_start is not None:
            metadata["paragraph_range_start"] = para_start
        if para_end is not None:
            metadata["paragraph_range_end"] = para_end
        if para_start and para_end:
            metadata["paragraph_range"] = f"[{para_start}]-[{para_end}]"
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{document_id}_chunk_{chunk_index}_{content_hash}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            token_count=self._count_tokens(content),
            chunk_index=chunk_index,
            total_chunks=0,
            metadata=metadata,
        )


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
