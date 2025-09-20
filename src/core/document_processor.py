"""
Enhanced Legal Document Demystifier - Core Document Processor
Production-ready modular implementation with advanced features
"""


import asyncio
import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles


import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image


from src.utils.security import sanitize_text, encrypt_file, redact_sensitive_info
from src.utils.validators import validate_file_size, validate_file_type
from src.utils.exceptions import DocumentProcessingError, SecurityError


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Advanced document processing with security, performance, and privacy features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_file_size = config.get('max_file_size', 25 * 1024 * 1024)  # 25MB
        self.allowed_types = config.get('allowed_types', ['.pdf'])
        self.temp_dir = Path(tempfile.gettempdir()) / "legal_demystifier"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Thread pools for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
    async def process_document(
        self, 
        file_path: Union[str, Path],
        original_filename: str,  # Add original filename for validation
        privacy_mode: str = "standard",
        secure_hash: bool = False
    ) -> Dict:
        """
        Main document processing pipeline with async support
        
        Args:
            file_path: Path to the document
            original_filename: The original name of the file
            privacy_mode: "standard" or "high" privacy mode
            secure_hash: Whether to generate secure hash for privacy
            
        Returns:
            Dict containing processed document data
        """
        try:
            file_path = Path(file_path)
            
            # Security validation
            await self._validate_file(file_path, original_filename)
            
            # Generate secure hash if needed
            file_hash = None
            if secure_hash:
                file_hash = await self._generate_secure_hash(file_path)
                logger.info(f"Document hash generated: {file_hash[:8]}...")
            
            # Extract text (async)
            pages_data, extraction_method = await self._extract_text_async(file_path)
            
            # Process and chunk clauses
            clauses = await self._chunk_into_clauses_async(pages_data)
            
            # Apply privacy redaction
            if privacy_mode in ["standard", "high"]:
                pages_data, clauses = await self._apply_privacy_redaction(
                    pages_data, clauses, privacy_mode
                )
            
            # Generate metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "extraction_method": extraction_method,
                "pages_count": len(pages_data),
                "clauses_count": len(clauses),
                "privacy_mode": privacy_mode,
                "file_hash": file_hash,
                "processing_timestamp": asyncio.get_event_loop().time()
            }
            
            return {
                "pages": pages_data,
                "clauses": clauses,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")
    
    async def _validate_file(self, file_path: Path, original_filename: str) -> None:
        """Validate file security and constraints"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file size - pass string path to validator
        if not validate_file_size(str(file_path), self.max_file_size):
            raise SecurityError(f"File too large. Max size: {self.max_file_size} bytes")
        
        # Validate file type using the original filename string, not the sanitized path
        if not validate_file_type(original_filename, self.allowed_types):
            raise SecurityError(f"File type not allowed: {Path(original_filename).suffix}. Allowed types: {self.allowed_types}")
        
        # Additional security checks
        await self._scan_for_malicious_content(file_path)
    
    async def _scan_for_malicious_content(self, file_path: Path) -> None:
        """Basic malicious content scanning"""
        # Check file headers for PDF
        async with aiofiles.open(file_path, 'rb') as f:
            header = await f.read(8)
            if not header.startswith(b'%PDF'):
                raise SecurityError("Invalid PDF file header")
    
    async def _generate_secure_hash(self, file_path: Path) -> str:
        """Generate secure hash for privacy mode"""
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            return hashlib.sha256(content).hexdigest()
    
    async def _extract_text_async(self, file_path: Path) -> Tuple[List[Tuple[int, str]], str]:
        """Async text extraction with fallback to OCR"""
        loop = asyncio.get_event_loop()
        
        try:
            # First try direct text extraction
            pages_text, method = await loop.run_in_executor(
                self.thread_executor, 
                self._extract_text_direct, 
                file_path
            )
            
            if any(text.strip() for _, text in pages_text):
                return pages_text, method
            
            # Fallback to OCR
            logger.info("Falling back to OCR extraction")
            pages_text, method = await loop.run_in_executor(
                self.process_executor,
                self._extract_text_ocr,
                file_path
            )
            
            return pages_text, method
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise DocumentProcessingError(f"Text extraction failed: {str(e)}")
    
    def _extract_text_direct(self, file_path: Path) -> Tuple[List[Tuple[int, str]], str]:
        """Direct text extraction from PDF"""
        try:
            reader = PdfReader(str(file_path))
            pages_text = []
            
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages_text.append((i, text))
            
            return pages_text, "direct_text"
        except Exception as e:
            logger.warning(f"Direct text extraction failed: {str(e)}")
            return [], "failed"
    
    def _extract_text_ocr(self, file_path: Path) -> Tuple[List[Tuple[int, str]], str]:
        """OCR-based text extraction"""
        try:
            # Convert PDF to images
            images = convert_from_path(str(file_path), dpi=300, thread_count=2)
            pages_text = []
            
            for i, img in enumerate(images, start=1):
                # Optimize image for OCR
                img = self._optimize_image_for_ocr(img)
                text = pytesseract.image_to_string(img, config='--psm 6')
                pages_text.append((i, text))
            
            return pages_text, "ocr"
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise DocumentProcessingError(f"OCR extraction failed: {str(e)}")
    
    def _optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimize image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast and sharpness
        from PIL import ImageEnhance, ImageFilter
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small
        if image.width < 1000:
            scale_factor = 1000 / image.width
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    async def _chunk_into_clauses_async(self, pages_data: List[Tuple[int, str]]) -> List[Dict]:
        """Async clause chunking with advanced pattern recognition"""
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.thread_executor,
            self._chunk_into_clauses,
            pages_data
        )
    
    def _chunk_into_clauses(self, pages_data: List[Tuple[int, str]]) -> List[Dict]:
        """Enhanced clause chunking with multiple strategies"""
        
        # Enhanced heading patterns
        heading_patterns = [
            r"^\s*((section|clause|article|paragraph)\s*\d+(\.\d+)*)[:.)-]?\s*",
            r"^\s*(\d+(\.\d+)*)[:.)-]\s*",
            r"^[A-Z][A-Z\s\-/]{4,}$",
            r"^\s*([A-Z]\.\s*|[IVX]+\.\s*)",  # A. or I. patterns
            r"^\s*\(\s*[a-z]\s*\)",  # (a) patterns
        ]
        
        combined_pattern = "|".join(f"({pattern})" for pattern in heading_patterns)
        heading_re = re.compile(combined_pattern, re.I | re.M)
        
        clauses = []
        clause_id = 0
        
        for page_num, text in pages_data:
            if not text.strip():
                continue
                
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            current_clause = {"id": None, "page": page_num, "heading": None, "text": []}
            
            for line in lines:
                heading_match = heading_re.match(line)
                
                if heading_match and current_clause["text"]:
                    # Save previous clause
                    clause_id += 1
                    current_clause["id"] = f"p{page_num}-c{clause_id}"
                    current_clause["text"] = "\n".join(current_clause["text"]).strip()
                    current_clause = self._enrich_clause_data(current_clause)
                    clauses.append(current_clause)
                    
                    # Start new clause
                    current_clause = {
                        "id": None, 
                        "page": page_num, 
                        "heading": line.strip(), 
                        "text": []
                    }
                elif heading_match and not current_clause["text"]:
                    current_clause["heading"] = line.strip()
                else:
                    current_clause["text"].append(line)
            
            # Add last clause if exists
            if current_clause["text"]:
                clause_id += 1
                current_clause["id"] = f"p{page_num}-c{clause_id}"
                current_clause["text"] = "\n".join(current_clause["text"]).strip()
                current_clause = self._enrich_clause_data(current_clause)
                clauses.append(current_clause)
        
        # Fallback: paragraph-based chunking if too few clauses
        if len(clauses) < 3:
            clauses = self._chunk_by_paragraphs(pages_data)
        
        return clauses
    
    def _enrich_clause_data(self, clause: Dict) -> Dict:
        """Add metadata to clause"""
        text = clause["text"]
        
        # Add text statistics
        clause["word_count"] = len(text.split())
        clause["char_count"] = len(text)
        clause["sentence_count"] = len(re.findall(r'[.!?]+', text))
        
        # Detect clause type
        clause["clause_type"] = self._detect_clause_type(text)
        
        # Extract key entities
        clause["entities"] = self._extract_entities(text)
        
        return clause
    
    def _detect_clause_type(self, text: str) -> str:
        """Detect the type of legal clause"""
        text_lower = text.lower()
        
        type_patterns = {
            "payment": r"(payment|pay|fee|cost|charge|amount|currency)",
            "termination": r"(terminat|end|expir|cancel)",
            "confidentiality": r"(confidential|secret|proprietary|non-disclosure)",
            "liability": r"(liable|liability|responsible|damage|loss)",
            "intellectual_property": r"(copyright|patent|trademark|ip|intellectual property)",
            "dispute": r"(dispute|arbitration|court|litigation|resolution)",
            "governance": r"(board|director|management|control|voting)",
            "definition": r"(means|defined|definition|refers to)",
            "force_majeure": r"(force majeure|act of god|unforeseeable)",
            "amendment": r"(amend|modify|change|alter)"
        }
        
        for clause_type, pattern in type_patterns.items():
            if re.search(pattern, text_lower):
                return clause_type
        
        return "general"
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract key entities from clause text"""
        entities = {
            "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', text),
            "amounts": re.findall(r'[₹$£€]\s*[\d,]+(?:\.\d{2})?|\b\d+\s*(?:million|billion|thousand)\b', text),
            "percentages": re.findall(r'\b\d+(?:\.\d+)?%\b', text),
            "time_periods": re.findall(r'\b\d+\s*(?:days?|months?|years?|weeks?)\b', text),
            "phone_numbers": re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
            "email_addresses": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        }
        
        # Remove empty lists
        return {k: v for k, v in entities.items() if v}
    
    def _chunk_by_paragraphs(self, pages_data: List[Tuple[int, str]]) -> List[Dict]:
        """Fallback paragraph-based chunking"""
        clauses = []
        clause_id = 0
        
        for page_num, text in pages_data:
            paragraphs = re.split(r'\n\s*\n', text or "")
            
            for para in paragraphs:
                if para.strip() and len(para.strip()) > 50:  # Minimum length
                    clause_id += 1
                    clause = {
                        "id": f"p{page_num}-c{clause_id}",
                        "page": page_num,
                        "heading": None,
                        "text": para.strip()
                    }
                    clause = self._enrich_clause_data(clause)
                    clauses.append(clause)
        
        return clauses
    
    async def _apply_privacy_redaction(
        self, 
        pages_data: List[Tuple[int, str]], 
        clauses: List[Dict], 
        privacy_mode: str
    ) -> Tuple[List[Tuple[int, str]], List[Dict]]:
        """Apply privacy redaction to text"""
        loop = asyncio.get_event_loop()
        
        # Redact pages
        redacted_pages = []
        for page_num, text in pages_data:
            redacted_text = await loop.run_in_executor(
                self.thread_executor,
                redact_sensitive_info,
                text,
                privacy_mode
            )
            redacted_pages.append((page_num, redacted_text))
        
        # Redact clauses
        redacted_clauses = []
        for clause in clauses:
            redacted_clause = clause.copy()
            redacted_clause["text"] = await loop.run_in_executor(
                self.thread_executor,
                redact_sensitive_info,
                clause["text"],
                privacy_mode
            )
            redacted_clauses.append(redacted_clause)
        
        return redacted_pages, redacted_clauses
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=False)
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=False)
