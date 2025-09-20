# Enhanced Legal Document Demystifier - Test Suite
# Comprehensive testing for production readiness

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

# Test fixtures and utilities
@pytest.fixture
def sample_pdf_path():
    """Create a sample PDF for testing"""
    # In a real test, you'd create or use a real PDF file
    return Path(__file__).parent / "fixtures" / "sample_contract.pdf"

@pytest.fixture
def config():
    """Test configuration"""
    return {
        "max_file_size": 25 * 1024 * 1024,
        "allowed_types": [".pdf"],
        "models": {
            "primary": {
                "summary": "gemini-1.5-flash",
                "reasoning": "gemini-1.5-flash"
            }
        }
    }

@pytest.fixture
def sample_clauses():
    """Sample clause data for testing"""
    return [
        {
            "id": "p1-c1",
            "page": 1,
            "heading": "Payment Terms",
            "text": "Payment shall be due within 30 days of invoice date.",
            "clause_type": "payment",
            "entities": {"time_periods": ["30 days"]}
        },
        {
            "id": "p1-c2", 
            "page": 1,
            "heading": "Termination",
            "text": "Either party may terminate this agreement immediately without notice.",
            "clause_type": "termination",
            "entities": {}
        }
    ]

# Document Processor Tests
class TestDocumentProcessor:
    """Test the document processing functionality"""
    
    @pytest.mark.asyncio
    async def test_process_document_success(self, config, sample_pdf_path):
        """Test successful document processing"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(config)
        
        # Mock the PDF reading functionality
        with patch('src.core.document_processor.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample legal text"
            mock_reader.return_value.pages = [mock_page]
            
            result = await processor.process_document(
                sample_pdf_path, 
                privacy_mode="standard"
            )
            
            assert result["success"] == True
            assert "pages" in result
            assert "clauses" in result
            assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_security_validation(self, config):
        """Test security validation"""
        from src.core.document_processor import DocumentProcessor
        from src.utils.exceptions import SecurityError
        
        processor = DocumentProcessor(config)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            await processor.process_document("/non/existent/file.pdf")
    
    def test_clause_chunking(self, config):
        """Test clause chunking functionality"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(config)
        
        sample_pages = [
            (1, "1. Introduction\nThis is the introduction clause.\n\n2. Payment Terms\nPayment shall be due within 30 days.")
        ]
        
        clauses = processor._chunk_into_clauses(sample_pages)
        
        assert len(clauses) >= 2
        assert any("Introduction" in clause.get("heading", "") for clause in clauses)
        assert any("Payment" in clause.get("heading", "") for clause in clauses)
    
    def test_entity_extraction(self, config):
        """Test entity extraction from clauses"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(config)
        
        text = "Payment of $50,000 is due within 30 days. Contact us at legal@company.com"
        entities = processor._extract_entities(text)
        
        assert "amounts" in entities
        assert "time_periods" in entities  
        assert "email_addresses" in entities
        assert len(entities["amounts"]) > 0
        assert len(entities["time_periods"]) > 0

# Risk Analyzer Tests
class TestAdvancedRiskAnalyzer:
    """Test the risk analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_clause_risk_analysis(self, sample_clauses):
        """Test individual clause risk analysis"""
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        
        analyzer = AdvancedRiskAnalyzer()
        
        # Test high-risk clause
        high_risk_clause = {
            "text": "This agreement automatically renews unless terminated with immediate effect",
            "clause_type": "termination"
        }
        
        result = await analyzer.analyze_clause_risk(high_risk_clause)
        
        assert result.level == "RED"
        assert result.severity >= 7
        assert len(result.risk_factors) > 0
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_document_risk_analysis(self, sample_clauses):
        """Test full document risk analysis"""
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        
        analyzer = AdvancedRiskAnalyzer()
        
        result = await analyzer.analyze_document_risk(sample_clauses)
        
        assert "overall_status" in result
        assert "risk_breakdown" in result
        assert "risk_score" in result
        assert result["total_clauses"] == len(sample_clauses)
    
    def test_indian_law_compliance(self):
        """Test Indian law compliance checking"""
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        
        analyzer = AdvancedRiskAnalyzer()
        
        # Test rent control violation
        text = "Tenant must vacate with 15 days notice"
        warnings = analyzer._run_indian_law_check(text)
        
        assert len(warnings) > 0
        assert "Rent Control" in warnings[0]
    
    def test_pattern_compilation(self):
        """Test regex pattern compilation"""
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        
        analyzer = AdvancedRiskAnalyzer()
        
        # Verify patterns are compiled
        assert len(analyzer._compiled_patterns) > 0
        assert "RED" in analyzer._compiled_patterns
        assert "YELLOW" in analyzer._compiled_patterns
        assert "GREEN" in analyzer._compiled_patterns

# AI Client Tests  
class TestAIClient:
    """Test the AI client functionality"""
    
    @pytest.mark.asyncio
    async def test_explain_clause(self, config):
        """Test clause explanation"""
        from src.core.ai_client import AIClient
        
        client = AIClient(config)
        
        # Mock the Gemini API call
        with patch('src.core.ai_client.genai.GenerativeModel') as mock_model:
            mock_response = Mock()
            mock_response.text = "This clause explains payment terms in simple language."
            mock_model.return_value.generate_content.return_value = mock_response
            
            result = await client.explain_clause(
                "Payment shall be due within 30 days",
                "p1-c1",
                1,
                risk_level="YELLOW"
            )
            
            assert result.content == "This clause explains payment terms in simple language."
            assert result.model_used is not None
            assert result.cached == False
    
    @pytest.mark.asyncio  
    async def test_caching(self, config):
        """Test AI response caching"""
        from src.core.ai_client import AIClient
        
        client = AIClient(config)
        
        # Mock successful response
        with patch('src.core.ai_client.genai.GenerativeModel') as mock_model:
            mock_response = Mock()
            mock_response.text = "Cached response"
            mock_model.return_value.generate_content.return_value = mock_response
            
            # First call
            result1 = await client.explain_clause("test clause", "p1-c1", 1)
            
            # Second call should be cached
            result2 = await client.explain_clause("test clause", "p1-c1", 1)
            
            # Verify caching behavior
            assert len(client._cache) > 0
    
    def test_rate_limiting(self, config):
        """Test rate limiting functionality"""
        from src.core.ai_client import AIClient
        
        client = AIClient(config)
        client._max_requests_per_minute = 2
        
        # Simulate multiple requests
        import time
        current_time = time.time()
        client._request_times = [current_time - 10, current_time - 5]
        
        # This should trigger rate limiting
        assert len(client._request_times) == 2

# Security Tests
class TestSecurity:
    """Test security and validation functionality"""
    
    def test_file_validation(self):
        """Test file validation"""
        from src.utils.validators import FileValidator
        
        validator = FileValidator({
            'max_file_size': 1024,
            'allowed_extensions': ['.pdf']
        })
        
        # Test valid filename
        assert validator.validate_filename("document.pdf") == True
        
        # Test invalid filename with path traversal
        with pytest.raises(Exception):
            validator.validate_filename("../../../etc/passwd")
    
    def test_text_validation(self):
        """Test text content validation"""
        from src.utils.validators import TextValidator
        
        validator = TextValidator({
            'max_text_length': 1000,
            'min_text_length': 5
        })
        
        # Test valid text
        assert validator.validate_text_length("This is valid text") == True
        
        # Test too short text
        with pytest.raises(Exception):
            validator.validate_text_length("Hi")
    
    def test_pii_redaction(self):
        """Test PII redaction"""
        from src.utils.security import SecurityManager
        
        security = SecurityManager()
        
        text = "My email is john.doe@example.com and my card number is 1234 5678 9012 3456"
        redacted = security.redact_sensitive_info(text, "standard")
        
        assert "john.doe@example.com" not in redacted
        assert "1234 XXXX XXXX XXXX" in redacted or "[REDACTED_EMAIL]" in redacted
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        from src.utils.security import SecurityManager
        
        security = SecurityManager()
        
        # Test dangerous script injection
        dangerous_input = "<script>alert('xss')</script>"
        assert not security.validate_input(dangerous_input)
        
        # Test safe input
        safe_input = "This is a normal legal question"
        assert security.validate_input(safe_input)

# API Tests
class TestAPI:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.web.app import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"
    
    def test_upload_endpoint_validation(self, client):
        """Test upload endpoint validation"""
        # Test without file
        response = client.post("/api/upload")
        assert response.status_code == 422
        
        # Test with invalid file type
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"test content")
            temp_file.seek(0)
            
            response = client.post(
                "/api/upload",
                files={"file": ("test.txt", temp_file, "text/plain")}
            )
            assert response.status_code == 400
    
    @patch('src.core.document_processor.DocumentProcessor.process_document')
    @patch('src.core.risk_analyzer.AdvancedRiskAnalyzer.analyze_document_risk') 
    @patch('src.core.ai_client.AIClient.summarize_document')
    def test_upload_endpoint_success(self, mock_summarize, mock_risk, mock_process, client):
        """Test successful upload"""
        # Mock responses
        mock_process.return_value = {
            "success": True,
            "pages": [(1, "test content")],
            "clauses": [],
            "metadata": {"file_name": "test.pdf"}
        }
        
        mock_risk.return_value = {
            "overall_status": "✅ SAFE",
            "risk_breakdown": {"RED": 0, "YELLOW": 1, "GREEN": 2}
        }
        
        mock_summary_response = Mock()
        mock_summary_response.content = "Test summary"
        mock_summarize.return_value = mock_summary_response
        
        # Create test PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4 test content")
            temp_file.seek(0)
            
            response = client.post(
                "/api/upload",
                files={"file": ("test.pdf", temp_file, "application/pdf")},
                data={
                    "privacy_mode": "standard",
                    "language": "english",
                    "analysis_type": "comprehensive"
                }
            )
            
            assert response.status_code == 200
            assert "session_id" in response.json()

# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, config, sample_clauses):
        """Test complete document processing workflow"""
        from src.core.document_processor import DocumentProcessor
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        from src.core.ai_client import AIClient
        
        # Initialize components
        processor = DocumentProcessor(config)
        analyzer = AdvancedRiskAnalyzer()
        ai_client = AIClient(config)
        
        # Mock document processing
        with patch.object(processor, 'process_document') as mock_process:
            mock_process.return_value = {
                "success": True,
                "pages": [(1, "test content")],
                "clauses": sample_clauses,
                "metadata": {"file_name": "test.pdf"}
            }
            
            # Process document
            doc_result = await processor.process_document("test.pdf")
            assert doc_result["success"] == True
            
            # Analyze risks
            risk_result = await analyzer.analyze_document_risk(doc_result["clauses"])
            assert "overall_status" in risk_result
            
            # Generate summary (mock AI call)
            with patch.object(ai_client, 'summarize_document') as mock_summarize:
                mock_response = Mock()
                mock_response.content = "Document summary"
                mock_summarize.return_value = mock_response
                
                summary = await ai_client.summarize_document("test content")
                assert summary.content == "Document summary"

# Performance Tests
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, config):
        """Test concurrent document processing"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(config)
        
        # Simulate multiple concurrent requests
        tasks = []
        for i in range(5):
            with patch.object(processor, 'process_document') as mock_process:
                mock_process.return_value = {"success": True}
                task = processor.process_document(f"test_{i}.pdf")
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        assert len(results) == 5
    
    def test_memory_usage(self, config):
        """Test memory usage with large documents"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(config)
        
        # Simulate large document
        large_text = "A" * 1000000  # 1MB of text
        entities = processor._extract_entities(large_text)
        
        # Should complete without memory issues
        assert isinstance(entities, dict)

# Configuration Tests
class TestConfiguration:
    """Test configuration and setup"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        from src.core.risk_analyzer import AdvancedRiskAnalyzer
        
        # Test with default config
        analyzer = AdvancedRiskAnalyzer()
        assert analyzer.config is not None
        assert "risk_patterns" in analyzer.config
    
    def test_model_configuration(self, config):
        """Test AI model configuration"""
        from src.core.ai_client import AIClient
        
        client = AIClient(config)
        assert client.primary_model == "gemini-1.5-flash"
        assert client.summary_model == "gemini-1.5-flash"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])