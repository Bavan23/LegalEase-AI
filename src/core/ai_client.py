"""
Enhanced Legal Document Demystifier - AI Client
Production-ready AI integration with caching, error handling, and multi-model support
"""
import os
import asyncio
import hashlib
import logging
import json
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

load_dotenv()  # reads .env
api_key = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)
genai.configure(api_key=api_key)
@dataclass
class AIResponse:
    """Structured AI response"""
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    cached: bool = False
    confidence: float = 1.0

class AIClient:
    """Production-ready AI client with advanced features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = config.get('models', {})
        self.cache_enabled = config.get('cache', {}).get('enabled', True)
        self.cache_ttl = config.get('cache', {}).get('ttl', 3600)
        
        # Initialize models
        self.primary_model = self.models.get('primary', {}).get('reasoning', 'gemini-1.5-flash')
        self.summary_model = self.models.get('primary', {}).get('summary', 'gemini-1.5-flash')
        self.fallback_model = self.models.get('primary', {}).get('fallback', 'gemini-1.5-flash')
        
        # Cache storage (in production, use Redis or similar)
        self._cache = {}
        self._cache_timestamps = {}
        
        # Rate limiting
        self._request_times = []
        self._max_requests_per_minute = 60
        
        # System prompts
        self.system_prompts = {
            'legal_explanation': (
                "You are a clear, neutral assistant that explains legal text to non-lawyers. "
                "Keep language simple (short sentences), avoid giving legal advice, and if you use "
                "document text quote short excerpts and cite which page/clause they came from."
            ),
            'risk_analysis': (
                "You are a legal risk analyst. Analyze the given text for potential risks, "
                "focusing on unfavorable terms, unusual clauses, and compliance issues. "
                "Provide specific, actionable insights."
            ),
            'multilingual': (
                "You are a multilingual legal assistant. Explain legal concepts in simple, "
                "conversational language that a common person can understand. Use relatable "
                "examples from Indian context when helpful."
            )
        }
    
    async def explain_clause(
        self, 
        clause_text: str, 
        clause_id: str, 
        page: int,
        risk_level: str = "UNKNOWN",
        language: str = "english",
        use_storytelling: bool = False
    ) -> AIResponse:
        """
        Generate explanation for a legal clause
        
        Args:
            clause_text: The clause text to explain
            clause_id: Unique identifier for the clause
            page: Page number where clause appears
            risk_level: Risk level (RED, YELLOW, GREEN)
            language: Target language for explanation
            use_storytelling: Whether to use storytelling approach
            
        Returns:
            AIResponse with explanation
        """
        # Create cache key
        cache_key = self._create_cache_key(
            "explain_clause", clause_text, language, use_storytelling, risk_level
        )
        
        # Check cache first
        if self.cache_enabled:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare prompt
        prompt = self._build_explanation_prompt(
            clause_text, clause_id, page, risk_level, language, use_storytelling
        )
        
        # Generate response
        try:
            response = await self._generate_content(
                prompt, 
                model_name=self.primary_model,
                temperature=0.3 if use_storytelling else 0.2
            )
            
            # Cache the response
            if self.cache_enabled:
                self._store_in_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to explain clause {clause_id}: {str(e)}")
            raise
    
    async def summarize_document(
        self, 
        document_text: str, 
        language: str = "english",
        summary_type: str = "comprehensive"
    ) -> AIResponse:
        """
        Generate document summary
        
        Args:
            document_text: Full document text
            language: Target language
            summary_type: Type of summary (brief, comprehensive, risk-focused)
            
        Returns:
            AIResponse with summary
        """
        cache_key = self._create_cache_key(
            "summarize_document", document_text[:1000], language, summary_type
        )
        
        # Check cache
        if self.cache_enabled:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare prompt based on summary type
        prompt = self._build_summary_prompt(document_text, language, summary_type)
        
        try:
            response = await self._generate_content(
                prompt,
                model_name=self.summary_model,
                temperature=0.15,
                max_output_tokens=1500
            )
            
            if self.cache_enabled:
                self._store_in_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to summarize document: {str(e)}")
            raise
    
    async def answer_question(
        self, 
        question: str, 
        context_clauses: List[Dict],
        language: str = "english"
    ) -> AIResponse:
        """
        Answer questions about the document using context
        
        Args:
            question: User's question
            context_clauses: Relevant clauses for context
            language: Target language for response
            
        Returns:
            AIResponse with answer
        """
        # Build context string
        context_parts = []
        for i, clause in enumerate(context_clauses, 1):
            risk_info = ""
            if 'risk_level' in clause:
                risk_icon = {"RED": "🚨", "YELLOW": "⚠️", "GREEN": "✅"}.get(clause['risk_level'], "❓")
                risk_info = f" {risk_icon} ({clause['risk_level']} risk)"
            
            context_parts.append(
                f"[{i}]{risk_info} (p{clause.get('page', 'N/A')}, id {clause.get('id', 'N/A')}) "
                f"{clause.get('text', '')}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = self._build_qa_prompt(question, context, language)
        
        try:
            response = await self._generate_content(
                prompt,
                model_name=self.primary_model,
                temperature=0.15
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise
    
    async def analyze_risk_narrative(
        self, 
        clause_text: str, 
        detected_risks: List[str],
        indian_law_warnings: List[str] = None
    ) -> AIResponse:
        """
        Generate narrative risk analysis
        
        Args:
            clause_text: The clause text
            detected_risks: List of detected risk factors
            indian_law_warnings: Indian law compliance warnings
            
        Returns:
            AIResponse with risk narrative
        """
        prompt = self._build_risk_analysis_prompt(
            clause_text, detected_risks, indian_law_warnings or []
        )
        
        try:
            response = await self._generate_content(
                prompt,
                model_name=self.primary_model,
                temperature=0.1,
                system_prompt=self.system_prompts['risk_analysis']
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to analyze risk narrative: {str(e)}")
            raise
    
    async def compare_documents(
        self, 
        doc1_summary: str, 
        doc2_summary: str,
        differences: Dict
    ) -> AIResponse:
        """
        Generate document comparison analysis
        
        Args:
            doc1_summary: Summary of first document
            doc2_summary: Summary of second document
            differences: Detected differences between documents
            
        Returns:
            AIResponse with comparison analysis
        """
        prompt = self._build_comparison_prompt(doc1_summary, doc2_summary, differences)
        
        try:
            response = await self._generate_content(
                prompt,
                model_name=self.primary_model,
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to compare documents: {str(e)}")
            raise
    
    def _build_explanation_prompt(
        self, 
        clause_text: str, 
        clause_id: str, 
        page: int,
        risk_level: str,
        language: str,
        use_storytelling: bool
    ) -> str:
        """Build prompt for clause explanation"""
        
        storytelling_instruction = ""
        if use_storytelling:
            storytelling_instruction = (
                "Use simple storytelling with Indian examples (like scenarios in Mumbai, Delhi, "
                "Chennai, etc.) to make the concept relatable. Make it feel like explaining to "
                "a friend over chai. Use analogies and real-world situations."
            )
        
        language_instruction = ""
        if language != "english":
            lang_names = {
                "hindi": "हिंदी", "tamil": "தமிழ்", "telugu": "తెలుగు", 
                "bengali": "বাংলা", "gujarati": "ગુજરાતી", "marathi": "मराठी", 
                "kannada": "ಕನ್ನಡ"
            }
            lang_name = lang_names.get(language, language)
            language_instruction = (
                f"Explain in {lang_name} language using simple words that a common person "
                f"can understand. Keep it conversational and friendly."
            )
        
        risk_context = ""
        if risk_level in ["RED", "YELLOW", "GREEN"]:
            risk_icons = {"RED": "🚨", "YELLOW": "⚠️", "GREEN": "✅"}
            risk_context = f"Note: This clause has been flagged as {risk_icons[risk_level]} {risk_level} risk. "
        
        prompt = f"""
        Explain the following legal clause in plain language (5-7 lines). 
        Then provide a one-line "What it means for you" summary.
        
        {risk_context}
        {storytelling_instruction}
        {language_instruction}
        
        Clause (page {page}, id {clause_id}):
        {clause_text}
        
        Structure your response as:
        1. Simple explanation
        2. What it means for you: [one clear line]
        
        Remember: This is informational only, not legal advice.
        """
        
        return prompt.strip()
    
    def _build_summary_prompt(
        self, 
        document_text: str, 
        language: str, 
        summary_type: str
    ) -> str:
        """Build prompt for document summary"""
        
        type_instructions = {
            "brief": "Provide a concise 3-bullet summary covering key obligations, payments, and risks.",
            "comprehensive": "Provide a detailed 5-bullet summary covering obligations, payments, key dates, renewal/termination, and risks.",
            "risk_focused": "Focus on potential risks, red flags, and areas requiring careful attention."
        }
        
        instruction = type_instructions.get(summary_type, type_instructions["comprehensive"])
        
        language_instruction = ""
        if language != "english":
            lang_names = {
                "hindi": "हिंदी", "tamil": "தமிழ்", "telugu": "తెలుగు", 
                "bengali": "বাংলা", "gujarati": "ગુજરાતী", "marathi": "मराठी", 
                "kannada": "ಕನ್ನಡ"
            }
            lang_name = lang_names.get(language, language)
            language_instruction = f"Use simple {lang_name} that a common person can understand."
        
        # Truncate document if too long
        max_chars = self.config.get('models', {}).get('settings', {}).get('max_text_chars', 120000)
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "\n\n[Document truncated for processing]"
        
        prompt = f"""
        Summarize this legal document. {instruction}
        {language_instruction}
        
        Document:
        {document_text}
        """
        
        return prompt.strip()
    
    def _build_qa_prompt(self, question: str, context: str, language: str) -> str:
        """Build prompt for question answering"""
        
        language_instruction = ""
        if language != "english":
            language_instruction = f"Answer in {language} using simple language."
        
        prompt = f"""
        Answer the following question using ONLY the context excerpts provided below. 
        Consider risk levels when relevant. If insufficient information is available, 
        say 'insufficient evidence in the provided context'.
        Cite excerpts using [1], [2], etc.
        {language_instruction}
        
        Question: {question}
        
        Context:
        {context}
        """
        
        return prompt.strip()
    
    def _build_risk_analysis_prompt(
        self, 
        clause_text: str, 
        detected_risks: List[str], 
        indian_law_warnings: List[str]
    ) -> str:
        """Build prompt for risk analysis narrative"""
        
        risks_text = "\n".join(f"- {risk}" for risk in detected_risks)
        warnings_text = "\n".join(f"- {warning}" for warning in indian_law_warnings)
        
        prompt = f"""
        Provide a clear risk analysis narrative for this legal clause.
        
        Clause Text:
        {clause_text}
        
        Detected Risk Factors:
        {risks_text}
        
        Indian Law Compliance Warnings:
        {warnings_text}
        
        Provide:
        1. Overall risk assessment (2-3 sentences)
        2. Specific concerns to watch out for
        3. Recommended actions or precautions
        
        Keep the language accessible to non-lawyers.
        """
        
        return prompt.strip()
    
    def _build_comparison_prompt(
        self, 
        doc1_summary: str, 
        doc2_summary: str, 
        differences: Dict
    ) -> str:
        """Build prompt for document comparison"""
        
        prompt = f"""
        Compare these two legal documents and highlight key differences:
        
        Document 1 Summary:
        {doc1_summary}
        
        Document 2 Summary:
        {doc2_summary}
        
        Detected Changes:
        - Added content items: {len(differences.get('added', []))}
        - Removed content items: {len(differences.get('removed', []))}
        - Total changes: {differences.get('total_changes', 0)}
        
        Provide:
        1. Overall comparison summary
        2. Key differences that matter
        3. Risk implications of the changes
        4. Recommendation on which version is more favorable
        
        Focus on practical implications rather than technical details.
        """
        
        return prompt.strip()
    
    async def _generate_content(
        self, 
        prompt: str, 
        model_name: str = None,
        temperature: float = 0.2,
        max_output_tokens: int = None,
        system_prompt: str = None
    ) -> AIResponse:
        """Generate content using Gemini API with error handling and fallback"""
        
        if model_name is None:
            model_name = self.primary_model
        
        # Rate limiting check
        await self._check_rate_limit()
        
        # Prepare generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens or 2048
        )
        
        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = self.system_prompts['legal_explanation']
        
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        start_time = time.time()
        
        try:
            # Try primary model
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [{"role": "user", "parts": [full_prompt]}],
                generation_config=generation_config
            )
            
            response_time = time.time() - start_time
            
            return AIResponse(
                content=response.text,
                model_used=model_name,
                tokens_used=self._estimate_tokens(full_prompt + response.text),
                response_time=response_time,
                cached=False
            )
            
        except Exception as e:
            logger.warning(f"Primary model {model_name} failed: {str(e)}")
            
            # Try fallback model
            if model_name != self.fallback_model:
                logger.info(f"Retrying with fallback model: {self.fallback_model}")
                try:
                    model = genai.GenerativeModel(self.fallback_model)
                    response = model.generate_content(
                        [{"role": "user", "parts": [full_prompt]}],
                        generation_config=generation_config
                    )
                    
                    response_time = time.time() - start_time
                    
                    return AIResponse(
                        content=response.text,
                        model_used=self.fallback_model,
                        tokens_used=self._estimate_tokens(full_prompt + response.text),
                        response_time=response_time,
                        cached=False
                    )
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {str(fallback_error)}")
                    raise fallback_error
            else:
                raise e
    
    async def _check_rate_limit(self):
        """Simple rate limiting implementation"""
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        self._request_times = [
            req_time for req_time in self._request_times 
            if current_time - req_time < 60
        ]
        
        # Check if we're at the limit
        if len(self._request_times) >= self._max_requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(current_time)
    
    def _create_cache_key(self, *args) -> str:
        """Create cache key from arguments"""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[AIResponse]:
        """Get response from cache if valid"""
        if cache_key not in self._cache:
            return None
        
        # Check if cache entry is still valid
        cache_time = self._cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self.cache_ttl:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        # Return cached response with cache flag
        cached_response = self._cache[cache_key]
        cached_response.cached = True
        return cached_response
    
    def _store_in_cache(self, cache_key: str, response: AIResponse):
        """Store response in cache"""
        self._cache[cache_key] = response
        self._cache_timestamps[cache_key] = time.time()
        
        # Simple cache size management
        if len(self._cache) > self.config.get('cache', {}).get('max_size', 1000):
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(), key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        # Very rough estimation: ~4 characters per token
        return len(text) // 4
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "requests_in_last_minute": len(self._request_times)
        }