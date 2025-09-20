import logging
import uuid
import tempfile
import aiofiles
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.core.document_processor import DocumentProcessor
from src.core.risk_analyzer import AdvancedRiskAnalyzer
from src.core.ai_client import AIClient
from src.utils.security import SecurityManager, PrivacyManager
from src.utils.validators import validate_file_size, validate_file_type
from src.utils.exceptions import DocumentProcessingError, SecurityError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DocumentAnalysisRequest(BaseModel):
    privacy_mode: str = "standard"
    language: str = "english"
    analysis_type: str = "comprehensive"

class QuestionRequest(BaseModel):
    question: str
    language: str = "english"

class ClauseExplanationRequest(BaseModel):
    clause_id: str
    language: str = "english"
    use_storytelling: bool = False

class ComparisonRequest(BaseModel):
    privacy_mode: str = "standard"

# Response models
class DocumentAnalysisResponse(BaseModel):
    success: bool
    session_id: str
    metadata: Dict
    risk_summary: Dict
    auto_summary: str
    privacy_notice: str

class ClauseExplanationResponse(BaseModel):
    explanation: str
    risk_level: str
    risk_factors: List[str]
    recommendations: List[str]
    legal_warnings: List[str]

# Create FastAPI app
app = FastAPI(
    title="Enhanced Legal Document Demystifier",
    description="AI-powered legal document analysis with advanced risk assessment",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Global configuration
CONFIG = {
    "max_file_size": 25 * 1024 * 1024,  # 25MB
    "allowed_types": [".pdf"],
    "models": {
        "primary": {
            "summary": "gemini-1.5-flash",
            "reasoning": "gemini-1.5-flash",
            "fallback": "gemini-1.5-flash"
        }
    },
    "cache": {
        "enabled": True,
        "ttl": 3600,
        "max_size": 1000
    }
}

# Global instances
document_processor = DocumentProcessor(CONFIG)
risk_analyzer = AdvancedRiskAnalyzer()
ai_client = AIClient(CONFIG)
security_manager = SecurityManager(CONFIG)
privacy_manager = PrivacyManager(CONFIG)

# Session storage (use Redis in production)
sessions: Dict[str, Dict] = {}

# Dependency to get current session
def get_session(session_id: str = None) -> Dict:
    if session_id and session_id in sessions:
        return sessions[session_id]
    return {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Legal Document Demystifier",
        "features": [
            "🚦 Risk Radar (Traffic Light System)",
            "🔍 Document Comparison",
            "🌐 Multilingual Support (8+ Languages)",
            "🚨 Red Flag Auto-Alert",
            "⚖️ Indian Law Cross-Reference",
            "🔒 Enhanced Privacy Mode",
            "📚 Gamified Storytelling Explanations"
        ]
    })

@app.post("/api/upload", response_model=DocumentAnalysisResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: DocumentAnalysisRequest = Depends()
):
    try:
        # Check file type using the original filename string (synchronous call)
        if not validate_file_type(file.filename, CONFIG["allowed_types"]):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Check file size directly from the UploadFile object's size attribute
        if file.size is None or file.size > CONFIG["max_file_size"]:
            raise HTTPException(status_code=400, detail="File too large")

        session_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.gettempdir()) / "legal_demystifier" / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename for the filesystem path
        file_path = temp_dir / security_manager.sanitize_filename(file.filename)
        
        # Read the file content and write it to disk
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Processing document: {file.filename}")
        
        # Pass both the file path and the original filename to the processor
        result = await document_processor.process_document(
            file_path,
            original_filename=file.filename,
            privacy_mode=request.privacy_mode,
            secure_hash=True
        )

        if not result["success"]:
            raise HTTPException(status_code=422, detail="Document processing failed")

        logger.info("Analyzing document risks")
        risk_analysis = await risk_analyzer.analyze_document_risk(result["clauses"])

        logger.info("Generating document summary")
        full_text = "\n\n".join(text for _, text in result["pages"])
        summary_response = await ai_client.summarize_document(
            full_text,
            language=request.language,
            summary_type=request.analysis_type
        )
        
        sessions[session_id] = {
            "file_name": file.filename,
            "pages": result["pages"],
            "clauses": result["clauses"],
            "metadata": result["metadata"],
            "risk_analysis": risk_analysis,
            "privacy_mode": request.privacy_mode,
            "language": request.language,
            "created_at": datetime.now().isoformat()
        }

        privacy_manager.log_data_access(
            action="document_upload",
            data_type="legal_document",
            user_id=session_id
        )

        background_tasks.add_task(cleanup_session, session_id, delay_minutes=60)
        privacy_notice = privacy_manager.get_privacy_notice(request.language)

        return DocumentAnalysisResponse(
            success=True,
            session_id=session_id,
            metadata=result["metadata"],
            risk_summary=risk_analysis,
            auto_summary=summary_response.content,
            privacy_notice=privacy_notice
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/session/{session_id}/clauses")
async def get_clauses(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    clauses = session.get("clauses", [])
    for clause in clauses:
        if 'risk_result' not in clause:
            clause['risk_result'] = await risk_analyzer.analyze_clause_risk(clause)
    return {"clauses": clauses}

@app.post("/api/session/{session_id}/explain", response_model=ClauseExplanationResponse)
async def explain_clause(
    session_id: str,
    request: ClauseExplanationRequest
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    clause = None
    for c in session.get("clauses", []):
        if c.get("id") == request.clause_id:
            clause = c
            break
    if not clause:
        raise HTTPException(status_code=404, detail="Clause not found")
    try:
        risk_result = await risk_analyzer.analyze_clause_risk(clause)
        explanation_response = await ai_client.explain_clause(
            clause["text"],
            clause["id"],
            clause["page"],
            risk_level=risk_result.level,
            language=request.language,
            use_storytelling=request.use_storytelling
        )
        return ClauseExplanationResponse(
            explanation=explanation_response.content,
            risk_level=risk_result.level,
            risk_factors=risk_result.risk_factors,
            recommendations=risk_result.recommendations,
            legal_warnings=risk_result.indian_law_warnings
        )
    except Exception as e:
        logger.error(f"Clause explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

from fastapi import status
@app.post("/api/session/{session_id}/question")
async def ask_question(session_id: str, request: QuestionRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        clauses = session.get("clauses", [])
        relevant_clauses = [
            c for c in clauses
            if request.question.lower() in c.get("text", "").lower()
        ][:5]

        if not relevant_clauses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant clauses found"
            )

        # Combine relevant clauses into one context string
        context_text = "\n\n".join(
            f"[Clause {c.get('id')} - Page {c.get('page')}] {c.get('text')}"
            for c in relevant_clauses
        )

        # Call AIClient with context text
        answer_response = await ai_client.answer_question(
            question=request.question,
            context=context_text,
            language=request.language
        )

        return {
            "answer": getattr(answer_response, "content", ""),
            "sources": [
                {
                    "clause_id": c.get("id"),
                    "page": c.get("page"),
                    "heading": c.get("heading"),
                    "preview": c.get("text", "")[:200] + "..."
                } for c in relevant_clauses
            ],
            "model_used": getattr(answer_response, "model_used", "unknown"),
            "cached": getattr(answer_response, "cached", False)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(e)}"
        )


@app.get("/api/session/{session_id}/risk-radar")
async def get_risk_radar(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    clauses = session.get("clauses", [])
    risk_summary = {"RED": 0, "YELLOW": 0, "GREEN": 0}
    risk_items = []
    for clause in clauses:
        risk_result = await risk_analyzer.analyze_clause_risk(clause)
        risk_summary[risk_result.level] += 1
        risk_items.append({
            "clause_id": clause.get("id"),
            "page": clause.get("page"),
            "heading": clause.get("heading") or f"Clause {clause.get('id')}",
            "risk_level": risk_result.level,
            "risk_factors": risk_result.risk_factors[:2],
            "severity": risk_result.severity
        })
    return {
        "risk_summary": risk_summary,
        "risk_items": risk_items[:20],
        "total_clauses": len(clauses)
    }

@app.post("/api/session/{session_id}/compare")
async def compare_documents(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: ComparisonRequest = Depends()
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        if not validate_file_type(file.filename, CONFIG["allowed_types"]):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if file.size is None or file.size > CONFIG["max_file_size"]:
            raise HTTPException(status_code=400, detail="File too large")
        
        temp_dir = Path(tempfile.gettempdir()) / "legal_demystifier" / f"{session_id}_compare"
        temp_dir.mkdir(parents=True, exist_ok=True)
        compare_file_path = temp_dir / security_manager.sanitize_filename(file.filename)
        
        async with aiofiles.open(compare_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Pass the original filename to the processor for correct validation
        compare_result = await document_processor.process_document(
            compare_file_path,
            original_filename=file.filename,
            privacy_mode=request.privacy_mode
        )

        compare_risk_analysis = await risk_analyzer.analyze_document_risk(
            compare_result["clauses"]
        )
        original_clauses = session.get("clauses", [])
        original_risk = session.get("risk_analysis", {})
        comparison_result = {
            "original_document": {
                "name": session.get("file_name"),
                "clauses_count": len(original_clauses),
                "risk_summary": original_risk.get("risk_breakdown", {}),
                "risk_score": original_risk.get("risk_score", 0)
            },
            "comparison_document": {
                "name": file.filename,
                "clauses_count": len(compare_result["clauses"]),
                "risk_summary": compare_risk_analysis.get("risk_breakdown", {}),
                "risk_score": compare_risk_analysis.get("risk_score", 0)
            },
            "risk_comparison": {
                "original_high_risk": original_risk.get("risk_breakdown", {}).get("RED", 0),
                "comparison_high_risk": compare_risk_analysis.get("risk_breakdown", {}).get("RED", 0),
                "risk_change": compare_risk_analysis.get("risk_score", 0) - original_risk.get("risk_score", 0)
            }
        }
        return comparison_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/api/session/{session_id}/export")
async def export_analysis(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        # For demo, just return session data
        return session
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"success": True}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Background tasks
async def cleanup_session(session_id: str, delay_minutes: int = 60):
    await asyncio.sleep(delay_minutes * 60)
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session {session_id} cleaned up after {delay_minutes} minutes")

async def cleanup_temp_files(temp_dir: Path):
    try:
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                file.unlink()
            temp_dir.rmdir()
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "active_sessions": len(sessions)
    }

@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=422,
        content={"detail": f"Document processing error: {str(exc)}"}
    )

@app.exception_handler(SecurityError)
async def security_exception_handler(request: Request, exc: SecurityError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Security error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
