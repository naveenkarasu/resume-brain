from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import settings
from models.requests import QuickAnalyzeRequest
from models.responses import AnalysisResponse
from services import pdf_parser, resume_analyzer

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
):
    # Validate file type
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read and validate size
    content = await resume_file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    if len(job_description) > 10000:
        raise HTTPException(status_code=400, detail="Job description too long (max 10000 chars)")

    # Extract text from PDF
    try:
        resume_text = pdf_parser.extract_text(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse PDF file")

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    return await resume_analyzer.analyze(resume_text, job_description)


@router.post("/analyze/quick", response_model=AnalysisResponse)
async def analyze_quick(request: QuickAnalyzeRequest):
    return await resume_analyzer.analyze(request.resume_text, request.job_description)
