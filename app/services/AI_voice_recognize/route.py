from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
from pathlib import Path
from app.services.AI_voice_recognize.service import pronunciation_service
from app.services.AI_voice_recognize.schema import PronunciationAnalysisResponse

router = APIRouter(prefix="/pronunciation", tags=["Pronunciation Analysis"])

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/analyze", response_model=PronunciationAnalysisResponse)
async def analyze_pronunciation(
    document: UploadFile = File(..., description="PDF or PNG file containing the text to be read"),
    audio: UploadFile = File(..., description="Audio file of the person reading the document")
):
    """
    Analyze pronunciation by comparing document text with audio transcription.
    
    - **document**: Upload a PDF or PNG file (the reference text). PDFs will be converted to images for OCR.
    - **audio**: Upload an audio file (MP3, WAV, M4A, etc.) of someone reading the document
    
    Returns detailed analysis of pronunciation mistakes and accuracy score.
    """
    document_path = None
    audio_path = None
    
    try:
        # Validate document file type
        doc_extension = os.path.splitext(document.filename)[1].lower()
        if doc_extension not in ['.pdf', '.png']:
            raise HTTPException(
                status_code=400, 
                detail="Document must be a PDF or PNG file"
            )
        
        # Validate audio file type
        audio_extension = os.path.splitext(audio.filename)[1].lower()
        if audio_extension not in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm']:
            raise HTTPException(
                status_code=400,
                detail="Audio must be MP3, WAV, M4A, OGG, FLAC, or WEBM format"
            )
        
        # Save document file
        document_path = UPLOAD_DIR / f"doc_{document.filename}"
        with open(document_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
        
        # Save audio file
        audio_path = UPLOAD_DIR / f"audio_{audio.filename}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Analyze pronunciation
        result = await pronunciation_service.analyze_pronunciation_from_files(
            document_path=str(document_path),
            audio_path=str(audio_path),
            doc_extension=doc_extension
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    
    finally:
        # Clean up uploaded files
        if document_path and os.path.exists(document_path):
            os.remove(document_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the pronunciation analysis service
    """
    return {"status": "healthy", "service": "pronunciation-analysis"}
