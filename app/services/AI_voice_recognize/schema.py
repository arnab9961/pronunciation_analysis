from pydantic import BaseModel, Field
from typing import List, Optional


class PronunciationMistake(BaseModel):
    word: str = Field(..., description="The mispronounced word")
    expected_pronunciation: str = Field(..., description="Expected pronunciation from document")
    actual_pronunciation: str = Field(..., description="Actual pronunciation from audio")
    position: int = Field(..., description="Position of the word in the text")
    confidence: float = Field(..., description="Confidence score of the mismatch")


class PronunciationAnalysisResponse(BaseModel):
    document_text: str = Field(..., description="Extracted text from document")
    transcribed_text: str = Field(..., description="Transcribed text from audio")
    pronunciation_mistakes: List[PronunciationMistake] = Field(..., description="List of pronunciation mistakes")
    mistakes_summary: List[str] = Field(..., description="List of unique mispronounced words")
    accuracy_score: float = Field(..., description="Overall pronunciation accuracy score (0-100)")
    total_words: int = Field(..., description="Total number of words analyzed")
    mistakes_count: int = Field(..., description="Number of pronunciation mistakes found")
