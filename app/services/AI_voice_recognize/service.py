import os
import base64
from typing import List, Tuple
from openai import OpenAI
from difflib import SequenceMatcher
import re
from app.core.config import settings
from app.services.AI_voice_recognize.schema import PronunciationMistake, PronunciationAnalysisResponse


class PronunciationAnalysisService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    async def extract_text_from_document(self, file_path: str, file_extension: str) -> str:
        """
        Extract text from PDF or DOC using OpenAI Vision API with OCR capability
        """
        try:
            # Read the file and encode it to base64
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # For PDF/DOC/PNG, we'll use OpenAI's vision API with image conversion
            # Note: For production, you might want to convert PDF pages to images first
            # Using a library like pdf2image or PyMuPDF
            
            # Determine the appropriate MIME type
            if file_extension.lower() == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'application/pdf'
            
            # If it's a supported format, use OpenAI Vision API
            if file_extension.lower() in ['.pdf', '.doc', '.docx', '.png']:
                base64_document = base64.b64encode(file_content).decode('utf-8')
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all the text from this document. Return only the text content, no additional commentary."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_document}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000
                )
                
                extracted_text = response.choices[0].message.content
                return extracted_text.strip()
            
            return ""
            
        except Exception as e:
            raise Exception(f"Error extracting text from document: {str(e)}")
    
    async def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file using OpenAI Whisper model
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return transcription.strip()
            
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    def normalize_text(self, text: str) -> List[str]:
        """
        Normalize text by removing punctuation and converting to lowercase
        """
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip().split()
    
    def analyze_pronunciation(self, expected_text: str, actual_text: str) -> Tuple[List[PronunciationMistake], float]:
        """
        Compare expected text with transcribed text to find pronunciation mistakes
        """
        expected_words = self.normalize_text(expected_text)
        actual_words = self.normalize_text(actual_text)
        
        mistakes = []
        matcher = SequenceMatcher(None, expected_words, actual_words)
        
        # Get matching blocks
        matching_blocks = matcher.get_matching_blocks()
        
        expected_idx = 0
        actual_idx = 0
        
        for match in matching_blocks:
            # Check for mismatches before this matching block
            while expected_idx < match.a or actual_idx < match.b:
                if expected_idx < match.a and actual_idx < match.b:
                    # Both have words, but they don't match
                    expected_word = expected_words[expected_idx]
                    actual_word = actual_words[actual_idx]
                    
                    # Calculate similarity
                    similarity = SequenceMatcher(None, expected_word, actual_word).ratio()
                    
                    if similarity < 0.8:  # Threshold for considering it a mistake
                        mistakes.append(PronunciationMistake(
                            word=expected_word,
                            expected_pronunciation=expected_word,
                            actual_pronunciation=actual_word,
                            position=expected_idx,
                            confidence=1.0 - similarity
                        ))
                    
                    expected_idx += 1
                    actual_idx += 1
                elif expected_idx < match.a:
                    # Word was skipped in audio
                    expected_word = expected_words[expected_idx]
                    mistakes.append(PronunciationMistake(
                        word=expected_word,
                        expected_pronunciation=expected_word,
                        actual_pronunciation="[skipped]",
                        position=expected_idx,
                        confidence=1.0
                    ))
                    expected_idx += 1
                else:
                    # Extra word in audio (not in document)
                    actual_idx += 1
            
            # Move past the matching block
            expected_idx = match.a + match.size
            actual_idx = match.b + match.size
        
        # Calculate accuracy score
        total_words = len(expected_words)
        mistakes_count = len(mistakes)
        accuracy_score = ((total_words - mistakes_count) / total_words * 100) if total_words > 0 else 0
        
        return mistakes, accuracy_score
    
    def create_mistakes_summary(self, mistakes: List[PronunciationMistake]) -> List[str]:
        """
        Create a list of unique mispronounced words
        """
        unique_words = list(set(mistake.word for mistake in mistakes))
        return sorted(unique_words)
    
    async def analyze_pronunciation_from_files(
        self, 
        document_path: str, 
        audio_path: str,
        doc_extension: str
    ) -> PronunciationAnalysisResponse:
        """
        Main method to analyze pronunciation from document and audio files
        """
        try:
            # Extract text from document
            document_text = await self.extract_text_from_document(document_path, doc_extension)
            
            # Transcribe audio
            transcribed_text = await self.transcribe_audio(audio_path)
            
            # Analyze pronunciation
            mistakes, accuracy_score = self.analyze_pronunciation(document_text, transcribed_text)
            
            # Create mistakes summary
            mistakes_summary = self.create_mistakes_summary(mistakes)
            
            # Create response
            response = PronunciationAnalysisResponse(
                document_text=document_text,
                transcribed_text=transcribed_text,
                pronunciation_mistakes=mistakes,
                mistakes_summary=mistakes_summary,
                accuracy_score=accuracy_score,
                total_words=len(self.normalize_text(document_text)),
                mistakes_count=len(mistakes)
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"Error analyzing pronunciation: {str(e)}")


# Create a singleton instance
pronunciation_service = PronunciationAnalysisService()
