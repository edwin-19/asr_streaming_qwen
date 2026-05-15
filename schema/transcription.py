from pydantic import BaseModel, Field
from typing import List, Optional

class WordSegment(BaseModel):
    word: str
    start: float
    end: float

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    words: List[WordSegment]