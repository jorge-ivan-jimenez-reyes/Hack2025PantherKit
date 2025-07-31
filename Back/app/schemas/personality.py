from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class MBTIQuestion(BaseModel):
    question_id: int
    dimension: str  # E/I, S/N, T/F, J/P
    user_choice: str  # E or I, S or N, etc.
    weight: float = Field(..., ge=0.0, le=1.0)  # between 0 and 1

class MBTIResult(BaseModel):
    MBTI_code: str  # e.g., "INTP"
    MBTI_vector: List[int]  # binary representation, e.g., [0, 1, 0, 0]
    MBTI_weights: Dict[str, float]  # e.g., {"E/I": 0.85, "S/N": 0.62, ...}
    
    # Campos individuales para compatibilidad
    ei: str = Field(..., description="E or I")  # Extraversion/Introversion
    sn: str = Field(..., description="S or N")  # Sensing/Intuition
    tf: str = Field(..., description="T or F")  # Thinking/Feeling
    jp: str = Field(..., description="J or P")  # Judging/Perceiving

class MIResult(BaseModel):
    MI_scores: Dict[str, float]  # e.g., {"Lin": 0.7, "LogMath": 0.9, ...}

class CareerPrediction(BaseModel):
    top_predictions: List[str]
    confidence: List[float]

class Career(BaseModel):
    nombre: str
    universidad: str
    descripcion: str
    ubicacion: str
    
class CareerMatch(BaseModel):
    nombre: str
    universidad: str
    ciudad: str
    match_score: float

class ProfileData(BaseModel):
    """Esquema para datos de perfil de usuario con MBTI y MI"""
    mbti_result: MBTIResult
    mi_scores: Dict[str, float]
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class UserProfile(BaseModel):
    mbti_result: Optional[MBTIResult] = None
    mi_result: Optional[MIResult] = None
    career_recommendations: Optional[List[CareerMatch]] = None
    profile_description: Optional[str] = None

class QuestionResponse(BaseModel):
    pregunta: str
    respuesta: str

class UserResponseCreate(BaseModel):
    responses: List[QuestionResponse]
    session_id: Optional[str] = None
    user_id: Optional[int] = None

class UserResponseDB(BaseModel):
    id: int
    session_id: str
    responses_data: List[Dict[str, str]]
    created_at: str
    user_id: Optional[int] = None
    
    class Config:
        orm_mode = True

class MBTIWeightDetail(BaseModel):
    value: str  # "I fuerte", "N medio", etc.
    score: Optional[float] = None

class LLMResultCreate(BaseModel):
    mbti_result: str  # e.g., "INTP"
    mbti_vector: Optional[List[float]] = None  # e.g., [0.1, 0.8, 0.2, 0.7]
    mbti_weights: Dict[str, str]  # e.g., {"E/I": "I fuerte", "S/N": "N medio", ...}
    mi_ranking: List[str]  # e.g., ["Espacial", "Interpersonal", ...]
    full_analysis: Optional[Dict[str, Any]] = None

class LLMResultDB(BaseModel):
    id: int
    user_response_id: int
    mbti_result: str
    mbti_vector: Optional[List[float]] = None
    mbti_weights: Dict[str, Any]
    mi_ranking: List[str]
    full_result: Dict[str, Any]
    prompt_used: str
    created_at: str
    user_id: Optional[int] = None
    
    class Config:
        orm_mode = True

class LLMRequestPrompt(BaseModel):
    responses: List[QuestionResponse]
    
class LLMResponse(BaseModel):
    status: str
    message: str
    prompt: str
    raw_data: List[Dict[str, str]] 