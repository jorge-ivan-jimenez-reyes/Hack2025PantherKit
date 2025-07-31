from fastapi import APIRouter, HTTPException, Query
import json
from pathlib import Path
import os
import logging
from typing import List, Dict, Optional

from app.services.llm_service import LLMService
from app.services.llm_api_service import LLMApiService
from app.services.neural_service import NeuralCareerService
from app.services.llm_profile_interpreter import LLMProfileInterpreter
from app.schemas.personality import QuestionResponse, LLMResponse, MBTIResult, MIResult
from app.db.mongodb_models import save_user_response

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("questions_api")

router = APIRouter()
llm_service = LLMService()
llm_api_service = LLMApiService()
neural_service = NeuralCareerService()

@router.get("/mbti")
async def get_mbti_questions():
    """
    Get all MBTI questions
    """
    try:
        # Path to the questions data
        data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../.." / "data" / "mbti_questions.json"
        
        # Check if file exists
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="MBTI questions file not found")
            
        # Read the questions
        with open(data_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        return {"mbti_questions": questions}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error loading MBTI questions: {str(e)}")

@router.get("/multiple-intelligence")
async def get_mi_questions():
    """
    Get all Multiple Intelligence questions
    """
    try:
        # Path to the questions data
        data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../.." / "data" / "mi_questions.json"
        
        # Check if file exists
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="MI questions file not found")
            
        # Read the questions
        with open(data_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        return {"mi_questions": questions}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error loading MI questions: {str(e)}")

@router.post("/save-response")
async def save_question_response(response: QuestionResponse):
    """
    Save a user's response to a question in MongoDB
    """
    try:
        # Prepare response data for MongoDB
        response_data = {
            "session_id": response.session_id,
            "question_type": response.question_type,
            "question_id": response.question_id,
            "response": response.response,
            "response_value": response.response_value,
            "intelligence_type": getattr(response, 'intelligence_type', None),
            "dimension": getattr(response, 'dimension', None)
        }
        
        # Save to MongoDB
        response_id = await save_user_response(response_data)
        
        return {
            "message": "Response saved successfully",
            "response_id": response_id
        }
        
    except Exception as e:
        logger.error(f"Error saving response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving response: {str(e)}")

@router.get("/health")
async def questions_health_check():
    """
    Health check for questions endpoint
    """
    return {
        "status": "healthy",
        "service": "questions",
        "endpoints": ["mbti", "multiple-intelligence", "save-response"]
    } 