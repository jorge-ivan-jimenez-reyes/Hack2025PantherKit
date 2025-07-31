from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from app.schemas.personality import ProfileData
from app.services.neural_service import NeuralCareerService
from app.services.llm_service import LLMService
from app.services.llm_api_service import LLMApiService
from app.db.mongodb_models import save_user_profile, save_career_recommendation

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction_api")

router = APIRouter()

# Servicios
def get_neural_service():
    return NeuralCareerService()

def get_llm_service():
    return LLMService()

def get_llm_api_service():
    return LLMApiService()

@router.post("/recommend_careers")
async def recommend_careers(
    profile_data: ProfileData,
    neural_service: NeuralCareerService = Depends(get_neural_service),
    llm_service: LLMService = Depends(get_llm_service),
    llm_api_service: LLMApiService = Depends(get_llm_api_service)
):
    """Recomienda carreras STEM basadas en el perfil MBTI y MI del usuario"""
    logger.info("Iniciando proceso de recomendación de carreras")
    logger.info(f"Perfil recibido: MBTI={profile_data.mbti_result}, MI disponibles={len(profile_data.mi_scores)}")
    
    try:
        # Convertir vector MBTI
        mbti_vector = [
            1 if profile_data.mbti_result.ei == "I" else 0,
            1 if profile_data.mbti_result.sn == "N" else 0,
            1 if profile_data.mbti_result.tf == "F" else 0,
            1 if profile_data.mbti_result.jp == "P" else 0
        ]
        logger.info(f"Vector MBTI creado: {mbti_vector}")
        
        # Extraer pesos MBTI
        mbti_weights = {
            "E/I": profile_data.mbti_result.ei_score,
            "S/N": profile_data.mbti_result.sn_score,
            "T/F": profile_data.mbti_result.tf_score,
            "J/P": profile_data.mbti_result.jp_score
        }
        logger.info(f"Pesos MBTI: {mbti_weights}")
        
        # Obtener scores MI
        mi_scores = profile_data.mi_scores
        logger.info(f"MI scores disponibles: {list(mi_scores.keys())}")
        
        # Realizar predicción con la RED NEURONAL (siempre)
        logger.info("Realizando predicción con red neuronal...")
        recommendations = neural_service.predict_careers(
            mbti_code=profile_data.mbti_result.MBTI_code,
            mbti_vector=mbti_vector,
            mbti_weights=mbti_weights,
            mi_scores=mi_scores,
            top_n=profile_data.num_recommendations or 5
        )
        logger.info(f"Recomendaciones generadas: {len(recommendations)}")
        
        # Guardar perfil y recomendaciones en MongoDB
        try:
            profile_data_dict = {
                "session_id": getattr(profile_data, 'session_id', f"session_{profile_data.mbti_result.MBTI_code}"),
                "mbti_code": profile_data.mbti_result.MBTI_code,
                "mbti_vector": mbti_vector,
                "mbti_weights": mbti_weights,
                "mi_scores": mi_scores
            }
            
            profile_id = await save_user_profile(profile_data_dict)
            logger.info(f"Perfil guardado con ID: {profile_id}")
            
            # Guardar recomendaciones
            recommendation_data = {
                "user_profile_id": profile_id,
                "session_id": profile_data_dict["session_id"],
                "recommendations": recommendations,
                "model_used": "neural_cnn",
                "model_version": "1.0",
                "prediction_confidence": max([r["match_score"] for r in recommendations]) if recommendations else 0.0,
                "top_scores": [r["match_score"] for r in recommendations]
            }
            
            recommendation_id = await save_career_recommendation(recommendation_data)
            logger.info(f"Recomendaciones guardadas con ID: {recommendation_id}")
            
        except Exception as db_error:
            logger.warning(f"Error guardando en BD: {str(db_error)}")
            # No fallar la predicción por errores de BD
        
        # Si se solicita análisis, generar con LLM
        if getattr(profile_data, 'include_analysis', False):
            logger.info("Generando análisis con LLM...")
            
            # Crear texto MI para el prompt
            mi_sorted = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            mi_text = "\n".join([f"- {name}: {score:.2f}" for name, score in mi_sorted])
            logger.info(f"MI texto para análisis: {mi_text[:100]}...")
            
            # Generar prompt para análisis
            prompt = llm_service.generate_career_analysis_prompt(
                mbti_code=profile_data.mbti_result.MBTI_code,
                mi_scores=mi_scores,
                career_recommendations=recommendations
            )
            logger.info(f"Prompt para análisis generado: {len(prompt)} caracteres")
            
            # Llamar al LLM para análisis
            llm_analysis = await llm_api_service.call_llm(prompt=prompt)
            logger.info(f"Análisis recibido del LLM: {len(llm_analysis)} caracteres")
            
            # Procesar respuesta
            analysis_result = llm_service.process_career_analysis_response(llm_analysis)
            logger.info("Análisis procesado correctamente")
            
            # Añadir análisis a la respuesta
            response = {
                "recommendations": recommendations,
                "analysis": analysis_result["analysis"]
            }
        else:
            logger.info("No se solicitó análisis LLM")
            # Solo devolver recomendaciones
            response = {
                "recommendations": recommendations
            }
        
        logger.info("Proceso de recomendación completado exitosamente")
        return response
        
    except Exception as e:
        logger.error(f"Error en recomendación de carreras: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generando recomendaciones: {str(e)}"
        )

@router.get("/health")
async def prediction_health_check():
    """Health check for prediction endpoint"""
    return {
        "status": "healthy",
        "service": "prediction",
        "model": "neural_cnn"
    } 