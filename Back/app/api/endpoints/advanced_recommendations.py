from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from app.schemas.personality import MBTIResult, MIResult, CareerMatch, ProfileData
from app.services.advanced_service import AdvancedCareerService
from app.db.mongodb_models import (
    get_database, save_user_profile, save_career_recommendation,
    save_model_metrics, get_user_profile_by_session
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_recommendations")

router = APIRouter()

# Instancia global del servicio avanzado
advanced_service = AdvancedCareerService()

@router.post("/predict-careers", response_model=Dict[str, Any])
async def predict_careers_advanced(
    profile_data: ProfileData,
    top_n: Optional[int] = Query(5, description="Número de recomendaciones a devolver"),
    save_to_db: Optional[bool] = Query(True, description="Guardar resultados en base de datos")
):
    """
    Predice carreras usando el modelo neural avanzado (Transformer) 
    con datos de Self-regulated learning.
    """
    try:
        logger.info(f"Iniciando predicción avanzada para perfil MBTI: {profile_data.mbti_result.MBTI_code}")
        
        # Convertir vector MBTI
        mbti_vector = [
            1 if profile_data.mbti_result.ei == "I" else 0,
            1 if profile_data.mbti_result.sn == "N" else 0,
            1 if profile_data.mbti_result.tf == "F" else 0,
            1 if profile_data.mbti_result.jp == "P" else 0
        ]
        
        # Extraer pesos MBTI
        mbti_weights = {
            "E/I": profile_data.mbti_result.ei_score,
            "S/N": profile_data.mbti_result.sn_score,
            "T/F": profile_data.mbti_result.tf_score,
            "J/P": profile_data.mbti_result.jp_score
        }
        
        # Realizar predicción con el servicio avanzado
        recommendations = advanced_service.predict_careers(
            mbti_code=profile_data.mbti_result.MBTI_code,
            mbti_vector=mbti_vector,
            mbti_weights=mbti_weights,
            mi_scores=profile_data.mi_scores,
            top_n=top_n
        )
        
        logger.info(f"Generadas {len(recommendations)} recomendaciones")
        
        # Guardar en base de datos si se solicita
        if save_to_db:
            try:
                # Guardar perfil de usuario
                profile_data_dict = {
                    "session_id": getattr(profile_data, 'session_id', f"session_{datetime.now().isoformat()}"),
                    "mbti_code": profile_data.mbti_result.MBTI_code,
                    "mbti_vector": mbti_vector,
                    "mbti_weights": mbti_weights,
                    "mi_scores": profile_data.mi_scores
                }
                
                profile_id = await save_user_profile(profile_data_dict)
                logger.info(f"Perfil guardado con ID: {profile_id}")
                
                # Guardar recomendaciones
                recommendation_data = {
                    "user_profile_id": profile_id,
                    "session_id": profile_data_dict["session_id"],
                    "recommendations": recommendations,
                    "model_used": "advanced_transformer",
                    "model_version": "1.0",
                    "prediction_confidence": max([r["match_score"] for r in recommendations]) if recommendations else 0.0,
                    "top_scores": [r["match_score"] for r in recommendations]
                }
                
                recommendation_id = await save_career_recommendation(recommendation_data)
                logger.info(f"Recomendaciones guardadas con ID: {recommendation_id}")
                
            except Exception as db_error:
                logger.warning(f"Error guardando en BD: {str(db_error)}")
                # No fallar la predicción por errores de BD
        
        # Obtener información del modelo
        model_info = advanced_service.get_model_info()
        
        return {
            "recommendations": recommendations,
            "model_info": {
                "type": model_info["model_type"],
                "architecture": model_info["architecture"],
                "data_source": model_info["data_source"],
                "is_trained": model_info["is_trained"]
            },
            "profile_summary": {
                "mbti_code": profile_data.mbti_result.MBTI_code,
                "dominant_mi": max(profile_data.mi_scores.items(), key=lambda x: x[1])[0],
                "top_mi_score": max(profile_data.mi_scores.values())
            },
            "prediction_metadata": {
                "num_recommendations": len(recommendations),
                "top_score": max([r["match_score"] for r in recommendations]) if recommendations else 0.0,
                "model_confidence": "high" if recommendations and max([r["match_score"] for r in recommendations]) > 0.7 else "medium"
            }
        }
        
    except Exception as e:
        logger.error(f"Error en predicción avanzada: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@router.post("/train-model", response_model=Dict[str, Any])
async def train_advanced_model(
    num_samples: Optional[int] = Query(10000, description="Número de muestras para entrenamiento"),
    epochs: Optional[int] = Query(80, description="Número de epochs"),
    batch_size: Optional[int] = Query(64, description="Tamaño del batch")
):
    """
    Entrena el modelo neural avanzado con datos de Self-regulated learning.
    """
    try:
        logger.info(f"Iniciando entrenamiento con {num_samples} muestras, {epochs} epochs")
        
        # Entrenar modelo
        training_results = advanced_service.train_model(
            num_samples=num_samples,
            epochs=epochs,
            batch_size=batch_size,
            validation=True
        )
        
        if "error" in training_results:
            raise HTTPException(status_code=500, detail=training_results["error"])
        
        # Guardar métricas en base de datos
        try:
            metrics_data = {
                "model_type": "advanced_transformer",
                "model_version": "1.0",
                "training_accuracy": training_results.get("final_val_accuracy", 0.0),
                "validation_accuracy": training_results.get("final_val_accuracy", 0.0),
                "test_accuracy": training_results.get("test_accuracy"),
                "training_loss": training_results.get("final_val_loss", 0.0),
                "validation_loss": training_results.get("final_val_loss", 0.0),
                "test_loss": training_results.get("test_loss"),
                "model_config": {
                    "embed_dim": advanced_service.neural_model.embed_dim,
                    "num_heads": advanced_service.neural_model.num_heads,
                    "ff_dim": advanced_service.neural_model.ff_dim,
                    "num_transformer_blocks": advanced_service.neural_model.num_transformer_blocks
                },
                "training_config": {
                    "num_samples": num_samples,
                    "epochs": epochs,
                    "batch_size": batch_size
                },
                "num_training_samples": num_samples,
                "num_classes": training_results.get("num_classes", 0),
                "epochs_trained": training_results.get("epochs_trained", 0)
            }
            
            metrics_id = await save_model_metrics(metrics_data)
            logger.info(f"Métricas guardadas con ID: {metrics_id}")
            
        except Exception as db_error:
            logger.warning(f"Error guardando métricas: {str(db_error)}")
        
        return {
            "message": "Modelo entrenado exitosamente",
            "training_results": training_results,
            "model_summary": advanced_service.neural_model.get_model_summary() if advanced_service.neural_model.model else "No disponible"
        }
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")

@router.get("/model-info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Obtiene información detallada sobre el modelo neural avanzado.
    """
    try:
        model_info = advanced_service.get_model_info()
        
        # Obtener métricas más recientes de la base de datos
        try:
            from app.db.mongodb_models import get_latest_model_metrics
            latest_metrics = await get_latest_model_metrics("advanced_transformer")
            
            if latest_metrics:
                model_info["latest_training_metrics"] = {
                    "training_accuracy": latest_metrics.get("training_accuracy"),
                    "validation_accuracy": latest_metrics.get("validation_accuracy"),
                    "test_accuracy": latest_metrics.get("test_accuracy"),
                    "trained_at": latest_metrics.get("trained_at"),
                    "num_samples": latest_metrics.get("num_training_samples"),
                    "epochs": latest_metrics.get("epochs_trained")
                }
        except Exception as db_error:
            logger.warning(f"Error obteniendo métricas de BD: {str(db_error)}")
            
        return model_info
        
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/retrain-model", response_model=Dict[str, Any])
async def retrain_model(
    num_samples: Optional[int] = Query(15000, description="Número de muestras para re-entrenamiento")
):
    """
    Re-entrena el modelo con nuevos datos generados.
    """
    try:
        logger.info(f"Re-entrenando modelo con {num_samples} muestras")
        
        results = advanced_service.retrain_with_new_data(num_samples=num_samples)
        
        return {
            "message": "Modelo re-entrenado exitosamente",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error en re-entrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Verifica el estado del servicio y la conexión a la base de datos.
    """
    try:
        # Verificar conexión a MongoDB
        db = await get_database()
        await db.command("ping")
        
        # Verificar estado del modelo
        model_status = "trained" if advanced_service.neural_model.model is not None else "not_trained"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model_status": model_status,
            "service": "advanced_career_recommendations",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}") 