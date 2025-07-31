import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.db.mongodb_models import save_user_response, get_database
from app.schemas.personality import QuestionResponse, LLMResultCreate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_service")

class LLMService:
    def __init__(self):
        logger.info("LLMService inicializado")
    
    def generate_career_analysis_prompt(self, mbti_code: str, mi_scores: Dict[str, float], 
                                      career_recommendations: List[Dict[str, Any]]) -> str:
        """
        Genera un prompt para que el LLM analice las recomendaciones de carrera
        
        Args:
            mbti_code: Código MBTI del usuario (ej. "INTJ")
            mi_scores: Puntuaciones de inteligencias múltiples
            career_recommendations: Lista de recomendaciones de carrera con puntuaciones
            
        Returns:
            Prompt para el LLM
        """
        logger.info(f"Generando prompt para análisis de carreras para perfil {mbti_code}")
        
        # Crear una representación ordenada de las puntuaciones MI
        mi_sorted = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        mi_text = "\n".join([f"- {name}: {score:.2f}" for name, score in mi_sorted])
        
        # Crear una representación de las recomendaciones
        rec_text = "\n".join([
            f"- {i+1}. {rec['nombre']} ({rec['universidad']}, {rec['ciudad']}): {rec['match_score']:.2f} match"
            for i, rec in enumerate(career_recommendations[:5])  # Limitar a 5 recomendaciones
        ])
        
        prompt = f"""Actúa como un consejero vocacional experto. Basándote en el perfil de personalidad MBTI, 
las puntuaciones de inteligencias múltiples y las recomendaciones de carrera proporcionadas, 
genera un análisis detallado y personalizado explicando:

1. Por qué estas carreras son adecuadas para este perfil específico
2. Cómo las características del perfil MBTI se alinean con cada carrera recomendada
3. Cómo las inteligencias múltiples del usuario se relacionan con las demandas de cada carrera
4. Qué habilidades específicas podría desarrollar el usuario para tener éxito en estas carreras
5. Oportunidades y desafíos potenciales que podría enfrentar en estos campos

Perfil MBTI: {mbti_code}

Inteligencias Múltiples (ordenadas de mayor a menor):
{mi_text}

Recomendaciones de Carrera:
{rec_text}

Proporciona un análisis detallado, personalizado y práctico que ayude al usuario a entender por qué estas 
carreras son compatibles con su perfil y cómo podría aprovechar sus fortalezas naturales.
"""
        
        logger.info(f"Prompt generado con longitud: {len(prompt)} caracteres")
        return prompt
    
    def process_career_analysis_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Procesa la respuesta del LLM sobre el análisis de carreras
        
        Args:
            llm_response: Respuesta del LLM en formato de texto
            
        Returns:
            Diccionario con el análisis procesado
        """
        logger.info("Procesando respuesta de análisis de carreras del LLM")
        logger.info(f"Longitud de la respuesta: {len(llm_response)} caracteres")
        
        # Simplemente devolvemos la respuesta como texto, no necesitamos procesarla
        return {
            "analysis": llm_response.strip(),
            "raw_response": llm_response
        }
    
    async def save_user_responses(self, responses: List[QuestionResponse], 
                                 user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Guarda las respuestas del usuario en MongoDB
        
        Args:
            responses: Lista de respuestas del usuario
            user_id: ID del usuario (opcional)
            session_id: ID de sesión (opcional, se genera uno si no se proporciona)
            
        Returns:
            ID del documento guardado en MongoDB
        """
        # Si no se proporciona session_id, generar uno
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generando nuevo session_id: {session_id}")
        else:
            logger.info(f"Usando session_id proporcionado: {session_id}")
        
        # Convertir las respuestas a formato compatible con MongoDB
        responses_data = [resp.dict() for resp in responses]
        logger.info(f"Guardando {len(responses_data)} respuestas en MongoDB")
        
        # Preparar datos para MongoDB
        user_responses_data = []
        for resp in responses:
            response_data = {
                "user_id": user_id,
                "session_id": session_id,
                "question_type": getattr(resp, 'question_type', 'unknown'),
                "question_id": getattr(resp, 'question_id', 0),
                "response": resp.respuesta,
                "response_value": getattr(resp, 'response_value', 0.0),
                "intelligence_type": getattr(resp, 'intelligence_type', None),
                "dimension": getattr(resp, 'dimension', None)
            }
            
            # Guardar cada respuesta individualmente en MongoDB
            response_id = await save_user_response(response_data)
            user_responses_data.append(response_id)
        
        logger.info(f"Respuestas guardadas en MongoDB: {len(user_responses_data)} documentos")
        return session_id
    
    def generate_llm_prompt(self, responses: List[QuestionResponse]) -> str:
        """
        Genera un prompt para el LLM basado en las respuestas del usuario
        
        Args:
            responses: Lista de respuestas del usuario
            
        Returns:
            Prompt formateado para enviar al LLM
        """
        logger.info(f"Generando prompt para {len(responses)} respuestas")
        
        prompt = """Actúa como un psicólogo vocacional.

Te daré una lista de preguntas y respuestas de un usuario.

Para cada una:
1. Clasifícala como MBTI o MI.
2. Si es MBTI, indica la dimensión (E/I, S/N, T/F, J/P) y qué polo refuerza.
3. Si es MI, indica qué inteligencia múltiple evalúa y si la respuesta del usuario sugiere que es alta, media o baja.

Luego, dame:
- El tipo MBTI más probable
- Un ranking de inteligencias múltiples (de más fuerte a más débil)

Ejemplo de respuesta final:

{
  "MBTI": "INFP",
  "MBTI_weights": {
    "E/I": "I fuerte",
    "S/N": "N medio",
    "T/F": "F fuerte",
    "J/P": "P leve"
  },
  "MI": ["Intrapersonal", "Lingüística", "Naturalista", "Interpersonal", ...]
}

---

Preguntas y respuestas del usuario:
"""
        
        # Añadir cada pregunta y respuesta al prompt
        for i, resp in enumerate(responses, 1):
            prompt += f"\n{i}. {resp.pregunta} → {resp.respuesta}"
        
        logger.info(f"Prompt generado con longitud: {len(prompt)} caracteres")
        return prompt
    
    async def save_llm_result(self, session_id: str, llm_result: LLMResultCreate, 
                             prompt_used: str, user_id: Optional[str] = None) -> str:
        """
        Guarda el resultado del LLM en MongoDB
        
        Args:
            session_id: ID de sesión del usuario
            llm_result: Resultado del LLM
            prompt_used: Prompt utilizado para generar el resultado
            user_id: ID del usuario (opcional)
            
        Returns:
            ID del documento guardado en MongoDB
        """
        logger.info(f"Guardando resultado LLM para session_id: {session_id}")
        
        # Preparar datos para MongoDB
        db = await get_database()
        llm_data = {
            "user_id": user_id,
            "session_id": session_id,
            "mbti_result": llm_result.mbti_result,
            "mbti_vector": llm_result.mbti_vector,
            "mbti_weights": llm_result.mbti_weights,
            "mi_ranking": llm_result.mi_ranking,
            "full_result": llm_result.full_analysis if llm_result.full_analysis else {},
            "prompt_used": prompt_used,
            "created_at": datetime.utcnow()
        }
        
        # Guardar en MongoDB
        result = await db.llm_results.insert_one(llm_data)
        logger.info(f"Resultado LLM guardado con ID: {result.inserted_id}")
        
        return str(result.inserted_id)
    
    def process_llm_response(self, llm_response: str) -> LLMResultCreate:
        """
        Procesa la respuesta del LLM y la convierte a un formato estructurado
        
        Args:
            llm_response: Respuesta del LLM en formato de texto
            
        Returns:
            Objeto LLMResultCreate con los datos procesados
        """
        logger.info("Procesando respuesta del LLM")
        logger.info(f"Longitud de la respuesta: {len(llm_response)} caracteres")
        logger.info(f"Primeros 100 caracteres de la respuesta: '{llm_response[:100]}'")
        logger.info(f"Últimos 100 caracteres de la respuesta: '{llm_response[-100:]}'")
        
        # Por ahora, asumimos que el LLM devuelve un JSON bien formateado
        # En un entorno real, se necesitaría un procesamiento más robusto
        try:
            # Extraer la parte JSON de la respuesta
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            logger.info(f"JSON encontrado: inicio={json_start}, fin={json_end}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                logger.info(f"Extrayendo JSON. Longitud: {len(json_str)} caracteres")
                logger.info(f"Primeros 100 caracteres del JSON extraído: '{json_str[:100]}'")
                logger.info(f"Últimos 100 caracteres del JSON extraído: '{json_str[-100:]}'")
                
                result_dict = json.loads(json_str)
                logger.info(f"JSON parseado correctamente. Claves: {', '.join(result_dict.keys())}")
                
                # Extraer los campos necesarios
                mbti_result = result_dict.get("MBTI", "")
                mbti_weights = result_dict.get("MBTI_weights", {})
                mi_ranking = result_dict.get("MI", [])
                
                logger.info(f"MBTI extraído: {mbti_result}")
                logger.info(f"MBTI weights extraídos: {mbti_weights}")
                logger.info(f"MI ranking extraído: {mi_ranking[:3]}...")
                
                # Opcionalmente, intentar extraer el vector MBTI si está presente
                mbti_vector = result_dict.get("MBTI_vector", None)
                logger.info(f"MBTI vector extraído: {mbti_vector}")
                
                # Verificar la estructura de los datos extraídos
                if not mbti_result:
                    logger.warning("MBTI code vacío en la respuesta")
                    
                if not mbti_weights:
                    logger.warning("MBTI weights vacíos en la respuesta")
                    
                if not mi_ranking:
                    logger.warning("MI ranking vacío en la respuesta")
                
                return LLMResultCreate(
                    mbti_result=mbti_result,
                    mbti_vector=mbti_vector,
                    mbti_weights=mbti_weights,
                    mi_ranking=mi_ranking,
                    full_analysis=result_dict
                )
            else:
                logger.error("No se encontró un JSON válido en la respuesta del LLM")
                logger.error(f"Respuesta completa: {llm_response}")
                raise ValueError("No se encontró un JSON válido en la respuesta del LLM")
                
        except Exception as e:
            logger.error(f"Error procesando respuesta del LLM: {str(e)}")
            logger.error(f"Respuesta que causó el error: {llm_response}")
            logger.error(f"Traza de error completa:", exc_info=True)
            
            # En caso de error, crear un resultado genérico
            return LLMResultCreate(
                mbti_result="XXXX",
                mbti_weights={"E/I": "Error", "S/N": "Error", "T/F": "Error", "J/P": "Error"},
                mi_ranking=["Error procesando respuesta del LLM"],
                full_analysis={"error": str(e), "raw_response": llm_response}
            ) 