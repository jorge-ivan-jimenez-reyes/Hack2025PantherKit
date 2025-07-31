from typing import Dict, List, Tuple, Any
import numpy as np
from app.models.advanced_neural_model import AdvancedNeuralCareerModel
from app.models.career_model import CareerRecommender
from app.utils.self_regulated_processor import SelfRegulatedProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
import random

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_service")

class AdvancedCareerService:
    """
    Servicio simplificado que usa únicamente el modelo neural de última generación
    con datos de Self-regulated learning para recomendación de carreras.
    """
    
    def __init__(self):
        self.neural_model = AdvancedNeuralCareerModel()
        self.career_recommender = CareerRecommender()
        self.srl_processor = SelfRegulatedProcessor()
        logger.info("AdvancedCareerService inicializado con modelo Transformer")
        
    def generate_training_data_from_srl(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Genera datos de entrenamiento usando el procesador de Self-regulated learning.
        
        Args:
            num_samples: Número de muestras a generar
            
        Returns:
            Tupla con (X_train, y_train, career_names)
        """
        logger.info(f"Generando datos de entrenamiento SRL con {num_samples} muestras...")
        
        # Obtener datos procesados del SRL processor
        training_data, career_names = self.srl_processor.prepare_training_data(sample_size=num_samples)
        
        # Convertir a arrays para el modelo
        X = []
        y = []
        
        for sample in training_data:
            # Crear vector de características (4 MBTI + 4 pesos + 8 MI = 16 dimensiones)
            features = (
                sample["mbti_vector"] + 
                [sample["mbti_weights"][dim] for dim in ["E/I", "S/N", "T/F", "J/P"]] +
                [sample["mi_scores"][mi_type] for mi_type in ["Lin", "LogMath", "Spa", "BodKin", "Mus", "Inter", "Intra", "Nat"]]
            )
            X.append(features)
            y.append(sample["career_label"])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        logger.info(f"Datos generados: {X.shape}, Carreras únicas: {len(career_names)}")
        logger.info(f"Carreras disponibles: {career_names}")
        
        return X, y, career_names
    
    def train_model(self, num_samples: int = 10000, epochs: int = 100, batch_size: int = 64, validation: bool = True):
        """
        Entrena el modelo neural avanzado con datos de Self-regulated learning.
        
        Args:
            num_samples: Número de muestras para entrenamiento
            epochs: Número máximo de epochs
            batch_size: Tamaño del batch
            validation: Si realizar validación durante entrenamiento
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        try:
            logger.info(f"Comenzando entrenamiento del modelo avanzado con {num_samples} muestras")
            
            # Generar datos de entrenamiento
            X, y, career_names = self.generate_training_data_from_srl(num_samples)
            
            if validation:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                logger.info(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
                
                # Entrenar modelo
                training_results = self.neural_model.train_model(
                    X_train, y_train, 
                    validation_split=0.2, 
                    epochs=epochs, 
                    batch_size=batch_size
                )
                
                # Evaluar en conjunto de prueba
                test_results = self.neural_model.evaluate_model(X_test, y_test)
                
                logger.info(f"Resultados de prueba: {test_results}")
                
                # Combinar resultados
                results = {
                    **training_results,
                    **test_results,
                    "career_names": career_names,
                    "num_samples": num_samples
                }
                
            else:
                # Entrenar sin validación
                results = self.neural_model.train_model(
                    X, y, 
                    validation_split=0.0, 
                    epochs=epochs, 
                    batch_size=batch_size
                )
                results.update({
                    "career_names": career_names,
                    "num_samples": num_samples
                })
            
            logger.info("Entrenamiento completado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}", exc_info=True)
            return {"error": f"Error durante el entrenamiento: {str(e)}"}
    
    def predict_careers(self, mbti_code: str, mbti_vector: List[int], 
                       mbti_weights: Dict[str, float], mi_scores: Dict[str, float], 
                       top_n: int = 5) -> List[Dict]:
        """
        Predice las carreras más adecuadas usando el modelo neural avanzado.
        
        Args:
            mbti_code: Código MBTI del usuario
            mbti_vector: Vector binario MBTI
            mbti_weights: Pesos de dimensiones MBTI
            mi_scores: Puntuaciones de inteligencias múltiples
            top_n: Número de recomendaciones a retornar
            
        Returns:
            Lista de recomendaciones de carreras
        """
        logger.info(f"Iniciando predicción para perfil MBTI: {mbti_code}")
        
        try:
            # Verificar que el modelo esté entrenado
            if self.neural_model.model is None:
                logger.info("No hay modelo entrenado. Entrenando nuevo modelo...")
                self.train_model(num_samples=8000, epochs=50, batch_size=64)
            
            # Obtener carreras disponibles del modelo
            if hasattr(self.neural_model.label_encoder, 'classes_'):
                career_names = list(self.neural_model.label_encoder.classes_)
            else:
                # Fallback: usar datos de entrenamiento para obtener nombres
                _, _, career_names = self.generate_training_data_from_srl(1000)
            
            logger.info(f"Prediciendo entre {len(career_names)} carreras disponibles")
            
            # Realizar predicción
            predictions = self.neural_model.predict_career(
                mbti_vector, mbti_weights, mi_scores, career_names
            )
            
            # Filtrar top_n predicciones
            top_predictions = predictions[:top_n]
            logger.info(f"Top {top_n} predicciones: {[f'{name}: {score:.4f}' for name, score in top_predictions]}")
            
            # Enriquecer con información de carreras
            results = []
            for career_name, score in top_predictions:
                # Buscar información de la carrera en la base de datos
                career_info = next(
                    (career for career in self.career_recommender.careers 
                     if career_name.lower() in career["nombre"].lower() or 
                        career["nombre"].lower() in career_name.lower()),
                    None
                )
                
                if career_info:
                    results.append({
                        "nombre": career_info["nombre"],
                        "universidad": career_info["universidad"],
                        "ciudad": career_info["ubicacion"],
                        "match_score": float(score),
                        "descripcion": career_info.get("descripcion", "")
                    })
                else:
                    # Si no se encuentra en la BD, usar nombre predicho
                    results.append({
                        "nombre": career_name,
                        "universidad": "Universidad por determinar",
                        "ciudad": "Ubicación por determinar",
                        "match_score": float(score),
                        "descripcion": "Descripción no disponible"
                    })
            
            logger.info(f"Recomendaciones finales generadas: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}", exc_info=True)
            
            # Fallback a recomendaciones basadas en reglas
            logger.info("Usando sistema de fallback basado en reglas...")
            return self._rule_based_fallback(mbti_code, mi_scores, top_n)
    
    def _rule_based_fallback(self, mbti_code: str, mi_scores: Dict[str, float], top_n: int) -> List[Dict]:
        """
        Sistema de fallback basado en reglas cuando el modelo neural falla.
        """
        # Mapeos MBTI a carreras STEM
        mbti_to_careers = {
            "INTJ": ["Ciencia de Datos", "Ingeniería en Sistemas Computacionales", "Nanotecnología"],
            "INTP": ["Física de Materiales", "Ciencia de Datos", "Biotecnología"],
            "ENTJ": ["Ingeniería Industrial", "Ingeniería Mecatrónica", "Administración de TI"],
            "ENTP": ["Ingeniería en Robótica", "Diseño UX", "Innovación Tecnológica"],
            "ISTJ": ["Ingeniería Civil", "Ingeniería Química", "Sistemas Computacionales"],
            "ISFJ": ["Ingeniería Biomédica", "Ingeniería Ambiental", "Biotecnología"],
            "ESTJ": ["Ingeniería Industrial", "Administración de Proyectos", "Ingeniería Mecánica"],
            "ESFJ": ["Ingeniería Biomédica", "Diseño UX", "Ingeniería Ambiental"]
        }
        
        # Obtener carreras sugeridas para el tipo MBTI
        suggested_careers = mbti_to_careers.get(mbti_code, ["Ciencia de Datos", "Ingeniería Mecatrónica", "Biotecnología"])
        
        results = []
        for i, career_name in enumerate(suggested_careers[:top_n]):
            # Buscar en base de datos
            career_info = next(
                (career for career in self.career_recommender.careers 
                 if career_name.lower() in career["nombre"].lower()),
                None
            )
            
            if career_info:
                # Calcular score basado en MI y posición
                base_score = 0.8 - (i * 0.1)  # Decrece con la posición
                mi_bonus = sum(mi_scores.values()) / len(mi_scores) * 0.2
                final_score = min(base_score + mi_bonus, 1.0)
                
                results.append({
                    "nombre": career_info["nombre"],
                    "universidad": career_info["universidad"],
                    "ciudad": career_info["ubicacion"],
                    "match_score": float(final_score),
                    "descripcion": career_info.get("descripcion", "")
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo actual.
        """
        return {
            "model_type": "Advanced Transformer Neural Network",
            "architecture": "Multi-Head Attention + Feed Forward",
            "embedding_dim": self.neural_model.embed_dim,
            "num_attention_heads": self.neural_model.num_heads,
            "num_transformer_blocks": self.neural_model.num_transformer_blocks,
            "data_source": "Self-regulated Learning + Academic Achievement + Socioeconomic",
            "model_summary": self.neural_model.get_model_summary(),
            "is_trained": self.neural_model.model is not None
        }
    
    def retrain_with_new_data(self, num_samples: int = 15000) -> Dict[str, Any]:
        """
        Re-entrena el modelo con nuevos datos generados.
        """
        logger.info(f"Re-entrenando modelo con {num_samples} nuevas muestras...")
        return self.train_model(num_samples=num_samples, epochs=80, batch_size=64, validation=True)

if __name__ == "__main__":
    # Ejemplo de uso
    service = AdvancedCareerService()
    
    # Entrenar modelo
    results = service.train_model(num_samples=5000, epochs=10)
    print(f"Resultados de entrenamiento: {results}")
    
    # Hacer predicción de ejemplo
    example_prediction = service.predict_careers(
        mbti_code="INTP",
        mbti_vector=[1, 1, 0, 1],
        mbti_weights={"E/I": 0.8, "S/N": 0.7, "T/F": 0.9, "J/P": 0.6},
        mi_scores={"Lin": 0.6, "LogMath": 0.9, "Spa": 0.7, "BodKin": 0.4, 
                  "Mus": 0.3, "Inter": 0.5, "Intra": 0.8, "Nat": 0.6},
        top_n=3
    )
    print(f"Predicción de ejemplo: {example_prediction}")
    
    # Información del modelo
    model_info = service.get_model_info()
    print(f"Información del modelo: {model_info}") 