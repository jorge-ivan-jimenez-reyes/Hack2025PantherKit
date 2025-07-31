from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from bson import ObjectId
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import os

class PyObjectId(ObjectId):
    """Wrapper para ObjectId de MongoDB que es compatible con Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserResponse(BaseModel):
    """Modelo para respuestas de usuario almacenadas en MongoDB"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[str] = None
    session_id: str
    question_type: str  # "mbti", "mi", "srl"
    question_id: int
    response: str
    response_value: float
    intelligence_type: Optional[str] = None
    dimension: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserProfile(BaseModel):
    """Modelo para perfiles de usuario completos"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[str] = None
    session_id: str
    
    # Resultados MBTI
    mbti_code: str
    mbti_vector: List[int]
    mbti_weights: Dict[str, float]
    
    # Resultados MI
    mi_scores: Dict[str, float]
    
    # Resultados SRL (si disponibles)
    srl_scores: Optional[Dict[str, float]] = None
    academic_scores: Optional[Dict[str, float]] = None
    socioeconomic_data: Optional[Dict[str, Any]] = None
    
    # Metadatos
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class CareerRecommendation(BaseModel):
    """Modelo para recomendaciones de carreras"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_profile_id: PyObjectId
    session_id: str
    
    # Recomendaciones
    recommendations: List[Dict[str, Any]]  # Lista de carreras recomendadas
    model_used: str  # "advanced_transformer", "neural_cnn", "rule_based"
    model_version: str
    
    # Métricas del modelo
    prediction_confidence: Optional[float] = None
    top_scores: Optional[List[float]] = None
    
    # Metadatos
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ModelMetrics(BaseModel):
    """Modelo para métricas de entrenamiento de modelos"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    model_type: str  # "advanced_transformer", "neural_cnn"
    model_version: str
    
    # Métricas de entrenamiento
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: Optional[float] = None
    training_loss: float
    validation_loss: float
    test_loss: Optional[float] = None
    
    # Configuración del modelo
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Datos de entrenamiento
    num_training_samples: int
    num_classes: int
    epochs_trained: int
    
    # Metadatos
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    data_source: str = "self_regulated_learning"
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Career(BaseModel):
    """Modelo para información de carreras"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    nombre: str
    universidad: str
    ciudad: str
    descripcion: str
    
    # Campos adicionales para análisis
    area_stem: Optional[str] = None
    nivel_matematicas: Optional[int] = None  # 1-10
    nivel_ciencias: Optional[int] = None     # 1-10
    nivel_tecnologia: Optional[int] = None   # 1-10
    
    # Metadatos
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class DatabaseManager:
    """Gestor de conexión a MongoDB"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        
    async def connect_to_mongodb(self):
        """Conecta a la base de datos MongoDB"""
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:admin123@localhost:27017/career_recommendations?authSource=admin")
        self.client = AsyncIOMotorClient(mongodb_url)
        self.database = self.client.career_recommendations
        
        # Crear índices
        await self.create_indexes()
        
    async def close_mongodb_connection(self):
        """Cierra la conexión a MongoDB"""
        if self.client:
            self.client.close()
            
    async def create_indexes(self):
        """Crea índices necesarios para optimizar consultas"""
        if not self.database:
            return
            
        # Índices para user_responses
        await self.database.user_responses.create_index([
            ("session_id", pymongo.ASCENDING),
            ("question_type", pymongo.ASCENDING)
        ])
        await self.database.user_responses.create_index("timestamp")
        
        # Índices para user_profiles
        await self.database.user_profiles.create_index("session_id", unique=True)
        await self.database.user_profiles.create_index("user_id")
        await self.database.user_profiles.create_index("created_at")
        
        # Índices para career_recommendations
        await self.database.career_recommendations.create_index("user_profile_id")
        await self.database.career_recommendations.create_index("session_id")
        await self.database.career_recommendations.create_index("model_used")
        await self.database.career_recommendations.create_index("created_at")
        
        # Índices para model_metrics
        await self.database.model_metrics.create_index([
            ("model_type", pymongo.ASCENDING),
            ("model_version", pymongo.ASCENDING)
        ])
        await self.database.model_metrics.create_index("trained_at")
        
        # Índices para careers
        await self.database.careers.create_index("nombre")
        await self.database.careers.create_index("universidad")
        await self.database.careers.create_index("ciudad")

# Instancia global del gestor de base de datos
database_manager = DatabaseManager()

async def get_database() -> AsyncIOMotorDatabase:
    """Dependency para obtener la instancia de la base de datos"""
    return database_manager.database

# Funciones de utilidad para trabajar con MongoDB

async def save_user_response(response_data: Dict[str, Any]) -> str:
    """Guarda una respuesta de usuario en la base de datos"""
    db = await get_database()
    user_response = UserResponse(**response_data)
    result = await db.user_responses.insert_one(user_response.dict(by_alias=True, exclude_unset=True))
    return str(result.inserted_id)

async def save_user_profile(profile_data: Dict[str, Any]) -> str:
    """Guarda un perfil de usuario en la base de datos"""
    db = await get_database()
    user_profile = UserProfile(**profile_data)
    result = await db.user_profiles.insert_one(user_profile.dict(by_alias=True, exclude_unset=True))
    return str(result.inserted_id)

async def save_career_recommendation(recommendation_data: Dict[str, Any]) -> str:
    """Guarda una recomendación de carrera en la base de datos"""
    db = await get_database()
    recommendation = CareerRecommendation(**recommendation_data)
    result = await db.career_recommendations.insert_one(recommendation.dict(by_alias=True, exclude_unset=True))
    return str(result.inserted_id)

async def save_model_metrics(metrics_data: Dict[str, Any]) -> str:
    """Guarda métricas de modelo en la base de datos"""
    db = await get_database()
    metrics = ModelMetrics(**metrics_data)
    result = await db.model_metrics.insert_one(metrics.dict(by_alias=True, exclude_unset=True))
    return str(result.inserted_id)

async def get_user_profile_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Obtiene un perfil de usuario por session_id"""
    db = await get_database()
    profile = await db.user_profiles.find_one({"session_id": session_id})
    return profile

async def get_career_recommendations_by_session(session_id: str) -> List[Dict[str, Any]]:
    """Obtiene recomendaciones de carrera por session_id"""
    db = await get_database()
    recommendations = await db.career_recommendations.find({"session_id": session_id}).to_list(length=None)
    return recommendations

async def get_latest_model_metrics(model_type: str) -> Optional[Dict[str, Any]]:
    """Obtiene las métricas más recientes de un tipo de modelo"""
    db = await get_database()
    metrics = await db.model_metrics.find({"model_type": model_type}).sort("trained_at", -1).limit(1).to_list(length=1)
    return metrics[0] if metrics else None

async def populate_careers_collection():
    """Puebla la colección de carreras con datos iniciales"""
    db = await get_database()
    
    # Verificar si ya existen carreras
    career_count = await db.careers.count_documents({})
    if career_count > 0:
        return  # Ya hay carreras en la base de datos
    
    # Carreras de ejemplo basadas en el archivo careers.json
    sample_careers = [
        {
            "nombre": "Ingeniería en Biotecnología",
            "universidad": "Tec de Monterrey",
            "ciudad": "Querétaro",
            "descripcion": "Carrera que combina biología y tecnología para el desarrollo de soluciones en salud, alimentos y medio ambiente.",
            "area_stem": "Biotecnología",
            "nivel_matematicas": 8,
            "nivel_ciencias": 9,
            "nivel_tecnologia": 7
        },
        {
            "nombre": "Ciencia de Datos",
            "universidad": "UNAM",
            "ciudad": "Ciudad de México", 
            "descripcion": "Carrera enfocada en el análisis de grandes volúmenes de datos, aprendizaje automático y estadística aplicada.",
            "area_stem": "Ciencias de la Computación",
            "nivel_matematicas": 9,
            "nivel_ciencias": 7,
            "nivel_tecnologia": 9
        },
        {
            "nombre": "Ingeniería Mecatrónica",
            "universidad": "IPN",
            "ciudad": "Ciudad de México",
            "descripcion": "Combina mecánica, electrónica, control y programación para crear sistemas robotizados y automatizados.",
            "area_stem": "Ingeniería",
            "nivel_matematicas": 8,
            "nivel_ciencias": 8,
            "nivel_tecnologia": 9
        }
    ]
    
    # Insertar carreras
    careers = [Career(**career_data) for career_data in sample_careers]
    await db.careers.insert_many([career.dict(by_alias=True, exclude_unset=True) for career in careers])

if __name__ == "__main__":
    import asyncio
    
    async def test_connection():
        await database_manager.connect_to_mongodb()
        await populate_careers_collection()
        print("Conexión a MongoDB establecida y datos iniciales cargados")
        await database_manager.close_mongodb_connection()
    
    asyncio.run(test_connection()) 