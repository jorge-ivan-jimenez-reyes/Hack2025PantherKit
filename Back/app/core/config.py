import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Settings:
    # Información de la aplicación
    API_TITLE: str = "STEM Career Recommendation API"
    API_DESCRIPTION: str = "API for recommending STEM careers based on personality and multiple intelligences"
    API_VERSION: str = "1.0.0"
    
    # Configuración de MongoDB
    MONGODB_URL: str = os.getenv(
        "MONGODB_URL", 
        "mongodb://admin:admin123@mongodb:27017/career_recommendations?authSource=admin"
    )
    
    # Configuración general
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ('true', '1', 't')
    
    # TensorFlow
    TF_CPP_MIN_LOG_LEVEL: str = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # CORS
    CORS_ORIGINS: list = ["*"]  # Permitir cualquier origen en desarrollo
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]  # Permitir cualquier método
    CORS_HEADERS: list = ["*"]  # Permitir cualquier header

settings = Settings() 