import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.db.mongodb_models import database_manager, populate_careers_collection

# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Incluir los routers
app.include_router(api_router)

# Eventos de startup y shutdown para MongoDB
@app.on_event("startup")
async def startup_event():
    """Inicializar la conexión a MongoDB al iniciar la aplicación"""
    try:
        await database_manager.connect_to_mongodb()
        await populate_careers_collection()
        print("✅ MongoDB conectado y datos iniciales cargados")
    except Exception as e:
        print(f"❌ Error conectando a MongoDB: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar la conexión a MongoDB al cerrar la aplicación"""
    try:
        await database_manager.close_mongodb_connection()
        print("✅ Conexión a MongoDB cerrada")
    except Exception as e:
        print(f"❌ Error cerrando MongoDB: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG) 