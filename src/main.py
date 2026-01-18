import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.routes import router
from src.api.dependencies import get_rag_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting up RAG application...")
    
    # Initialize RAG system (pre-load models and indexes)
    rag_system = get_rag_system()
    logger.info(f"RAG system ready with {rag_system.vector_store.get_stats()['total_vectors']} vectors")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG application...")
    # Save indexes on shutdown
    try:
        rag_system.save_indexes()
    except Exception as e:
        logger.error(f"Error saving indexes on shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Enterprise RAG Knowledge Base with Hybrid Search (Vector + Keyword)",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise RAG Knowledge Base API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )