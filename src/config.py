"""Configuration management for RAG system."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    API_TITLE: str = "Enterprise RAG Knowledge Base"
    API_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    INDEX_DIR: Path = DATA_DIR / "indexes"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # Text Processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    
    # Search Settings
    VECTOR_TOP_K: int = 10
    KEYWORD_TOP_K: int = 10
    HYBRID_TOP_K: int = 5
    VECTOR_WEIGHT: float = 0.7
    KEYWORD_WEIGHT: float = 0.3
    
    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "openai" or "groq"
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama-3.1-70b-versatile"  # or "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500
    
    # AWS Bedrock Settings (optional)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create necessary directories
settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)