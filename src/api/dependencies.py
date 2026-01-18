import logging
from pathlib import Path
from functools import lru_cache

from src.config import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_processor import TextProcessor
from src.ingestion.embedder import Embedder
from src.search.vector_store import VectorStore
from src.search.keyword_search import KeywordSearch
from src.search.hybrid_search import HybridSearch
from src.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system with all components."""
    
    def __init__(self):
        # Initialize components
        logger.info("Initializing RAG system...")
        
        self.loader = DocumentLoader()
        self.processor = TextProcessor()
        self.embedder = Embedder()
        
        self.vector_store = VectorStore(self.embedder)
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch(self.vector_store, self.keyword_search)
        
        self.generator = LLMGenerator()
        
        # Try to load existing indexes
        self.load_indexes()
        
        logger.info("RAG system initialized")
    
    def load_indexes(self):
        """Load existing indexes if available."""
        index_path = settings.INDEX_DIR / "faiss_index.bin"
        metadata_path = settings.INDEX_DIR / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                logger.info("Loading existing indexes...")
                self.vector_store.load(index_path, metadata_path)
                
                # Rebuild keyword search from loaded documents
                self.keyword_search.build_index(self.vector_store.documents)
                
                logger.info("Indexes loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load indexes: {e}")
                logger.info("Starting with empty indexes")
        else:
            logger.info("No existing indexes found. Starting fresh.")
    
    def save_indexes(self):
        """Save indexes to disk."""
        index_path = settings.INDEX_DIR / "faiss_index.bin"
        metadata_path = settings.INDEX_DIR / "metadata.pkl"
        
        try:
            self.vector_store.save(index_path, metadata_path)
            logger.info("Indexes saved successfully")
        except Exception as e:
            logger.error(f"Failed to save indexes: {e}")


@lru_cache()
def get_rag_system() -> RAGSystem:
    """
    Get or create RAG system instance (cached).
    This ensures we have a single instance across all requests.
    """
    return RAGSystem()