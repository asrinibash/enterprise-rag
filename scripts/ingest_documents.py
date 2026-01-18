"""Script to ingest documents from the data/documents directory."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_processor import TextProcessor
from src.ingestion.embedder import Embedder
from src.search.vector_store import VectorStore
from src.search.keyword_search import KeywordSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Document Ingestion Pipeline")
    logger.info("=" * 80)
    
    # Check if documents directory has files
    doc_files = list(settings.DOCUMENTS_DIR.glob("**/*"))
    doc_files = [f for f in doc_files if f.is_file()]
    
    if not doc_files:
        logger.warning(f"No documents found in {settings.DOCUMENTS_DIR}")
        logger.info("Please add documents (PDF, TXT, DOCX) to the data/documents/ directory")
        return
    
    logger.info(f"Found {len(doc_files)} files in {settings.DOCUMENTS_DIR}")
    
    # Step 1: Load documents
    logger.info("\n[1/5] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(settings.DOCUMENTS_DIR, recursive=True)
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    stats = loader.get_statistics()
    logger.info(f"Loaded: {stats}")
    
    # Step 2: Process documents
    logger.info("\n[2/5] Processing and chunking documents...")
    processor = TextProcessor()
    chunks = processor.process_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Initialize embedder
    logger.info("\n[3/5] Initializing embedding model...")
    embedder = Embedder()
    
    # Step 4: Build vector store
    logger.info("\n[4/5] Building FAISS vector index...")
    vector_store = VectorStore(embedder)
    vector_store.build_index(chunks)
    
    # Step 5: Build keyword search
    logger.info("\n[5/5] Building BM25 keyword index...")
    keyword_search = KeywordSearch()
    keyword_search.build_index(chunks)
    
    # Save indexes
    logger.info("\nSaving indexes to disk...")
    index_path = settings.INDEX_DIR / "faiss_index.bin"
    metadata_path = settings.INDEX_DIR / "metadata.pkl"
    vector_store.save(index_path, metadata_path)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Ingestion Complete!")
    logger.info("=" * 80)
    logger.info(f"Total documents processed: {stats['total_documents']}")
    logger.info(f"Total chunks created: {len(chunks)}")
    logger.info(f"Vector index size: {vector_store.get_stats()['total_vectors']}")
    logger.info(f"Indexes saved to: {settings.INDEX_DIR}")
    logger.info("\nYou can now start the API server with: uv run python -m src.main")


if __name__ == "__main__":
    main()