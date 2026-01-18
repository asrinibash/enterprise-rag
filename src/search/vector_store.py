"""FAISS vector store for similarity search."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from langchain_core.documents import Document

from src.config import settings
from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.dimension = embedder.embedding_dim
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.doc_embeddings: Optional[np.ndarray] = None
    
    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS index from documents."""
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts, show_progress=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and embeddings
        self.documents = documents
        self.doc_embeddings = embeddings
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = settings.VECTOR_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not built")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results (convert L2 distance to similarity score)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity (inverse)
                similarity = 1 / (1 + dist)
                results.append((self.documents[idx], float(similarity)))
        
        return results
    
    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save documents and embeddings
        metadata = {
            "documents": self.documents,
            "embeddings": self.doc_embeddings,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, index_path: Path, metadata_path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load documents and embeddings
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata["documents"]
        self.doc_embeddings = metadata["embeddings"]
        
        logger.info(f"Index loaded from {index_path} ({self.index.ntotal} vectors)")
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
        }