"""Hybrid search combining vector and keyword search."""

import logging
from typing import List, Tuple, Dict
from langchain_core.documents import Document

from src.config import settings
from src.search.vector_store import VectorStore
from src.search.keyword_search import KeywordSearch

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        vector_weight: float = settings.VECTOR_WEIGHT,
        keyword_weight: float = settings.KEYWORD_WEIGHT,
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1/(k + rank(d))
        where k is a constant (default 60) and rank is the position in the list.
        """
        # Create document ID mapping (using content hash as ID)
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}
        
        # Process vector results
        for rank, (doc, score) in enumerate(vector_results, start=1):
            doc_id = hash(doc.page_content)
            rrf_score = self.vector_weight / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_objects[doc_id] = doc
        
        # Process keyword results
        for rank, (doc, score) in enumerate(keyword_results, start=1):
            doc_id = hash(doc.page_content)
            rrf_score = self.keyword_weight / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_objects[doc_id] = doc
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create final results
        results = [
            (doc_objects[doc_id], score)
            for doc_id, score in sorted_docs
        ]
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = settings.HYBRID_TOP_K,
        use_rrf: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_rrf: Use Reciprocal Rank Fusion (True) or simple weighted combination
        
        Returns:
            List of (Document, score) tuples
        """
        # Perform both searches
        vector_results = self.vector_store.search(
            query, top_k=settings.VECTOR_TOP_K
        )
        keyword_results = self.keyword_search.search(
            query, top_k=settings.KEYWORD_TOP_K
        )
        
        logger.info(
            f"Vector search: {len(vector_results)} results, "
            f"Keyword search: {len(keyword_results)} results"
        )
        
        # Combine results
        if use_rrf:
            combined_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results
            )
        else:
            # Simple weighted combination
            doc_scores: Dict[str, Tuple[Document, float]] = {}
            
            for doc, score in vector_results:
                doc_id = hash(doc.page_content)
                doc_scores[doc_id] = (doc, score * self.vector_weight)
            
            for doc, score in keyword_results:
                doc_id = hash(doc.page_content)
                if doc_id in doc_scores:
                    existing_doc, existing_score = doc_scores[doc_id]
                    doc_scores[doc_id] = (
                        existing_doc,
                        existing_score + score * self.keyword_weight
                    )
                else:
                    doc_scores[doc_id] = (doc, score * self.keyword_weight)
            
            combined_results = sorted(
                doc_scores.values(),
                key=lambda x: x[1],
                reverse=True
            )
        
        # Return top-k results
        return combined_results[:top_k]