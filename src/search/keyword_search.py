"""BM25 keyword-based search."""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


class KeywordSearch:
    """BM25-based keyword search engine."""
    
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced with nltk/spacy)."""
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation
        tokens = [token.strip('.,!?;:()[]{}"\'-') for token in tokens]
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def build_index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents."""
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.documents = documents
        
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 index built with {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = settings.KEYWORD_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """Search for documents using BM25."""
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Only include documents with non-zero scores
                results.append((self.documents[idx], score))
        
        return results
    
    def get_stats(self) -> dict:
        """Get keyword search statistics."""
        return {
            "total_documents": len(self.documents),
            "avg_tokens_per_doc": (
                sum(len(tokens) for tokens in self.tokenized_corpus) 
                // len(self.tokenized_corpus)
                if self.tokenized_corpus else 0
            ),
        }