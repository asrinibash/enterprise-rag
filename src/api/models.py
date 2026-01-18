from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to retrieve", ge=1, le=20)
    use_citations: Optional[bool] = Field(True, description="Include citations in response")


class Source(BaseModel):
    """Source document information."""
    content: str = Field(..., description="Document content snippet")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    answer: str
    sources: List[Source]
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    message: str
    files_processed: int
    chunks_created: int
    index_updated: bool


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    file_name: str
    file_type: str
    source: str
    loaded_at: str
    chunks: int


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    total_documents: int
    documents: List[DocumentInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    llm_available: bool
    index_stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """System statistics response."""
    vector_store: Dict[str, Any]
    keyword_search: Dict[str, Any]
    total_chunks: int
    total_documents: int