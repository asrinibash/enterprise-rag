import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.api.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    HealthResponse,
    StatsResponse,
    Source,
    DocumentInfo,
)
from src.api.dependencies import get_rag_system

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_system=Depends(get_rag_system)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        llm_available=rag_system.generator.is_llm_available(),
        index_stats=rag_system.vector_store.get_stats(),
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_system=Depends(get_rag_system)
):
    """Query the RAG system."""
    try:
        start_time = time.time()
        
        # Perform hybrid search
        retrieval_start = time.time()
        results = rag_system.hybrid_search.search(
            request.query,
            top_k=request.top_k
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for the query"
            )
        
        # Extract documents
        documents = [doc for doc, score in results]
        
        # Generate response
        generation_start = time.time()
        response = rag_system.generator.generate(
            request.query,
            documents,
            use_citations=request.use_citations
        )
        generation_time = (time.time() - generation_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            answer=response["answer"],
            sources=[Source(**src) for src in response["sources"]],
            model_used=response["model_used"],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=DocumentUploadResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    rag_system=Depends(get_rag_system)
):
    """Upload and ingest documents."""
    try:
        from src.config import settings
        
        uploaded_files = []
        
        # Save uploaded files
        for file in files:
            file_path = settings.DOCUMENTS_DIR / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_files.append(file_path)
        
        # Load documents
        all_docs = []
        for file_path in uploaded_files:
            docs = rag_system.loader.load_file(file_path)
            all_docs.extend(docs)
        
        # Process documents
        chunks = rag_system.processor.process_documents(all_docs)
        
        # Rebuild indexes
        rag_system.vector_store.build_index(chunks)
        rag_system.keyword_search.build_index(chunks)
        
        # Save indexes
        rag_system.save_indexes()
        
        return DocumentUploadResponse(
            message="Documents ingested successfully",
            files_processed=len(files),
            chunks_created=len(chunks),
            index_updated=True,
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(rag_system=Depends(get_rag_system)):
    """List all indexed documents."""
    try:
        # Get unique documents from vector store
        docs_info = {}
        
        for doc in rag_system.vector_store.documents:
            source = doc.metadata.get("source", "Unknown")
            
            if source not in docs_info:
                docs_info[source] = DocumentInfo(
                    file_name=doc.metadata.get("file_name", "Unknown"),
                    file_type=doc.metadata.get("file_type", "Unknown"),
                    source=source,
                    loaded_at=doc.metadata.get("loaded_at", "Unknown"),
                    chunks=0,
                )
            
            docs_info[source].chunks += 1
        
        return DocumentListResponse(
            total_documents=len(docs_info),
            documents=list(docs_info.values()),
        )
        
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_statistics(rag_system=Depends(get_rag_system)):
    """Get system statistics."""
    try:
        return StatsResponse(
            vector_store=rag_system.vector_store.get_stats(),
            keyword_search=rag_system.keyword_search.get_stats(),
            total_chunks=len(rag_system.vector_store.documents),
            total_documents=len(set(
                doc.metadata.get("source") 
                for doc in rag_system.vector_store.documents
            )),
        )
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def clear_documents(rag_system=Depends(get_rag_system)):
    """Clear all documents and rebuild empty indexes."""
    try:
        from src.config import settings
        
        # Clear indexes
        rag_system.vector_store.documents = []
        rag_system.keyword_search.documents = []
        
        # Rebuild empty indexes
        rag_system.vector_store.build_index([])
        rag_system.keyword_search.build_index([])
        
        return {"message": "All documents cleared"}
        
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))