"""Text preprocessing and chunking."""

import logging
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text documents."""
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        
        # Remove multiple dots
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preservation."""
        chunked_docs = []
        
        for doc in documents:
            # Clean the text
            cleaned_text = self.clean_text(doc.page_content)
            
            # Create a temporary document with cleaned text
            temp_doc = Document(
                page_content=cleaned_text,
                metadata=doc.metadata.copy()
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([temp_doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                })
            
            chunked_docs.extend(chunks)
        
        logger.info(
            f"Processed {len(documents)} documents into {len(chunked_docs)} chunks"
        )
        return chunked_docs
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Complete processing pipeline: clean and chunk."""
        return self.chunk_documents(documents)