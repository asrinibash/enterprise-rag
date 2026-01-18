"""Document loading from various file formats."""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document

# Lazy import to avoid slow transformers loading
def _get_pdf_loader():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader

def _get_text_loader():
    from langchain_community.document_loaders import TextLoader
    return TextLoader

def _get_docx_loader():
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    return UnstructuredWordDocumentLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from multiple file formats."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": "_get_pdf_loader",
        ".txt": "_get_text_loader",
        ".md": "_get_text_loader",
        ".docx": "_get_docx_loader",
    }
    
    def __init__(self):
        self.loaded_documents: List[Document] = []
    
    def load_file(self, file_path: Path) -> List[Document]:
        """Load a single file and return documents."""
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        try:
            loader_func = globals()[self.SUPPORTED_EXTENSIONS[extension]]
            loader_class = loader_func()
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": extension,
                    "loaded_at": datetime.now().isoformat(),
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory: Path, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory."""
        all_documents = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping {file_path.name}: {str(e)}")
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory}")
        self.loaded_documents = all_documents
        return all_documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        if not self.loaded_documents:
            return {"total_documents": 0}
        
        file_types = {}
        total_chars = 0
        
        for doc in self.loaded_documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.page_content)
        
        return {
            "total_documents": len(self.loaded_documents),
            "file_types": file_types,
            "total_characters": total_chars,
            "avg_chars_per_doc": total_chars // len(self.loaded_documents),
        }