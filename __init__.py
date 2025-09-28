# vertex_ai_rag/__init__.py
"""
Vertex AI Vector Search RAG System
A modular implementation for document indexing and retrieval with metadata filtering
"""

from .core.rag_system import VertexAIVectorSearchRAG
from .config.settings import Config

__version__ = "1.0.0"
__all__ = ["VertexAIVectorSearchRAG", "Config"]

# config/__init__.py
from .settings import Config

__all__ = ["Config"]

# core/__init__.py
from .rag_system import VertexAIVectorSearchRAG
from .document_loader import DocumentLoader
from .embeddings import EmbeddingProcessor
from .vector_index import VectorIndexManager
from .search_engine import SearchEngine

__all__ = [
    "VertexAIVectorSearchRAG",
    "DocumentLoader", 
    "EmbeddingProcessor",
    "VectorIndexManager",
    "SearchEngine"
]

# services/__init__.py
from .gcp_client import GCPClientService

__all__ = ["GCPClientService"]

# utils/__init__.py
from .helpers import print_search_results, safe_execute, validate_file_path, check_deployment_readiness

__all__ = ["print_search_results", "safe_execute", "validate_file_path", "check_deployment_readiness"]