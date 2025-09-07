"""
Configuration settings for the Vertex AI Vector Search RAG System
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration class for the RAG system"""
    
    # GCP Settings
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "gapmaanim")
    location: str = "europe-west4"
    bucket_name: str = "gap-vector-search"
    
    # Vector Search Settings
    index_display_name: str = "gap-rag-txt-index"
    index_endpoint_display_name: str = "gap-rag-txt-endpoint"
    dimensions: int = 3072  # gemini-embedding-001 dimensions
    
    # Index Configuration
    approximate_neighbors_count: int = 100
    distance_measure_type: str = "COSINE_DISTANCE"
    leaf_node_embedding_count: int = 1000
    leaf_nodes_to_search_percent: int = 7
    
    # Deployment Settings
    machine_type: str = "e2-standard-16"
    min_replica_count: int = 1
    max_replica_count: int = 2
    
    # Text Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Search Settings
    default_top_k: int = 5
    max_wait_minutes: int = 10
    
    @classmethod
    def from_env(cls, project_id: Optional[str] = None, 
                 location: Optional[str] = None, 
                 bucket_name: Optional[str] = None) -> 'Config':
        """Create configuration from environment variables with overrides"""
        config = cls()
        
        if project_id:
            config.project_id = project_id
        if location:
            config.location = location
        if bucket_name:
            config.bucket_name = bucket_name
            
        return config