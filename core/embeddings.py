"""
Embedding processing and storage utilities
"""

import json
from datetime import datetime
from typing import List

from langchain_core.documents import Document

from config.settings import Config
from services.gcp_client import GCPClientService


class EmbeddingProcessor:
    """Handles embedding creation and storage for documents"""
    
    def __init__(self, config: Config, gcp_client: GCPClientService):
        self.config = config
        self.gcp_client = gcp_client
    
    def create_embeddings(self, documents: List[Document]) -> List:
        """Generate embeddings for documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        print(f"Creating embeddings for {len(documents)} documents...")
        embeddings = self.gcp_client.create_embeddings(
            [doc.page_content for doc in documents]
        )
        return embeddings
    
    def prepare_embeddings_for_index(self, documents: List[Document], embeddings: List) -> str:
        """Prepare embeddings in the required format for Vertex AI Vector Search"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"index_data/embeddings_{timestamp}.jsonl"
        
        bucket = self.gcp_client.get_bucket()
        data_lines = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            index_item = self._create_index_item(i, doc, embedding)
            data_lines.append(json.dumps(index_item))
        
        # Upload to Cloud Storage
        blob = bucket.blob(output_path)
        blob.upload_from_string('\n'.join(data_lines))
        
        gcs_uri = f"gs://{self.config.bucket_name}/{output_path}"
        print(f"Uploaded {len(data_lines)} embeddings to {gcs_uri}")
        return gcs_uri
    
    def _create_index_item(self, index: int, doc: Document, embedding) -> dict:
        """Create a single index item with proper structure"""
        # Create metadata filters
        restricts = []
        
        if 'population' in doc.metadata:
            restricts.append({
                "namespace": "population",
                "allow": [doc.metadata['population']]
            })
        
        if 'code_maane' in doc.metadata:
            restricts.append({
                "namespace": "code_maane",
                "allow": [doc.metadata['code_maane']]
            })
        
        return {
            "id": str(index),
            "embedding": embedding.values,
            "restricts": restricts,
            "crowding_tag": doc.metadata.get('population', 'general'),
            "original_data": {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        }