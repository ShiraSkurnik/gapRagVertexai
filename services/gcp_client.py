"""
Google Cloud Platform client services
"""

from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1.services.match_service import MatchServiceClient
from vertexai.language_models import TextEmbeddingModel

from config.settings import Config


class GCPClientService:
    """Service for managing GCP clients"""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all GCP clients"""
        # Initialize AI Platform
        aiplatform.init(project=self.config.project_id, location=self.config.location)
        
        # Initialize clients
        self.embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        self.storage_client = storage.Client()
        self.match_client = MatchServiceClient()
    
    def get_bucket(self):
        """Get the configured GCS bucket"""
        return self.storage_client.bucket(self.config.bucket_name)
    
    def create_embeddings(self, texts: list) -> list:
        """Create embeddings for a list of texts"""
        return self.embedding_model.get_embeddings(
            texts, 
            output_dimensionality=self.config.dimensions
        )
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensions of embeddings from the model"""
        test_embedding = self.embedding_model.get_embeddings(["test"])
        return len(test_embedding[0].values)