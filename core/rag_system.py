"""
Main RAG System orchestrating all components
"""

from typing import Dict, List, Optional

from config.settings import Config
from services.gcp_client import GCPClientService
from core.document_loader import DocumentLoader
from core.embeddings import EmbeddingProcessor
from core.vector_index import VectorIndexManager
from core.search_engine import SearchEngine
from utils.helpers import safe_execute, validate_file_path, print_search_results


class VertexAIVectorSearchRAG:
    """A comprehensive RAG system using Vertex AI Vector Search"""
    
    def __init__(self, project_id: Optional[str] = None, 
                 location: Optional[str] = None, 
                 bucket_name: Optional[str] = None):
        """Initialize the RAG system"""
        # Load configuration
        self.config = Config.from_env(project_id, location, bucket_name)
        
        # Initialize services
        self.gcp_client = GCPClientService(self.config)
        self.document_loader = DocumentLoader(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config, self.gcp_client)
        self.index_manager = VectorIndexManager(self.config)
        self.search_engine = SearchEngine(self.config, self.gcp_client, self.index_manager)
    
    def setup_from_file(self, file_path: str) -> bool:
        """Complete setup process from a document file"""
        print("Starting setup process...")
        
        # Validate file
        if not validate_file_path(file_path):
            return False
        
        # Check for existing resources first
        if self.index_manager.find_existing_resources():
            print("✓ System is already set up and ready!")
            return True
        
        return safe_execute(
            self._perform_setup,
            file_path,
            error_message="Setup failed"
        ) or False
    
    def _perform_setup(self, file_path: str) -> bool:
        """Perform the actual setup process"""
        # Load documents
        documents = self.document_loader.load_documents(file_path)
        if not documents:
            print("No documents loaded. Aborting setup.")
            return False
        
        # Create embeddings
        embeddings = self.embedding_processor.create_embeddings(documents)
        
        # Prepare for indexing
        embeddings_uri = self.embedding_processor.prepare_embeddings_for_index(
            documents, embeddings
        )
        
        # Create index if needed
        if not self.index_manager.index:
            self.index_manager.create_index(embeddings_uri)
        
        # Create endpoint if needed
        if not self.index_manager.index_endpoint:
            self.index_manager.create_index_endpoint()
        
        # Deploy index
        if not self.index_manager.deployed_index_id:
            self.index_manager.deploy_index()
        
        print("✓ Setup completed successfully!")
        return True
    
    def search(self, query: str, top_k: int = 5, 
               population: Optional[str] = None, 
               code_maane: Optional[str] = None) -> List[Dict]:
        """Search for similar documents"""
        return self.search_engine.search(query, top_k, population, code_maane)
    
    def test_search(self, test_queries: Optional[List[str]] = None) -> bool:
        """Test the search functionality with predefined queries"""
        if not test_queries:
            test_queries = [
                "פיזיקה קוונטית",
                "מהם עקרונות היסוד של פיזיקה קוונטית?"
            ]
        
        print("\n" + "="*50)
        print("Testing search functionality...")
        
        for query in test_queries:
            results = self.search(query, top_k=3)
            print_search_results(results, query)
        
        return len([q for q in test_queries if self.search(q)]) > 0
    
    def check_system_status(self) -> bool:
        """Check if the system is ready for search operations"""
        print("Checking system status...")
        
        # Try to find existing resources first
        if not self.index_manager.find_existing_resources():
            print("✗ No existing resources found")
            return False
        
        if not self.index_manager.check_deployment_status():
            print("✗ Deployment not ready")
            return False
        
        print("✓ System ready for search")
        return True
    
    def get_embedding_dimensions(self) -> int:
        """Get the actual embedding dimensions from the model"""
        return self.gcp_client.get_embedding_dimensions()
    
    def check_endpoint_config(self):
        """Check endpoint configuration for debugging"""
        if not self.index_manager.index_endpoint:
            print("No endpoint found")
            return
        
        print("Endpoint configuration:")
        print(f"Resource name: {self.index_manager.index_endpoint.resource_name}")
        print(f"Display name: {self.index_manager.index_endpoint.display_name}")
        
        try:
            endpoint_info = self.index_manager.index_endpoint.to_dict()
            print(f"Endpoint details: {endpoint_info}")           
        except Exception as e:
            print(f"Error checking endpoint config: {e}")