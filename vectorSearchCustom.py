"""
Vertex AI Vector Search RAG System
A clean implementation for document indexing and retrieval with metadata filtering
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vertexai.language_models import TextEmbeddingModel


class VertexAIVectorSearchRAG:
    """A comprehensive RAG system using Vertex AI Vector Search"""
    
    def __init__(self, project_id: str = None, location: str = "europe-west4", bucket_name: str = "gap-vector-search"):
        """Initialize the RAG system"""
        load_dotenv()
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "gapmaanim")
        self.location = location
        self.bucket_name = bucket_name
        
        # Constants
        self.INDEX_DISPLAY_NAME = "gap-rag-txt-index"
        self.INDEX_ENDPOINT_DISPLAY_NAME = "gap-rag-txt-endpoint"
        self.DIMENSIONS = 3072  # gemini-embedding-001 dimensions
        
        # Initialize GCP services
        aiplatform.init(project=self.project_id, location=self.location)
        self.embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        self.storage_client = storage.Client()
        
        # Vector search components
        self.index = None
        self.index_endpoint = None
        self.deployed_index_id = None
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load and process documents from JSON or TXT files"""
        documents = []
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.json':
                documents = self._load_json_documents(file_path)
            elif file_ext == '.txt':
                documents = self._load_text_documents(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            print(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    def _load_json_documents(self, file_path: str) -> List[Document]:
        """Load documents from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data):
            # Assign population based on index
            if i < 1:
                population = "מוסד"
            elif i < 20:
                population = "רשות"
            else:
                population = "מחז"
            
            content = json.dumps(item, indent=2, ensure_ascii=False)
            metadata = {
                "population": population,
                "type": "json",
                "code_maane": str(item.get("קוד_מענה", "")),
                "index": i
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _load_text_documents(self, file_path: str) -> List[Document]:
        """Load documents from text file with chunking"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Assign population based on chunk index
            if i < 2:
                population = "מוסד"
            elif i < 4:
                population = "רשות"
            else:
                population = "מחז"
            
            metadata = {
                "type": "txt",
                "chunk_index": i,
                "source": Path(file_path).name,
                "population": population
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def create_embeddings(self, documents: List[Document]) -> List:
        """Generate embeddings for documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        print(f"Creating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.get_embeddings(
            [doc.page_content for doc in documents],
            output_dimensionality=self.DIMENSIONS
        )
        return embeddings
    
    def prepare_embeddings_for_index(self, documents: List[Document], embeddings: List) -> str:
        """Prepare embeddings in the required format for Vertex AI Vector Search"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"index_data/embeddings_{timestamp}.jsonl"
        
        bucket = self.storage_client.bucket(self.bucket_name)
        data_lines = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
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
            
            index_item = {
                "id": str(i),
                "embedding": embedding.values,
                "restricts": restricts,
                "crowding_tag": doc.metadata.get('population', 'general'),
                "original_data": {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            
            data_lines.append(json.dumps(index_item))
        
        # Upload to Cloud Storage
        blob = bucket.blob(output_path)
        blob.upload_from_string('\n'.join(data_lines))
        
        gcs_uri = f"gs://{self.bucket_name}/{output_path}"
        print(f"Uploaded {len(data_lines)} embeddings to {gcs_uri}")
        return gcs_uri
    
    def create_index(self, embeddings_gcs_uri: str) -> MatchingEngineIndex:
        """Create a new Vector Search index"""
        print("Creating new Vector Search index...")
        
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.INDEX_DISPLAY_NAME,
            contents_delta_uri=embeddings_gcs_uri,
            dimensions=self.DIMENSIONS,
            approximate_neighbors_count=100,
            distance_measure_type="COSINE_DISTANCE",
            leaf_node_embedding_count=1000,
            leaf_nodes_to_search_percent=7,
            description="RAG index for documents with metadata filtering",
            labels={
                "environment": "production",
                "use_case": "rag"
            }
        )
        
        print(f"Index created: {index.resource_name}")
        self.index = index
        return index
    
    def create_index_endpoint(self) -> MatchingEngineIndexEndpoint:
        """Create an index endpoint"""
        print("Creating index endpoint...")
        
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.INDEX_ENDPOINT_DISPLAY_NAME,
            description="Endpoint for RAG index",
            public_endpoint_enabled=True,
            labels={
                "environment": "production",
                "use_case": "rag",
                "access": "public"
            }
        )
        
        print(f"Index endpoint created: {index_endpoint.resource_name}")
        self.index_endpoint = index_endpoint
        return index_endpoint
    
    def deploy_index(self, machine_type: str = "e2-standard-16") -> bool:
        """Deploy the index to the endpoint"""
        if not self.index or not self.index_endpoint:
            print("Index or endpoint not found")
            return False
        
        print("Deploying index to endpoint...")
        
        # Check if already deployed
        deployed_indexes = self.index_endpoint.deployed_indexes
        for deployed in deployed_indexes:
            if deployed.index == self.index.resource_name:
                print(f"Index is already deployed with ID: {deployed.id}")
                self.deployed_index_id = deployed.id
                return True
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deployed_id = f"deployed_{timestamp}"
        
        try:
            self.index_endpoint.deploy_index(
                index=self.index,
                deployed_index_id=deployed_id,
                display_name=f"Deployed {self.INDEX_DISPLAY_NAME}",
                machine_type=machine_type,
                min_replica_count=1,
                max_replica_count=2,
            )
            
            self.deployed_index_id = deployed_id
            print(f"Index deployed successfully with ID: {deployed_id}")
            
            # Wait for deployment to be ready
            self._wait_for_deployment(deployed_id)
            return True
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
    
    def _wait_for_deployment(self, deployed_id: str, max_wait_minutes: int = 10):
        """Wait for deployment to be ready"""
        print("Waiting for deployment to be ready...")
        
        for i in range(max_wait_minutes * 2):  # Check every 30 seconds
            time.sleep(30)
            
            # Refresh endpoint info
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint.resource_name
            )
            
            for deployed in endpoint.deployed_indexes:
                if deployed.id == deployed_id:
                    print("✓ Deployment completed and ready!")
                    return
            
            print(f"  ... still deploying ({i+1}/{max_wait_minutes * 2})")
        
        print("⚠ Deployment is taking longer than expected")
    
    def find_existing_resources(self):
        """Find existing index and endpoint"""
        # Find existing index
        indexes = aiplatform.MatchingEngineIndex.list(
            filter=f'display_name="{self.INDEX_DISPLAY_NAME}"'
        )
        
        if indexes:
            self.index = indexes[0]
            print(f"Found existing index: {self.index.display_name}")
        
        # Find existing endpoint
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{self.INDEX_ENDPOINT_DISPLAY_NAME}"'
        )
        
        if endpoints:
            self.index_endpoint = endpoints[0]
            print(f"Found existing endpoint: {self.index_endpoint.display_name}")
            
            # Check if index is deployed
            deployed_indexes = self.index_endpoint.deployed_indexes
            for deployed in deployed_indexes:
                if self.index and deployed.index == self.index.resource_name:
                    self.deployed_index_id = deployed.id
                    print(f"Found deployed index: {self.deployed_index_id}")
                    return True
        
        return False
    
    def check_deployment_status(self):
        """Check if the deployed index is ready"""
        if not self.index_endpoint or not self.deployed_index_id:
            return False
        
        try:
            # Try to get endpoint info
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint.resource_name
            )
            
            for deployed in endpoint.deployed_indexes:
                if deployed.id == self.deployed_index_id:
                    print(f"Deployed index found: {deployed.id}")
                    # Check if there's a state or status field
                    print(f"Deployed index details: {deployed}")
                    return True
            
            print("Deployed index not found")
            return False
            
        except Exception as e:
            print(f"Error checking deployment status: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, population: Optional[str] = None, 
               code_maane: Optional[str] = None) -> List[Dict]:
        """Search for similar documents"""
        if not self.index_endpoint or not self.deployed_index_id:
            print("System not ready. Please run setup first.")
            return []
        
        try:
            # Create query embedding
            print(f"Searching for: '{query}'")
            query_embedding = self.embedding_model.get_embeddings(
                [query], output_dimensionality=self.DIMENSIONS
            )[0].values
            
            # Build filters
            filters = []
            if population:
                filters.append({
                    "namespace": "population",
                    "allow_list": [population]
                })
            
            if code_maane:
                filters.append({
                    "namespace": "code_maane",
                    "allow_list": [code_maane]
                })
            
            # Perform search
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=top_k
            )
            
            # Process results
            results = []
            if response and len(response) > 0:
                for neighbor in response[0]:
                    similarity_score = 1 - neighbor.distance
                    
                    result = {
                        "id": neighbor.id,
                        "distance": neighbor.distance,
                        "similarity": similarity_score,
                        "similarity_percent": f"{similarity_score * 100:.1f}%"
                    }
                    results.append(result)
            
            print(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def setup_from_file(self, file_path: str) -> bool:
        """Complete setup process from a document file"""
        print("Starting setup process...")
        
        # Check for existing resources first
        if self.find_existing_resources():
            print("✓ System is already set up and ready!")
            return True
        
        try:
            # Load documents
            documents = self.load_documents(file_path)
            if not documents:
                print("No documents loaded. Aborting setup.")
                return False
            
            # Create embeddings
            embeddings = self.create_embeddings(documents)
            
            # Prepare for indexing
            embeddings_uri = self.prepare_embeddings_for_index(documents, embeddings)
            
            # Create index if needed
            if not self.index:
                self.create_index(embeddings_uri)
            
            # Create endpoint if needed
            if not self.index_endpoint:
                self.create_index_endpoint()
            
            # Deploy index
            if not self.deployed_index_id:
                self.deploy_index()
            
            print("✓ Setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"Setup failed: {e}")
            return False

  
        """Test the simple search method"""
        print("Testing simple search...")
        
        # Test with Hebrew query
        results = self.search2("פיזיקה", top_k=3)
        
        if results:
            print("SUCCESS! Found results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
            return True
        else:
            print("No results found")
            
            # Try with English
            print("Trying with English query...")
            results = self.search2("physics quantum", top_k=3)
            if results:
                print("SUCCESS with English!")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
                return True
            
            return False
    def check_endpoint_config(self):
        """Check if endpoint has public access"""
        if not self.index_endpoint:
            print("No endpoint found")
            return
        
        print("Endpoint configuration:")
        print(f"Resource name: {self.index_endpoint.resource_name}")
        print(f"Display name: {self.index_endpoint.display_name}")
        
        # Check if public endpoint is enabled
        try:
            endpoint_info = self.index_endpoint.to_dict()
            print(f"Endpoint details: {endpoint_info}")           
        except Exception as e:
            print(f"Error checking endpoint config: {e}")

    def search(self, query: str, top_k: int = 5, population: Optional[str] = None, 
            code_maane: Optional[str] = None) -> List[Dict]:
        """Search using the correct API structure"""
        if not self.index_endpoint or not self.deployed_index_id:
            print("System not ready. Please run setup first.")
            return []
        
        try:
            # Create query embedding
            print(f"Searching for: '{query}'")
            query_embedding = self.embedding_model.get_embeddings(
                [query], output_dimensionality=self.DIMENSIONS
            )[0].values
            
            print(f"Query embedding dimension: {len(query_embedding)}")
            
            # Use the correct v1 API with proper imports
            from google.cloud.aiplatform_v1.services.match_service import MatchServiceClient
            from google.cloud.aiplatform_v1.types import FindNeighborsRequest
            
            # Create client
            client = MatchServiceClient()
            
            # Get endpoint name
            endpoint_name = self.index_endpoint.resource_name
            print(f"Using endpoint: {endpoint_name}")
            print(f"Using deployed index: {self.deployed_index_id}")
            
            # Create the request with the correct structure
            # Based on the actual API documentation structure
            queries = [
                {
                    "datapoint": {
                        "datapoint_id": "query_0",
                        "feature_vector": query_embedding
                    },
                    "neighbor_count": top_k
                }
            ]
            
            request = FindNeighborsRequest(
                index_endpoint=endpoint_name,
                deployed_index_id=self.deployed_index_id,
                queries=queries
            )
            
            print("Executing search...")
            response = client.find_neighbors(request=request)
            
            print(f"Response received: {type(response)}")
            
            # Process results
            results = []
            if hasattr(response, 'nearest_neighbors') and response.nearest_neighbors:
                neighbors_list = response.nearest_neighbors[0]
                if hasattr(neighbors_list, 'neighbors'):
                    neighbors = neighbors_list.neighbors
                    print(f"Found {len(neighbors)} neighbors")
                    
                    for neighbor in neighbors:
                        similarity = 1 - neighbor.distance
                        neighbor_id = neighbor.datapoint.datapoint_id if hasattr(neighbor.datapoint, 'datapoint_id') else str(neighbor.datapoint)
                        
                        result = {
                            "id": neighbor_id,
                            "distance": neighbor.distance,
                            "similarity": similarity,
                            "similarity_percent": f"{similarity * 100:.1f}%"
                        }
                        results.append(result)
                        print(f"  Result: ID={result['id']}, Similarity={result['similarity_percent']}")
            else:
                print("No neighbors found in response")
            
            print(f"Total results: {len(results)}")
            return results
            
        except Exception as e:
            print(f"API search failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Final fallback: Try a simple HTTP request
            print("Trying direct HTTP request...")
            return self._search_direct_http(query, top_k)

    def _search_direct_http(self, query: str, top_k: int) -> List[Dict]:
        """Direct HTTP request to Vector Search endpoint"""
        try:
            import requests
            import json
            
            # Create embedding
            query_embedding = self.embedding_model.get_embeddings([query])[0].values
            
            # Get access token from gcloud (this should work since gcloud auth list shows active account)
            import subprocess
            
            # Get access token using gcloud
            result = subprocess.run([
                'gcloud', 'auth', 'print-access-token'
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode != 0:
                print("Failed to get access token from gcloud")
                return []
            
            access_token = result.stdout.strip()
            print("Got access token from gcloud")
            
            # Get endpoint info
            endpoint_info = self.index_endpoint.to_dict()
            public_domain = endpoint_info.get('publicEndpointDomainName')
            
            if not public_domain:
                print("No public domain available")
                return []
            
            # Make direct API call
            url = f"https://{public_domain}/v1/projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.index_endpoint.resource_name.split('/')[-1]}:findNeighbors"
            
            payload = {
                "deployedIndexId": self.deployed_index_id,
                "queries": [{
                    "datapoint": {
                        "datapointId": "query_0",
                        "featureVector": query_embedding
                    },
                    "neighborCount": top_k
                }]
            }
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            print(f"Making HTTP request to: {url}")
            response = requests.post(url, json=payload, headers=headers)
            
            print(f"HTTP Response status: {response.status_code}")
            print(f"Full HTTP Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"HTTP Success! Response keys: {result.keys()}")
                print(f"Full response content: {result}")
                
                results = []
                # Try different possible response formats
                if "nearestNeighbors" in result:
                    neighbors = result["nearestNeighbors"][0].get("neighbors", [])
                    print(f"Found {len(neighbors)} neighbors via HTTP")
                elif "nearest_neighbors" in result:
                    neighbors = result["nearest_neighbors"][0].get("neighbors", [])
                    print(f"Found {len(neighbors)} neighbors via HTTP (snake_case)")
                elif result:  # If there's any content but not in expected format
                    print(f"Unexpected response format: {result}")
                    neighbors = []
                else:
                    print("Empty response - this might indicate no matches found or wrong request format")
                    neighbors = []
                
                for neighbor in neighbors:
                    similarity = 1 - neighbor.get("distance", 1.0)
                    results.append({
                        "id": neighbor.get("datapoint", {}).get("datapointId", "unknown"),
                        "distance": neighbor.get("distance", 1.0),
                        "similarity": similarity,
                        "similarity_percent": f"{similarity * 100:.1f}%"
                    })
                
                # If we got an empty response, try with a simpler query
                if not results and not result:
                    print("Trying with a test embedding to see if endpoint responds...")
                    test_embedding = [0.1] * self.DIMENSIONS
                    test_payload = {
                        "deployedIndexId": self.deployed_index_id,
                        "queries": [{
                            "datapoint": {
                                "datapointId": "test_0",
                                "featureVector": test_embedding
                            },
                            "neighborCount": 1
                        }]
                    }
                    
                    test_response = requests.post(url, json=test_payload, headers=headers)
                    print(f"Test response status: {test_response.status_code}")
                    print(f"Test response: {test_response.text}")
                
                return results
            else:
                print(f"HTTP request failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Direct HTTP request failed: {e}")
            return []
   
def main():
    """Main function to demonstrate the RAG system"""
    # Initialize the system
    rag_system = VertexAIVectorSearchRAG()

     # Test embedding dimensions
    test_embedding = rag_system.embedding_model.get_embeddings(["test"])
    print(f"Actual embedding dimension: {len(test_embedding[0].values)}")

    # Check if system is already set up
    if rag_system.find_existing_resources():
        print("✓ Found existing resources")
        rag_system.check_endpoint_config()
        print("Testing vector search...")
        results = rag_system.search("פיזיקה קוונטית", top_k=3)
        
        if results:
            print("SUCCESS! Found results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
        else:
            print("Still no results found")

        # Check deployment status
        if rag_system.check_deployment_status():
            print("✓ Deployment appears ready")
            
            # Test basic search
            if rag_system.search("פיזיקה קוונטית"):
                print("✓ Basic search working")
            else:
                print("✗ Basic search failed")
        else:
            print("✗ Deployment not ready")
    else:
        print("No existing resources found - need to run setup")

    # Setup from file
    data_file = "files/data.txt"  # Change this to your file path
    
    if rag_system.setup_from_file(data_file):
        print("\n" + "="*50)
        print("Testing search functionality...")
        
        # Test searches
        test_queries = [
            "מהם עקרונות היסוד של פיזיקה קוונטית?"           
        ]
        
        for query in test_queries:
            print(f"\nSearching: {query}")
            results = rag_system.search(query, top_k=5)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
            else:
                print("  No results found")
    
    else:
        print("Setup failed. Please check the logs for errors.")


if __name__ == "__main__":
    main()