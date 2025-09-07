"""
Simplified search implementation that should work with your existing deployment
"""

import json
import shutil
import subprocess
from typing import Dict, List, Optional

import requests
from google.auth import default
from google.auth.transport.requests import Request

from config.settings import Config
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager


class SimpleSearchEngine:
    """A simplified search engine that focuses on what works"""
    
    def __init__(self, config: Config, gcp_client: GCPClientService, 
                 index_manager: VectorIndexManager):
        self.config = config
        self.gcp_client = gcp_client
        self.index_manager = index_manager
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple search implementation"""
        print(f"Searching for: '{query}'")
        
        # Ensure we have the resources loaded
        if not self.index_manager.index_endpoint or not self.index_manager.deployed_index_id:
            print("Loading existing resources...")
            if not self.index_manager.find_existing_resources():
                print("✗ Could not find existing resources")
                return []
        
        # Create query embedding
        try:
            embeddings = self.gcp_client.create_embeddings([query])
            query_embedding = embeddings[0].values
            print(f"✓ Created embedding (dimension: {len(query_embedding)})")
        except Exception as e:
            print(f"✗ Failed to create embedding: {e}")
            return []
        
        # Try different search approaches - stop at first success
        search_methods = [
            ("Direct client API", self._search_with_direct_client),
            ("Public HTTP endpoint", self._search_with_http_public),
            ("Google APIs endpoint", self._search_with_http_googleapis)
        ]
        
        for method_name, method in search_methods:
            try:
                print(f"Trying {method_name}...")
                results = method(query_embedding, top_k)
                if results:  # If we got results, return immediately
                    print(f"✓ {method_name} succeeded! Found {len(results)} results")
                    return results
                else:
                    print(f"✗ {method_name} returned no results")
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                continue
        
        print("All search methods failed")
        return []
    
    def _search_with_direct_client(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Try using the endpoint's find_neighbors method directly"""
        response = self.index_manager.index_endpoint.find_neighbors(
            deployed_index_id=self.index_manager.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k
        )
        
        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                similarity = 1 - neighbor.distance
                results.append({
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                    "similarity": similarity,
                    "similarity_percent": f"{similarity * 100:.1f}%"
                })
        
        return results
    
    def _search_with_http_public(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Search using public endpoint HTTP API"""
        # Get endpoint info
        endpoint_info = self.index_manager.index_endpoint.to_dict()
        public_domain = endpoint_info.get('publicEndpointDomainName')
        
        if not public_domain:
            raise Exception("No public domain available")
        
        # Get access token
        token = self._get_access_token()
        if not token:
            raise Exception("Could not get access token")
        
        # Extract the correct project number from the resource name
        resource_name = self.index_manager.index_endpoint.resource_name
        project_number = resource_name.split('/')[1]  # Extract project number from resource name
        endpoint_id = resource_name.split('/')[-1]
        
        # Use the working URL format (project number)
        url = f"https://{public_domain}/v1/projects/{project_number}/locations/{self.config.location}/indexEndpoints/{endpoint_id}:findNeighbors"
        
        payload = {
            "deployedIndexId": self.index_manager.deployed_index_id,
            "queries": [{
                "datapoint": {
                    "datapointId": f"query_{hash(str(query_embedding[:10]))}",
                    "featureVector": query_embedding
                },
                "neighborCount": top_k
            }]
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        print(f"Making request to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Raw response: {json.dumps(response_data, indent=2)}")
            
            # Process the successful response
            results = self._process_http_response(response_data)
            print(f"Processed {len(results)} results")
            return results
        else:
            print(f"Error response: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def _search_with_http_googleapis(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Search using googleapis.com endpoint"""
        token = self._get_access_token()
        if not token:
            raise Exception("Could not get access token")
        
        url = f"https://aiplatform.googleapis.com/v1/{self.index_manager.index_endpoint.resource_name}:findNeighbors"
        
        payload = {
            "deployedIndexId": self.index_manager.deployed_index_id,
            "queries": [{
                "datapoint": {
                    "datapointId": f"query_{hash(str(query_embedding[:10]))}",
                    "featureVector": query_embedding
                },
                "neighborCount": top_k
            }]
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        return self._process_http_response(response.json())
    
    def _get_access_token(self) -> Optional[str]:
        """Get access token with multiple fallback methods"""
        # Method 1: gcloud CLI
        try:
            gcloud_path = shutil.which('gcloud')
            if gcloud_path:
                result = subprocess.run([
                    gcloud_path, 'auth', 'print-access-token'
                ], capture_output=True, text=True, check=False, shell=True)
                
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception:
            pass
        
        # Method 2: Application Default Credentials
        try:
            credentials, _ = default()
            credentials.refresh(Request())
            return credentials.token
        except Exception:
            pass
        
        return None
    
    def _process_http_response(self, response_data: dict) -> List[Dict]:
        """Process HTTP response and extract neighbors"""
        results = []
        
        # Handle different response formats
        neighbors = []
        if "nearestNeighbors" in response_data:
            neighbors_data = response_data["nearestNeighbors"]
            if neighbors_data and len(neighbors_data) > 0:
                neighbors = neighbors_data[0].get("neighbors", [])
        elif "nearest_neighbors" in response_data:
            neighbors_data = response_data["nearest_neighbors"]
            if neighbors_data and len(neighbors_data) > 0:
                neighbors = neighbors_data[0].get("neighbors", [])
        
        for neighbor in neighbors:
            distance = neighbor.get("distance", 1.0)
            similarity = 1 - distance
            
            # Extract neighbor ID
            neighbor_id = "unknown"
            if "datapoint" in neighbor:
                datapoint = neighbor["datapoint"]
                neighbor_id = datapoint.get("datapointId", datapoint.get("datapoint_id", "unknown"))
            elif "id" in neighbor:
                neighbor_id = neighbor["id"]
            
            results.append({
                "id": neighbor_id,
                "distance": distance,
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%"
            })
        
        return results


def test_simple_search():
    """Test function for the simple search"""
    from config.settings import Config
    from services.gcp_client import GCPClientService
    from core.vector_index import VectorIndexManager
    
    # Initialize components
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)
    
    # Create simple search engine
    search_engine = SimpleSearchEngine(config, gcp_client, index_manager)
    
    # Test search
    results = search_engine.search("פיזיקה קוונטית", top_k=3)
    
    if results:
        print("✓ Search successful!")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
    else:
        print("✗ Search failed")
    
    return len(results) > 0


if __name__ == "__main__":
    test_simple_search()