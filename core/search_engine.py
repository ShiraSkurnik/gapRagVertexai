"""
Search engine for vector similarity search with improved error handling
"""

import json
import subprocess
from typing import Dict, List, Optional

import requests
from google.cloud.aiplatform_v1.types import FindNeighborsRequest

from config.settings import Config
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager


class SearchEngine:
    """Handles vector similarity search operations"""
    
    def __init__(self, config: Config, gcp_client: GCPClientService, 
                 index_manager: VectorIndexManager):
        self.config = config
        self.gcp_client = gcp_client
        self.index_manager = index_manager
    
    def search(self, query: str, top_k: Optional[int] = None, 
               population: Optional[str] = None, 
               code_maane: Optional[str] = None) -> List[Dict]:
        """Search for similar documents with multiple fallback methods"""
        top_k = top_k or self.config.default_top_k
        
        endpoint_name, deployed_index_id = self.index_manager.get_deployment_info()
        
        if not endpoint_name or not deployed_index_id:
            print("System not ready. Please run setup first.")
            return []
        
        try:
            # Create query embedding
            print(f"Searching for: '{query}'")
            query_embedding = self.gcp_client.create_embeddings([query])[0].values
            print(f"Query embedding dimension: {len(query_embedding)}")
            
            # Try multiple search methods in order of preference
            search_methods = [
                ("High-level client API", self._search_with_high_level_api),
                ("Direct HTTP to public endpoint", self._search_with_public_http),
                ("HTTP with resource name", self._search_with_resource_http),
                ("Alternative API format", self._search_with_alternative_api)
            ]
            
            for method_name, search_method in search_methods:
                print(f"Trying {method_name}...")
                try:
                    results = search_method(query_embedding, top_k, endpoint_name, deployed_index_id)
                    if results:
                        print(f"✓ Success with {method_name}")
                        print(f"Total results: {len(results)}")
                        return results
                    else:
                        print(f"✗ {method_name} returned no results")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
                    continue
            
            print("All search methods failed")
            return []
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _search_with_high_level_api(self, query_embedding: List[float], top_k: int,
                                   endpoint_name: str, deployed_index_id: str) -> List[Dict]:
        """Search using the high-level aiplatform client"""
        from google.cloud import aiplatform
        
        # Try to use the index endpoint directly
        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_name
        )
        
        # Use the find_neighbors method
        response = endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k
        )
        
        return self._process_high_level_response(response)
    
    def _search_with_public_http(self, query_embedding: List[float], top_k: int,
                                endpoint_name: str, deployed_index_id: str) -> List[Dict]:
        """Search using HTTP to public endpoint"""
        # Get access token
        access_token = self._get_access_token()
        if not access_token:
            raise Exception("Could not obtain access token")
        
        # Get public domain
        endpoint_info = self.index_manager.index_endpoint.to_dict()
        public_domain = endpoint_info.get('publicEndpointDomainName')
        
        if not public_domain:
            raise Exception("No public domain available")
        
        # Extract endpoint ID from resource name
        endpoint_id = endpoint_name.split('/')[-1]
        
        # Construct URL
        url = f"https://{public_domain}/v1/projects/{self.config.project_id}/locations/{self.config.location}/indexEndpoints/{endpoint_id}:findNeighbors"
        
        payload = {
            "deployedIndexId": deployed_index_id,
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
        
        print(f"Making request to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            return self._process_http_response(response.json())
        else:
            print(f"Response content: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def _search_with_resource_http(self, query_embedding: List[float], top_k: int,
                                  endpoint_name: str, deployed_index_id: str) -> List[Dict]:
        """Search using HTTP with full resource name"""
        access_token = self._get_access_token()
        if not access_token:
            raise Exception("Could not obtain access token")
        
        # Use the full resource name approach
        url = f"https://aiplatform.googleapis.com/v1/{endpoint_name}:findNeighbors"
        
        payload = {
            "deployedIndexId": deployed_index_id,
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
        
        print(f"Making request to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return self._process_http_response(response.json())
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def _search_with_alternative_api(self, query_embedding: List[float], top_k: int,
                                   endpoint_name: str, deployed_index_id: str) -> List[Dict]:
        """Search using alternative API format"""
        # Try using the match client with different request format
        from google.cloud.aiplatform_v1.types import FindNeighborsRequest
        
        # Alternative request structure
        queries = [{
            "datapoint": {
                "datapoint_id": "query_0",
                "feature_vector": query_embedding
            },
            "neighbor_count": top_k
        }]
        
        request = FindNeighborsRequest(
            index_endpoint=endpoint_name,
            deployed_index_id=deployed_index_id,
            queries=queries,
            return_full_datapoint=False
        )
        
        response = self.gcp_client.match_client.find_neighbors(request=request)
        return self._process_api_response(response)
    
    def _get_access_token(self) -> Optional[str]:
        """Get access token from gcloud with better error handling (Windows compatible)"""
        try:
            import shutil
            
            # Find gcloud executable
            gcloud_path = shutil.which('gcloud')
            if not gcloud_path:
                print("gcloud CLI not found in PATH")
                return self._get_token_from_credentials()
            
            # Try gcloud first
            result = subprocess.run([
                gcloud_path, 'auth', 'print-access-token'
            ], capture_output=True, text=True, check=False, shell=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            print(f"gcloud auth failed: {result.stderr}")
            return self._get_token_from_credentials()
            
        except Exception as e:
            print(f"Error getting access token: {e}")
            return self._get_token_from_credentials()
    
    def _get_token_from_credentials(self) -> Optional[str]:
        """Get token using application default credentials"""
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            
            credentials, project = default()
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            print(f"Error getting token from credentials: {e}")
            return None
    
    def _process_high_level_response(self, response) -> List[Dict]:
        """Process high-level API response"""
        results = []
        
        if response and len(response) > 0:
            for neighbor in response[0]:
                similarity = 1 - neighbor.distance
                result = {
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                    "similarity": similarity,
                    "similarity_percent": f"{similarity * 100:.1f}%"
                }
                results.append(result)
                print(f"  Result: ID={result['id']}, Similarity={result['similarity_percent']}")
        
        return results
    
    def _process_api_response(self, response) -> List[Dict]:
        """Process low-level API response"""
        results = []
        
        if hasattr(response, 'nearest_neighbors') and response.nearest_neighbors:
            neighbors_list = response.nearest_neighbors[0]
            if hasattr(neighbors_list, 'neighbors'):
                neighbors = neighbors_list.neighbors
                print(f"Found {len(neighbors)} neighbors")
                
                for neighbor in neighbors:
                    similarity = 1 - neighbor.distance
                    neighbor_id = (neighbor.datapoint.datapoint_id 
                                 if hasattr(neighbor.datapoint, 'datapoint_id') 
                                 else str(neighbor.datapoint))
                    
                    result = {
                        "id": neighbor_id,
                        "distance": neighbor.distance,
                        "similarity": similarity,
                        "similarity_percent": f"{similarity * 100:.1f}%"
                    }
                    results.append(result)
                    print(f"  Result: ID={result['id']}, Similarity={result['similarity_percent']}")
        
        return results
    
    def _process_http_response(self, response_data: dict) -> List[Dict]:
        """Process HTTP response with detailed logging"""
        print(f"HTTP Response keys: {list(response_data.keys())}")
        print(f"Full response: {json.dumps(response_data, indent=2)}")
        
        results = []
        
        # Handle different response formats
        neighbors = []
        if "nearestNeighbors" in response_data:
            neighbors_data = response_data["nearestNeighbors"]
            if neighbors_data and len(neighbors_data) > 0:
                neighbors = neighbors_data[0].get("neighbors", [])
                print(f"Found {len(neighbors)} neighbors in nearestNeighbors")
        elif "nearest_neighbors" in response_data:
            neighbors_data = response_data["nearest_neighbors"]
            if neighbors_data and len(neighbors_data) > 0:
                neighbors = neighbors_data[0].get("neighbors", [])
                print(f"Found {len(neighbors)} neighbors in nearest_neighbors")
        else:
            print("No neighbors found in expected response format")
            print("Available keys:", list(response_data.keys()))
        
        for i, neighbor in enumerate(neighbors):
            print(f"Processing neighbor {i}: {neighbor}")
            
            distance = neighbor.get("distance", 1.0)
            similarity = 1 - distance
            
            # Try different ways to get the ID
            neighbor_id = "unknown"
            if "datapoint" in neighbor:
                datapoint = neighbor["datapoint"]
                neighbor_id = datapoint.get("datapointId", datapoint.get("datapoint_id", "unknown"))
            elif "id" in neighbor:
                neighbor_id = neighbor["id"]
            
            result = {
                "id": neighbor_id,
                "distance": distance,
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%"
            }
            results.append(result)
            print(f"  Processed result: ID={result['id']}, Similarity={result['similarity_percent']}")
        
        return results
    
    def test_endpoint_connectivity(self):
        """Test if we can connect to the endpoint"""
        try:
            endpoint_info = self.index_manager.index_endpoint.to_dict()
            print(f"Endpoint info: {json.dumps(endpoint_info, indent=2, default=str)}")
            
            public_domain = endpoint_info.get('publicEndpointDomainName')
            if public_domain:
                print(f"Public domain: {public_domain}")
                
                # Test basic connectivity
                test_url = f"https://{public_domain}"
                response = requests.get(test_url, timeout=10)
                print(f"Connectivity test: {response.status_code}")
            else:
                print("No public domain found")
                
        except Exception as e:
            print(f"Connectivity test failed: {e}")