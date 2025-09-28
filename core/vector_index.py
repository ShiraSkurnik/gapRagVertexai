"""
Vector index management for Vertex AI Vector Search
"""

import time
from datetime import datetime
from typing import Optional, Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint

from config.settings import Config


class VectorIndexManager:
    """Manages creation and deployment of vector search indexes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.index: Optional[MatchingEngineIndex] = None
        self.index_endpoint: Optional[MatchingEngineIndexEndpoint] = None
        self.deployed_index_id: Optional[str] = None
    
    def find_existing_resources(self) -> bool:
        """Find existing index and endpoint with improved error handling"""
        try:
            # Find existing index
            print(f"Searching for index with display name: {self.config.index_display_name}")
            indexes = aiplatform.MatchingEngineIndex.list(
                filter=f'display_name="{self.config.index_display_name}"'
            )
            
            if indexes:
                self.index = indexes[0]
                print(f"Found existing index: {self.index.display_name}")
            else:
                print("No matching index found")
            
            # Find existing endpoint
            print(f"Searching for endpoint with display name: {self.config.index_endpoint_display_name}")
            endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
                filter=f'display_name="{self.config.index_endpoint_display_name}"'
            )
            
            if endpoints:
                self.index_endpoint = endpoints[0]
                print(f"Found existing endpoint: {self.index_endpoint.display_name}")
                print(f"Endpoint resource name: {self.index_endpoint.resource_name}")
                
                # Check if index is deployed
                deployed_indexes = self.index_endpoint.deployed_indexes
                print(f"Found {len(deployed_indexes)} deployed indexes")
                
                for deployed in deployed_indexes:
                    print(f"Deployed index: {deployed.id}, Index: {deployed.index}")
                    if self.index and deployed.index == self.index.resource_name:
                        self.deployed_index_id = deployed.id
                        print(f"Found deployed index: {self.deployed_index_id}")
                        return True
                    elif not self.index:
                        # If we have a deployed index but couldn't find the index resource
                        self.deployed_index_id = deployed.id
                        print(f"Found deployed index without matching index resource: {self.deployed_index_id}")
                        return True
            else:
                print("No matching endpoint found")
                
                # Try to list all endpoints to debug
                print("Listing all available endpoints:")
                all_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
                for ep in all_endpoints:
                    print(f"  - {ep.display_name} ({ep.resource_name})")
        
        except Exception as e:
            print(f"Error in find_existing_resources: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def create_index(self, embeddings_gcs_uri: str) -> MatchingEngineIndex:
        """Create a new Vector Search index"""
        print("Creating new Vector Search index...")
        
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.config.index_display_name,
            contents_delta_uri=embeddings_gcs_uri,
            dimensions=self.config.dimensions,
            approximate_neighbors_count=self.config.approximate_neighbors_count,
            distance_measure_type=self.config.distance_measure_type,
            leaf_node_embedding_count=self.config.leaf_node_embedding_count,
            leaf_nodes_to_search_percent=self.config.leaf_nodes_to_search_percent,
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
            display_name=self.config.index_endpoint_display_name,
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
    
    def deploy_index(self) -> bool:
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
                display_name=f"Deployed {self.config.index_display_name}",
                machine_type=self.config.machine_type,
                min_replica_count=self.config.min_replica_count,
                max_replica_count=self.config.max_replica_count,
            )
            
            self.deployed_index_id = deployed_id
            print(f"Index deployed successfully with ID: {deployed_id}")
            
            # Wait for deployment to be ready
            self._wait_for_deployment(deployed_id)
            return True
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
    
    def _wait_for_deployment(self, deployed_id: str):
        """Wait for deployment to be ready"""
        print("Waiting for deployment to be ready...")
        
        for i in range(self.config.max_wait_minutes * 2):  # Check every 30 seconds
            time.sleep(30)
            
            # Refresh endpoint info
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint.resource_name
            )
            
            for deployed in endpoint.deployed_indexes:
                if deployed.id == deployed_id:
                    print("✓ Deployment completed and ready!")
                    return
            
            print(f"  ... still deploying ({i+1}/{self.config.max_wait_minutes * 2})")
        
        print("⚠ Deployment is taking longer than expected")
    
    def check_deployment_status(self) -> bool:
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
                    print(f"Deployed index details: {deployed}")
                    return True
            
            print("Deployed index not found")
            return False
            
        except Exception as e:
            print(f"Error checking deployment status: {e}")
            return False
    
    def get_deployment_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get deployment information"""
        return (
            self.index_endpoint.resource_name if self.index_endpoint else None,
            self.deployed_index_id
        )