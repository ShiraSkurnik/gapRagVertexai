"""
Debug utilities for troubleshooting the RAG system
"""

import json
import subprocess
from typing import Dict, Any

import requests
from google.cloud import aiplatform


def debug_deployment_info(index_manager):
    """Debug deployment information"""
    print("=== DEPLOYMENT DEBUG INFO ===")
    
    if not index_manager.index_endpoint:
        print("✗ No index endpoint found")
        return
    
    print(f"✓ Index endpoint: {index_manager.index_endpoint.display_name}")
    print(f"✓ Resource name: {index_manager.index_endpoint.resource_name}")
    print(f"✓ Deployed index ID: {index_manager.deployed_index_id}")
    
    # Get detailed endpoint info
    try:
        endpoint_dict = index_manager.index_endpoint.to_dict()
        
        print("\n--- Endpoint Details ---")
        print(f"Public endpoint enabled: {endpoint_dict.get('publicEndpointEnabled', False)}")
        print(f"Public domain: {endpoint_dict.get('publicEndpointDomainName', 'None')}")
        print(f"Network: {endpoint_dict.get('network', 'Default')}")
        
        # Deployed indexes info
        deployed_indexes = endpoint_dict.get('deployedIndexes', [])
        print(f"\nDeployed indexes count: {len(deployed_indexes)}")
        
        for i, deployed in enumerate(deployed_indexes):
            print(f"  Index {i+1}:")
            print(f"    ID: {deployed.get('id', 'Unknown')}")
            print(f"    Display name: {deployed.get('displayName', 'Unknown')}")
            print(f"    Index: {deployed.get('index', 'Unknown')}")
            
            # Check resources
            if 'dedicatedResources' in deployed:
                resources = deployed['dedicatedResources']
                print(f"    Machine type: {resources.get('machineSpec', {}).get('machineType', 'Unknown')}")
                print(f"    Replicas: {resources.get('minReplicaCount', 0)}-{resources.get('maxReplicaCount', 0)}")
            
    except Exception as e:
        print(f"Error getting endpoint details: {e}")


def debug_api_versions():
    """Check API versions and available methods"""
    print("\n=== API VERSION DEBUG ===")
    
    try:
        from google.cloud.aiplatform_v1 import MatchServiceClient
        from google.cloud.aiplatform_v1.types import FindNeighborsRequest
        
        client = MatchServiceClient()
        print(f"✓ MatchServiceClient available")
        print(f"✓ Client info: {type(client)}")
        
        # Check available methods
        methods = [method for method in dir(client) if not method.startswith('_')]
        print(f"✓ Available methods: {methods}")
        
    except Exception as e:
        print(f"✗ API version check failed: {e}")


def debug_authentication():
    """Debug authentication setup"""
    print("\n=== AUTHENTICATION DEBUG ===")
    
    try:
        # Check gcloud auth (Windows compatible)
        import shutil
        gcloud_path = shutil.which('gcloud')
        
        if not gcloud_path:
            print("✗ gcloud CLI not found in PATH")
        else:
            print(f"✓ gcloud found at: {gcloud_path}")
            
            try:
                result = subprocess.run([gcloud_path, 'auth', 'list'], 
                                       capture_output=True, text=True, check=False, shell=True)
                
                if result.returncode == 0:
                    print("✓ gcloud authentication:")
                    print(result.stdout)
                else:
                    print(f"✗ gcloud auth failed: {result.stderr}")
            except Exception as e:
                print(f"✗ gcloud command failed: {e}")
        
        # Check application default credentials
        try:
            from google.auth import default
            credentials, project = default()
            print(f"✓ Default credentials found for project: {project}")
            print(f"✓ Credentials type: {type(credentials)}")
        except Exception as e:
            print(f"✗ Default credentials error: {e}")
        
        # Test token (Windows compatible)
        if gcloud_path:
            try:
                result = subprocess.run([gcloud_path, 'auth', 'print-access-token'], 
                                       capture_output=True, text=True, check=False, shell=True)
                if result.returncode == 0:
                    token = result.stdout.strip()
                    print(f"✓ Access token obtained (length: {len(token)})")
                else:
                    print(f"✗ Token error: {result.stderr}")
            except Exception as e:
                print(f"✗ Token test failed: {e}")
            
    except Exception as e:
        print(f"Authentication debug failed: {e}")


def debug_endpoint_connectivity(index_manager):
    """Test endpoint connectivity"""
    print("\n=== CONNECTIVITY DEBUG ===")
    
    if not index_manager.index_endpoint:
        print("✗ No endpoint to test")
        return
    
    try:
        endpoint_info = index_manager.index_endpoint.to_dict()
        public_domain = endpoint_info.get('publicEndpointDomainName')
        
        if not public_domain:
            print("✗ No public domain available")
            return
        
        print(f"✓ Testing connectivity to: {public_domain}")
        
        # Test basic HTTPS connectivity
        test_url = f"https://{public_domain}"
        try:
            response = requests.get(test_url, timeout=10)
            print(f"✓ Basic connectivity: {response.status_code}")
        except requests.exceptions.Timeout:
            print("✗ Connection timeout")
        except requests.exceptions.ConnectionError:
            print("✗ Connection error")
        except Exception as e:
            print(f"✗ Connectivity error: {e}")
        
        # Test API endpoint structure
        endpoint_id = index_manager.index_endpoint.resource_name.split('/')[-1]
        project_id = index_manager.index_endpoint.resource_name.split('/')[1]
        location = index_manager.index_endpoint.resource_name.split('/')[3]
        
        api_url = f"https://{public_domain}/v1/projects/{project_id}/locations/{location}/indexEndpoints/{endpoint_id}:findNeighbors"
        print(f"API URL would be: {api_url}")
        
    except Exception as e:
        print(f"Connectivity debug failed: {e}")


def debug_resource_discovery(config):
    """Debug resource discovery issues"""
    print("\n=== RESOURCE DISCOVERY DEBUG ===")
    
    try:
        # List all indexes
        print("Listing all indexes:")
        all_indexes = aiplatform.MatchingEngineIndex.list()
        print(f"Found {len(all_indexes)} total indexes:")
        for i, index in enumerate(all_indexes):
            print(f"  {i+1}. Name: {index.display_name}")
            print(f"      Resource: {index.resource_name}")
            print(f"      Created: {index.create_time}")
            print()
        
        # List all endpoints
        print("Listing all index endpoints:")
        all_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
        print(f"Found {len(all_endpoints)} total endpoints:")
        for i, endpoint in enumerate(all_endpoints):
            print(f"  {i+1}. Name: {endpoint.display_name}")
            print(f"      Resource: {endpoint.resource_name}")
            print(f"      Created: {endpoint.create_time}")
            
            # Show deployed indexes
            deployed = endpoint.deployed_indexes
            print(f"      Deployed indexes: {len(deployed)}")
            for j, dep in enumerate(deployed):
                print(f"        {j+1}. ID: {dep.id}")
                print(f"           Index: {dep.index}")
                print(f"           Display: {dep.display_name}")
            print()
        
        # Test specific filters
        print(f"Testing filter for index: display_name=\"{config.index_display_name}\"")
        filtered_indexes = aiplatform.MatchingEngineIndex.list(
            filter=f'display_name="{config.index_display_name}"'
        )
        print(f"Filtered result count: {len(filtered_indexes)}")
        
        print(f"Testing filter for endpoint: display_name=\"{config.index_endpoint_display_name}\"")
        filtered_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{config.index_endpoint_display_name}"'
        )
        print(f"Filtered result count: {len(filtered_endpoints)}")
        
    except Exception as e:
        print(f"Resource discovery debug failed: {e}")
        import traceback
        traceback.print_exc()
    """Debug the search request format"""
    print("\n=== REQUEST FORMAT DEBUG ===")
    
    try:
        # Create a simple test embedding
        test_embedding = gcp_client.create_embeddings(["test"])[0].values
        print(f"✓ Test embedding created (dimension: {len(test_embedding)})")
        
        # Show different request formats
        print("\n--- Request Format 1 (camelCase) ---")
        format1 = {
            "deployedIndexId": "test_id",
            "queries": [{
                "datapoint": {
                    "datapointId": "query_0",
                    "featureVector": test_embedding[:5]  # Show first 5 values
                },
                "neighborCount": 5
            }]
        }
        print(json.dumps(format1, indent=2))
        
        print("\n--- Request Format 2 (snake_case) ---")
        format2 = {
            "deployed_index_id": "test_id",
            "queries": [{
                "datapoint": {
                    "datapoint_id": "query_0",
                    "feature_vector": test_embedding[:5]
                },
                "neighbor_count": 5
            }]
        }
        print(json.dumps(format2, indent=2))
        
    except Exception as e:
        print(f"Request format debug failed: {e}")


def run_full_debug(rag_system):
    """Run all debug checks"""
    print("Starting full debug session...\n")
    
    debug_resource_discovery(rag_system.config)
    debug_deployment_info(rag_system.index_manager)
    debug_api_versions()
    debug_authentication()
    debug_endpoint_connectivity(rag_system.index_manager)
    debug_search_request_format(rag_system.config, rag_system.gcp_client)
    
    # Test endpoint connectivity method if endpoint exists
    if rag_system.index_manager.index_endpoint:
        rag_system.search_engine.test_endpoint_connectivity()
    
    print("\n=== DEBUG COMPLETE ===")
    print("Next steps:")
    print("1. Check if resources are found properly")
    print("2. Verify resource names match exactly")  
    print("3. Check if public endpoint is enabled")
    print("4. Verify API permissions")
    print("5. Try manual curl request")
    print("6. Check Vertex AI API quotas/limits")