"""
Generate curl commands to test the endpoint manually
"""

import json
import shutil
import subprocess
from config.settings import Config
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager


def generate_curl_tests():
    """Generate curl test commands"""
    print("Generating curl test commands...")
    
    # Initialize components
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)
    
    # Load existing resources
    if not index_manager.find_existing_resources():
        print("Could not find resources")
        return
    
    # Get endpoint info
    endpoint_info = index_manager.index_endpoint.to_dict()
    public_domain = endpoint_info.get('publicEndpointDomainName')
    resource_name = index_manager.index_endpoint.resource_name
    project_number = resource_name.split('/')[1]
    endpoint_id = resource_name.split('/')[-1]
    
    # Create a simple test embedding
    test_embedding = gcp_client.create_embeddings(["test"])[0].values[:10]  # Just first 10 values for testing
    
    # Get access token
    try:
        gcloud_path = shutil.which('gcloud')
        if gcloud_path:
            result = subprocess.run([
                gcloud_path, 'auth', 'print-access-token'
            ], capture_output=True, text=True, check=False, shell=True)
            
            if result.returncode == 0:
                token = result.stdout.strip()
                print(f"Access token: {token[:20]}...")
            else:
                print("Could not get access token")
                return
    except Exception as e:
        print(f"Token error: {e}")
        return
    
    # Test payload
    payload = {
        "deployedIndexId": index_manager.deployed_index_id,
        "queries": [{
            "datapoint": {
                "datapointId": "test_query",
                "featureVector": test_embedding
            },
            "neighborCount": 3
        }]
    }
    
    # Generate different curl commands to try
    urls = [
        f"https://{public_domain}/v1/projects/{project_number}/locations/{config.location}/indexEndpoints/{endpoint_id}:findNeighbors",
        f"https://{public_domain}/v1/projects/{config.project_id}/locations/{config.location}/indexEndpoints/{endpoint_id}:findNeighbors",
        f"https://{public_domain}/v1beta1/projects/{project_number}/locations/{config.location}/indexEndpoints/{endpoint_id}:findNeighbors"
    ]
    
    print(f"\nEndpoint details:")
    print(f"Public domain: {public_domain}")
    print(f"Project number: {project_number}")
    print(f"Project ID: {config.project_id}")
    print(f"Location: {config.location}")
    print(f"Endpoint ID: {endpoint_id}")
    print(f"Deployed index ID: {index_manager.deployed_index_id}")
    
    print(f"\nTest curl commands:")
    print("="*80)
    
    for i, url in enumerate(urls, 1):
        print(f"\n# Test {i}:")
        curl_command = f'''curl -X POST \\
  -H "Authorization: Bearer {token}" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload)}' \\
  "{url}"'''
        
        print(curl_command)
        print()
    
    print("="*80)
    print("\nCopy and run these commands in your terminal to test manually.")
    print("Look for which one returns actual data instead of 501 errors.")


if __name__ == "__main__":
    generate_curl_tests()