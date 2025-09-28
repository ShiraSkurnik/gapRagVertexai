"""
Diagnostic tool to check index status and content
"""

import json
from config.settings import Config
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager
from google.cloud import aiplatform


def diagnose_index():
    """Diagnose index status and content"""
    print("Diagnosing index status...")
    
    # Initialize components
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)
    
    # Load existing resources
    if not index_manager.find_existing_resources():
        print("Could not find resources")
        return
    
    print(f"\n=== INDEX INFORMATION ===")
    index = index_manager.index
    endpoint = index_manager.index_endpoint
    
    print(f"Index name: {index.display_name}")
    print(f"Index resource: {index.resource_name}")
    print(f"Index created: {index.create_time}")
    
    # Get detailed index information
    try:
        index_dict = index.to_dict()
        print(f"\nIndex details:")
        print(f"  State: {index_dict.get('state', 'Unknown')}")
        print(f"  Dimensions: {index_dict.get('indexStats', {}).get('vectorsCount', 'Unknown')} vectors")
        print(f"  Index update time: {index_dict.get('indexUpdateTime', 'Unknown')}")
        
        # Check metadata schema
        if 'metadataSchemaUri' in index_dict:
            print(f"  Metadata schema: {index_dict['metadataSchemaUri']}")
        
        # Check algorithm config
        if 'algorithmConfig' in index_dict:
            algo_config = index_dict['algorithmConfig']
            print(f"  Algorithm: {algo_config.get('treeAhConfig', {}).get('leafNodeEmbeddingCount', 'Unknown')}")
    
    except Exception as e:
        print(f"Error getting index details: {e}")
    
    print(f"\n=== ENDPOINT INFORMATION ===")
    print(f"Endpoint name: {endpoint.display_name}")
    print(f"Endpoint resource: {endpoint.resource_name}")
    print(f"Endpoint created: {endpoint.create_time}")
    
    # Get detailed endpoint information
    try:
        endpoint_dict = endpoint.to_dict()
        print(f"\nEndpoint details:")
        print(f"  Public endpoint enabled: {endpoint_dict.get('publicEndpointEnabled', False)}")
        print(f"  Public domain: {endpoint_dict.get('publicEndpointDomainName', 'None')}")
        
        # Check deployed indexes
        deployed_indexes = endpoint_dict.get('deployedIndexes', [])
        print(f"  Deployed indexes: {len(deployed_indexes)}")
        
        for i, deployed in enumerate(deployed_indexes):
            print(f"    {i+1}. ID: {deployed.get('id')}")
            print(f"       Index: {deployed.get('index')}")
            print(f"       Created: {deployed.get('createTime')}")
            print(f"       Index sync time: {deployed.get('indexSyncTime')}")
            
            # Check if fully deployed
            create_time = deployed.get('createTime', {})
            sync_time = deployed.get('indexSyncTime', {})
            print(f"       Status: {'Synced' if sync_time else 'Syncing'}")
    
    except Exception as e:
        print(f"Error getting endpoint details: {e}")
    
    # Check if we can read any datapoints
    print(f"\n=== DATAPOINT CHECK ===")
    try:
        from google.cloud.aiplatform_v1.services.match_service import MatchServiceClient
        from google.cloud.aiplatform_v1.types import ReadIndexDatapointsRequest
        
        client = MatchServiceClient()
        
        request = ReadIndexDatapointsRequest(
            index_endpoint=index_manager.index_endpoint.resource_name,
            deployed_index_id=index_manager.deployed_index_id,
            ids=["0", "1", "2"]  # Try to read first few datapoints
        )
        
        response = client.read_index_datapoints(request=request)
        
        if hasattr(response, 'datapoints') and response.datapoints:
            print(f"✓ Found {len(response.datapoints)} datapoints in index")
            for i, dp in enumerate(response.datapoints[:2]):  # Show first 2
                print(f"  Datapoint {i+1}: ID={dp.datapoint_id}")
                if hasattr(dp, 'restricts') and dp.restricts:
                    print(f"    Restricts: {[r.namespace + ':' + str(r.allow_list) for r in dp.restricts]}")
        else:
            print("✗ No datapoints found in index - this explains the empty search results!")
    
    except Exception as e:
        print(f"Could not read datapoints: {e}")
    
    # Test a simple embedding to verify dimensions
    print(f"\n=== EMBEDDING TEST ===")
    try:
        test_embedding = gcp_client.create_embeddings(["test"])[0]
        print(f"✓ Embedding service working - dimension: {len(test_embedding.values)}")
        print(f"  Sample values: {test_embedding.values[:5]}")
    except Exception as e:
        print(f"✗ Embedding service error: {e}")
    
    # Check if there are any files in the bucket that should have been indexed
    print(f"\n=== BUCKET CHECK ===")
    try:
        bucket = gcp_client.get_bucket()
        blobs = list(bucket.list_blobs(prefix="index_data/"))
        
        if blobs:
            print(f"✓ Found {len(blobs)} files in bucket:")
            for blob in blobs[:5]:  # Show first 5
                print(f"  - {blob.name} (size: {blob.size} bytes)")
        else:
            print("✗ No index data files found in bucket!")
    except Exception as e:
        print(f"Bucket check error: {e}")


def suggest_fixes():
    """Suggest potential fixes based on diagnosis"""
    print(f"\n=== SUGGESTED FIXES ===")
    print("Based on the empty search results, try these solutions:")
    print()
    print("1. **Re-index your data** - The index might be empty")
    print("   python main.py setup_from_file your_data_file.txt")
    print()
    print("2. **Check deployment sync** - Wait for index sync to complete")
    print("   The indexSyncTime should be recent")
    print()
    print("3. **Verify data format** - Ensure your source data was processed correctly")
    print("   Check the bucket for properly formatted embeddings files")
    print()
    print("4. **Test with broader query** - Try very general terms")
    print("   python test_simple.py interactive")
    print()


if __name__ == "__main__":
    diagnose_index()
    suggest_fixes()