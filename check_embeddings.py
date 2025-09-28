"""
Check the format and content of existing embedding files
"""

import json
from config.settings import Config
from services.gcp_client import GCPClientService


def check_embedding_files():
    """Check the content and format of embedding files in the bucket"""
    print("Checking embedding files...")
    
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    
    try:
        bucket = gcp_client.get_bucket()
        
        # Get the most recent embedding file
        blobs = sorted(
            bucket.list_blobs(prefix="index_data/embeddings_"), 
            key=lambda x: x.name, 
            reverse=True
        )
        
        if not blobs:
            print("No embedding files found")
            return
        
        latest_blob = blobs[0]
        print(f"Checking latest file: {latest_blob.name} (size: {latest_blob.size} bytes)")
        
        # Download and examine the file content
        content = latest_blob.download_as_text()
        lines = content.strip().split('\n')
        
        print(f"File has {len(lines)} lines (documents)")
        
        # Check first few entries
        for i, line in enumerate(lines[:3]):
            try:
                entry = json.loads(line)
                print(f"\nEntry {i+1}:")
                print(f"  ID: {entry.get('id', 'missing')}")
                print(f"  Embedding length: {len(entry.get('embedding', []))}")
                print(f"  Has restricts: {bool(entry.get('restricts'))}")
                print(f"  Crowding tag: {entry.get('crowding_tag', 'missing')}")
                
                # Check restricts format
                if 'restricts' in entry:
                    print(f"  Restricts: {entry['restricts']}")
                
                # Check if original data exists
                if 'original_data' in entry:
                    orig = entry['original_data']
                    content_preview = str(orig.get('content', ''))[:100] + "..."
                    print(f"  Content preview: {content_preview}")
                    print(f"  Metadata: {orig.get('metadata', {})}")
                
            except json.JSONDecodeError as e:
                print(f"Entry {i+1}: JSON decode error - {e}")
            except Exception as e:
                print(f"Entry {i+1}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"Error checking embedding files: {e}")
        return False


def rebuild_index_with_existing_data():
    """Rebuild the index using the most recent embedding file"""
    print("\nRebuild index using existing embedding file...")
    
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    
    try:
        bucket = gcp_client.get_bucket()
        
        # Get the most recent embedding file
        blobs = sorted(
            bucket.list_blobs(prefix="index_data/embeddings_"), 
            key=lambda x: x.name, 
            reverse=True
        )
        
        if not blobs:
            print("No embedding files found")
            return False
        
        latest_blob = blobs[0]
        embeddings_uri = f"gs://{config.bucket_name}/{latest_blob.name}"
        
        print(f"Using embedding file: {embeddings_uri}")
        
        # Delete the current index and create a new one
        from core.vector_index import VectorIndexManager
        from google.cloud import aiplatform
        
        index_manager = VectorIndexManager(config)
        
        # Find existing resources
        index_manager.find_existing_resources()
        
        # Undeploy the current index first
        if index_manager.index_endpoint and index_manager.deployed_index_id:
            print("Undeploying current index...")
            try:
                index_manager.index_endpoint.undeploy_index(
                    deployed_index_id=index_manager.deployed_index_id
                )
                print("✓ Index undeployed")
            except Exception as e:
                print(f"Warning: Could not undeploy index: {e}")
        
        # Delete the current index
        if index_manager.index:
            print("Deleting current index...")
            try:
                index_manager.index.delete()
                print("✓ Index deleted")
            except Exception as e:
                print(f"Warning: Could not delete index: {e}")
        
        # Create new index with the existing embedding data
        print("Creating new index...")
        new_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=config.index_display_name,
            contents_delta_uri=embeddings_uri,
            dimensions=config.dimensions,
            approximate_neighbors_count=100,
            distance_measure_type="COSINE_DISTANCE",
            leaf_node_embedding_count=1000,
            leaf_nodes_to_search_percent=7,
            description="Rebuilt RAG index with existing embedding data"
        )
        
        print(f"✓ New index created: {new_index.resource_name}")
        
        # Deploy to existing endpoint
        if index_manager.index_endpoint:
            print("Deploying new index...")
            from datetime import datetime
            deployed_id = f"deployed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            index_manager.index_endpoint.deploy_index(
                index=new_index,
                deployed_index_id=deployed_id,
                display_name=f"Deployed {config.index_display_name}",
                machine_type="e2-standard-16",
                min_replica_count=1,
                max_replica_count=2,
            )
            
            print(f"✓ Index deployed with ID: {deployed_id}")
        
        return True
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if check_embedding_files():
        print("\n" + "="*60)
        response = input("Do you want to rebuild the index with existing data? (y/n): ")
        if response.lower() == 'y':
            rebuild_index_with_existing_data()
        else:
            print("You can rebuild manually later or re-run setup with new data.")