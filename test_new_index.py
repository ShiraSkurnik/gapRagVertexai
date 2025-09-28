"""
Test the rebuilt index with different approaches
"""

import json
import time
from datetime import datetime, timezone
import dateutil.parser
from config.settings import Config
from files.testVector import testVectorArr
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager
import requests
import shutil
import subprocess


def test_rebuilt_index():
    """Test if the rebuilt index actually contains data"""
    print("Testing rebuilt index...")

    # Initialize services
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)

    if not index_manager.find_existing_resources():
        print("Could not find resources")
        return

    print(f"Testing with deployed index: {index_manager.deployed_index_id}")

    # --- TEST 1: Zero-vector or sample-vector search ---
    print("\n=== TEST 1: Search test with a sample vector ===")
    try:
        # Use gcloud to get access token
        gcloud_path = shutil.which('gcloud')
        if gcloud_path is None:
            raise RuntimeError("gcloud CLI not found in PATH")

        result = subprocess.run(
            [gcloud_path, 'auth', 'print-access-token'],
            capture_output=True, text=True, check=True, shell=True
        )
        token = result.stdout.strip()

        # Use a vector you know exists or fallback to zero vector
        test_vector = testVectorArr
        print(f"test_vector.length: {len(test_vector)}")
   
        endpoint_info = index_manager.index_endpoint.to_dict()
        public_domain = endpoint_info.get('publicEndpointDomainName')
        if not public_domain:
            raise RuntimeError("Index endpoint public domain not found")

        resource_name = index_manager.index_endpoint.resource_name
        project_number = resource_name.split('/')[1]
        endpoint_id = resource_name.split('/')[-1]

        url = f"https://{public_domain}/v1/projects/{project_number}/locations/{config.location}/indexEndpoints/{endpoint_id}:findNeighbors"

        payload = {
            "deployedIndexId": index_manager.deployed_index_id,
            "queries": [{
                "datapoint": {
                    "datapointId": "test_query",
                    "featureVector": test_vector
                },
                "neighborCount": 5
            }]
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        print("Sending search query...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("neighbors"):
                print(f"✓ Search returned {len(response_data['neighbors'][0])} neighbors")
                for neighbor in response_data['neighbors'][0]:
                    print(f"  - Neighbor ID: {neighbor['datapoint']['datapointId']}, Distance: {neighbor.get('distance')}")
            else:
                print("⚠ Search returned no results. Index might be empty or vectors do not match the query")
        else:
            print(f"✗ Request failed: {response.text}")

    except Exception as e:
        print(f"✗ Search test failed: {e}")

    # --- TEST 2: Index statistics ---
    print("\n=== TEST 2: Index statistics ===")
    try:
        index_dict = index_manager.index.to_dict()
        state = index_dict.get('state', 'Unknown')
        shards_count = index_dict.get('shardsCount', 'Unknown')
        print(f"Index state: {state}")
        print(f"Shards count: {shards_count}")

        if 'indexStats' in index_dict:
            stats = index_dict['indexStats']
            print(f"Vector count: {stats.get('vectorsCount', 'Unknown')}")
        else:
            print("Vector count: Unknown (stats not available from SDK)")

        print(f"Index update time: {index_dict.get('indexUpdateTime', 'Unknown')}")

    except Exception as e:
        print(f"✗ Statistics check failed: {e}")

    # --- TEST 3: Index readiness check ---
    print("\n=== TEST 3: Index readiness check ===")
    try:
        endpoint_dict = index_manager.index_endpoint.to_dict()
        deployed_indexes = endpoint_dict.get('deployedIndexes', [])

        for deployed in deployed_indexes:
            if deployed.get('id') == index_manager.deployed_index_id:
                create_time = deployed.get('createTime')
                sync_time = deployed.get('indexSyncTime')
                print(f"Deployed at: {create_time}")
                print(f"Last sync: {sync_time}")

                if create_time:
                    created = dateutil.parser.parse(create_time)
                    now = datetime.now(timezone.utc)
                    age_minutes = (now - created).total_seconds() / 60
                    print(f"Index age: {age_minutes:.1f} minutes")

                    if age_minutes < 5:
                        print("⚠ Index is very new - might still be building")
                        print("Consider waiting a few minutes and trying again")
                    else:
                        print("✓ Index should be ready")
                break

    except Exception as e:
        print(f"✗ Readiness check failed: {e}")


if __name__ == "__main__":
    test_rebuilt_index()
