"""
Test script using the simplified search engine
"""

from config.settings import Config
from services.gcp_client import GCPClientService
from core.vector_index import VectorIndexManager
from core.simple_search import SimpleSearchEngine


def test_simple_system():
    """Test the simplified search system"""
    print("Testing simplified search system...")
    
    # Initialize components
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)
    
    # Create simple search engine
    search_engine = SimpleSearchEngine(config, gcp_client, index_manager)
    
    # Test queries
    test_queries = [
        "פיזיקה קוונטית",
        "מהם עקרונות היסוד של פיזיקה קוונטית?",
        "physics",
        "quantum"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing query: '{query}'")
        print('='*50)
        
        results = search_engine.search(query, top_k=3)
        
        if results:
            success_count += 1
            print(f"✓ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
        else:
            print("✗ No results found")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {success_count}/{len(test_queries)} queries successful")
    print('='*50)
    
    return success_count > 0


def interactive_simple_search():
    """Interactive mode with simple search"""
    config = Config.from_env()
    gcp_client = GCPClientService(config)
    index_manager = VectorIndexManager(config)
    search_engine = SimpleSearchEngine(config, gcp_client, index_manager)
    
    print("Simple Search Interactive Mode")
    print("Type 'exit' to quit, 'debug' for system info")
    
    while True:
        query = input("\nSearch query: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'debug':
            print(f"Index endpoint: {index_manager.index_endpoint.display_name if index_manager.index_endpoint else 'None'}")
            print(f"Deployed index: {index_manager.deployed_index_id}")
            continue
        elif not query:
            continue
        
        results = search_engine.search(query, top_k=5)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
        else:
            print("No results found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_simple_search()
    else:
        test_simple_system()