"""
Main entry point for the Vertex AI Vector Search RAG System
"""

from core.rag_system import VertexAIVectorSearchRAG
from utils.debug import run_full_debug


def main():
    """Main function to demonstrate the RAG system"""
    # Initialize the system
    rag_system = VertexAIVectorSearchRAG()

    # Test embedding dimensions
    print(f"Actual embedding dimension: {rag_system.get_embedding_dimensions()}")

    # Check if system is already set up
    if rag_system.check_system_status():
        print("✓ System is ready for search!")
        
        # Test search functionality
        if rag_system.test_search():
            print("✓ Search functionality working")
        else:
            print("✗ Search functionality failed")
    else:
        print("System not ready - need to run setup")
        
        # Setup from file
        data_file = "files/data.txt"  # Change this to your file path
        
        if rag_system.setup_from_file(data_file):
            print("✓ Setup completed successfully!")
            
            # Test after setup
            rag_system.test_search()
        else:
            print("✗ Setup failed. Please check the logs for errors.")


def interactive_search():
    """Interactive search mode"""
    rag_system = VertexAIVectorSearchRAG()
    
    if not rag_system.check_system_status():
        print("System not ready. Please run setup first.")
        return
    
    print("Interactive search mode. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        results = rag_system.search(query, top_k=5)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")
        else:
            print("No results found")


def debug_mode():
    """Debug mode to troubleshoot issues"""
    rag_system = VertexAIVectorSearchRAG()
    run_full_debug(rag_system)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_search()
        elif sys.argv[1] == "debug":
            debug_mode()
        else:
            print("Available modes: interactive, debug")
    else:
        main()