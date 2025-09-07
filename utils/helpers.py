"""
Utility functions and helpers
"""

import traceback
from typing import Any, Dict, List


def print_search_results(results: List[Dict], query: str = ""):
    """Pretty print search results"""
    if not results:
        print("No results found")
        return
    
    if query:
        print(f"\nResults for query: '{query}'")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. ID: {result['id']}, Similarity: {result['similarity_percent']}")


def safe_execute(func, *args, error_message: str = "Operation failed", **kwargs) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"{error_message}: {e}")
        traceback.print_exc()
        return None


def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and has supported extension"""
    import os
    from pathlib import Path
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    supported_extensions = ['.json', '.txt']
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in supported_extensions:
        print(f"Unsupported file type: {file_ext}. Supported: {supported_extensions}")
        return False
    
    return True


def check_deployment_readiness(index_manager) -> bool:
    """Check if the deployment is ready for search operations"""
    if not index_manager.index_endpoint or not index_manager.deployed_index_id:
        print("✗ Index or endpoint not deployed")
        return False
    
    if not index_manager.check_deployment_status():
        print("✗ Deployment not ready")
        return False
    
    print("✓ Deployment ready for search")
    return True