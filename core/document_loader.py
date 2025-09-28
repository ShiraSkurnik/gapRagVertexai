"""
Document loading and processing utilities
"""

import json
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import Config


class DocumentLoader:
    """Handles loading and processing of documents from various sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load and process documents from JSON or TXT files"""
        documents = []
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.json':
                documents = self._load_json_documents(file_path)
            elif file_ext == '.txt':
                documents = self._load_text_documents(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            print(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    def _load_json_documents(self, file_path: str) -> List[Document]:
        """Load documents from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data):
            # Assign population based on index
            population = self._get_population_by_index(i)
            
            content = json.dumps(item, indent=2, ensure_ascii=False)
            metadata = {
                "population": population,
                "type": "json",
                "code_maane": str(item.get("קוד_מענה", "")),
                "index": i
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _load_text_documents(self, file_path: str) -> List[Document]:
        """Load documents from text file with chunking"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Assign population based on chunk index
            population = self._get_population_by_index(i)
            
            metadata = {
                "type": "txt",
                "chunk_index": i,
                "source": Path(file_path).name,
                "population": population
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _get_population_by_index(self, index: int) -> str:
        """Assign population based on index"""
        if index < 1:
            return "מוסד"
        elif index < 20:
            return "רשות"
        else:
            return "מחז"