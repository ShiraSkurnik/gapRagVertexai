# Vertex AI RAG Engine

import os
import json
import logging
from typing import List, Dict, Any
import tempfile
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import time
from google.cloud import storage
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_vertex_ai(project_id: str, region: str):
    """Initialize Vertex AI with the given project and region."""
    try:
        logger.info(f"Initializing Vertex AI with project: {project_id}, region: {region}")
        vertexai.init(project=project_id, location=region)
        logger.info("Vertex AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {str(e)}")
        print(f"Error initializing Vertex AI: {str(e)}")
        raise

def create_embedding_model_config():
    """Create RAG embedding model configuration."""
    try:
        logger.info("Creating RAG embedding model configuration")
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-multilingual-embedding-002"
            )
        )
        logger.info("RAG embedding model configuration created successfully")
        return embedding_model_config
    except Exception as e:
        logger.error(f"Failed to create embedding model configuration: {str(e)}")
        print(f"Error creating embedding model configuration: {str(e)}")
        raise

def get_or_create_rag_corpus(display_name: str, embedding_model_config):
    """Check if RAG corpus exists, create new one if it doesn't."""
    try:
        logger.info(f"Checking if RAG corpus with display name '{display_name}' already exists")
        # Check if corpus already exists
        existing_corpora = rag.list_corpora()
        rag_corpus = None
        
        for corpus in existing_corpora:
            if corpus.display_name == display_name:
                rag_corpus = corpus
                logger.info(f"Found existing RAG corpus: {rag_corpus.name}")
                print(f"Using existing RAG corpus: {display_name}")
                break
        
        # Create new corpus if it doesn't exist
        if rag_corpus is None:
            logger.info(f"Creating new RAG corpus with display name: {display_name}")
            rag_corpus = rag.create_corpus(
                display_name=display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )
            logger.info(f"RAG corpus created successfully with name: {rag_corpus.name}")
            print(f"Created new RAG corpus: {display_name}")
            
        return rag_corpus
        
    except Exception as e:
        logger.error(f"Failed to check/create RAG corpus: {str(e)}")
        print(f"Error checking/creating RAG corpus: {str(e)}")
        raise

def load_documents(file_path: str) -> List[Document]:
    """Load file and convert each item to a Document object."""
    try:
        file_ext = Path(file_path).suffix.lower()
        documents = []

        if file_ext == '.json':
            logger.info(f"Loading JSON file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                content = f"שם מענה: {item['שם_מענה']}\nקוד מענה: {item['קוד_מענה']}\nתקציבים: {', '.join(item['תקציבים'])}\nאוכלוסיה: {item['אוכלוסיה']}"
                print(content)
                
                metadata = {
                    "population": str(item.get("אוכלוסיה", "")),
                    "type": "json",
                    "code_maane": str(item.get("קוד_מענה", "")),
                    "index": i,
                    "source": file_path  # Fixed: was using undefined json_file_path
                }
                
                # Create Document object for consistency
                documents.append(Document(page_content=content, metadata=metadata))
        
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                if i < 2:
                    population = "student"
                elif i < 4:
                    population = "teacher"
                else:
                    population = "principle"  
                    
                metadata = {
                    "type": "txt",
                    "chunk_index": i,
                    "source": Path(file_path).name,
                    "population": population
                }   
                documents.append(Document(page_content=chunk, metadata=metadata))
                
        logger.info(f"Loaded {len(documents)} documents from file")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {str(e)}")
        print(f"Error loading file: {str(e)}")
        raise
        

def import_json_to_corpus(corpus_name: str, json_file_path: str):
    """Import JSON data to RAG corpus with each item as a separate document."""
    
    def get_existing_files(corpus_name: str) -> set:
        """Get set of existing file names in corpus."""
        try:
            existing_files = rag.list_files(corpus_name=corpus_name)
            existing_names = {file.display_name for file in existing_files}
            print(f"Found {len(existing_names)} existing files in corpus")
            return existing_names
        except Exception as e:
            print(f"Warning: Could not list existing files: {e}")
            return set()
    
    def upload_document(temp_dir: str, doc: Document, doc_index: int, display_name: str) -> bool:
        """Upload a single document to the corpus. Returns True if successful."""
        try:
            # Create temporary file (required by RAG API that only accepts file paths)
            temp_file_path = os.path.join(temp_dir, f"{display_name}.txt")
            
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)
            
            # Upload to RAG corpus
            rag.upload_file(
                corpus_name=corpus_name,
                path=temp_file_path,
                display_name=display_name
            )
            
            print(f"Uploaded document {doc_index}: {display_name}")
            print(f"++ content: {doc.page_content}")
            print(f"++ metadata: {doc.metadata}")
            return True
            
        except Exception as e:
            print(f"Failed to upload document {doc_index}: {e}")
            return False
    
    try:
        print(f"Starting import to corpus: {corpus_name}")
        
        # Load documents and get existing files
        documents = load_documents(json_file_path)
        total_docs = len(documents)
        existing_names = get_existing_files(corpus_name)
        
        print(f"Loaded {total_docs} documents from JSON file")
        
        # Process documents
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_count = 0
            skipped_count = 0
            
            for i, doc in enumerate(documents, 1):
                display_name = f"json_doc_{i}"
                
                # Skip if document already exists
                if display_name in existing_names:
                    print(f"Skipping document {i}/{total_docs} - already exists: {display_name}")
                    skipped_count += 1
                    continue
                
                # Upload document
                if upload_document(temp_dir, doc, i, display_name):
                    uploaded_count += 1
        
        print(f"Import completed - Uploaded: {uploaded_count}, Skipped: {skipped_count}, Total: {total_docs}")
        return uploaded_count
        
    except Exception as e:
        print(f"Failed to import JSON to corpus: {e}")
        raise


def import_json_to_corpus_jsonl(corpus_name: str, json_file_path: str, gcs_bucket: str):
    """
    Generate a .jsonl ingestion file for Vertex AI RAG corpus.
    Each item in the JSON input becomes a separate document with its own metadata.
    Checks if corpus already has the JSONL file to avoid duplicate work.
    """
    
    def get_gcs_paths(corpus_name: str, gcs_bucket: str) -> tuple[str, str]:
        """Generate GCS blob name and URI for the corpus."""
        safe_name = os.path.basename(corpus_name)
        blob_name = f"rag_ingestions/{safe_name}_ingestion.jsonl"
        gcs_uri = f"gs://{gcs_bucket}/{blob_name}"
        return blob_name, gcs_uri
    
    def check_file_already_imported(corpus_name: str, gcs_uri: str, blob_name: str) -> bool:
        """Check if file is already imported to the corpus."""
        try:
            corpus_files = rag.list_files(corpus_name=corpus_name)
            return any(
                gcs_uri in str(file_info) or blob_name in str(file_info)
                for file_info in corpus_files
            )
        except Exception as e:
            print(f"Could not check corpus files: {e}")
            return False
    
    def import_existing_file(corpus_name: str, gcs_uri: str, documents: List[Document]) -> int:
        """Import existing GCS file to corpus and return document count."""
        rag.import_files(corpus_name=corpus_name, paths=[gcs_uri])
        total_docs = len(documents)
        print(f"Import completed – Used existing JSONL with {total_docs} documents.")
        return total_docs
    
    def create_and_upload_jsonl(documents: List[Document], blob_name: str, gcs_bucket: str, corpus_name: str) -> str:
        """Create JSONL file, upload to GCS, and return GCS URI."""
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(gcs_bucket)
        safe_name = os.path.basename(corpus_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, f"{safe_name}_ingestion.jsonl")
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i, doc in enumerate(documents, 1):
                    record = {
                        "id": f"json_doc_{i}",
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"Created local JSONL ingestion file: {jsonl_path}")
            
            # Upload to GCS
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(jsonl_path)
            
            gcs_uri = f"gs://{gcs_bucket}/{blob_name}"
            print(f"Uploaded JSONL to {gcs_uri}")
            return gcs_uri
    
    try:
        print(f"Starting JSONL ingestion preparation for corpus: {corpus_name}")
        
        # Load documents first (needed in all code paths)
        documents = load_documents(json_file_path)
        total_docs = len(documents)
        print(f"Loaded {total_docs} documents from JSON file")
        
        # Get GCS paths
        blob_name, gcs_uri = get_gcs_paths(corpus_name, gcs_bucket)
        
        # Check if file exists in GCS
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(gcs_bucket)
        blob = bucket.blob(blob_name)
        
        if blob.exists():
            print(f"JSONL file already exists at {gcs_uri}")
            
            # Check if already imported to corpus
            if check_file_already_imported(corpus_name, gcs_uri, blob_name):
                print(f"File {gcs_uri} is already imported to corpus {corpus_name}")
                print("Skipping import - no action needed")
                return total_docs
            else:
                print(f"File exists in GCS but not imported to corpus. Importing {gcs_uri}")
                return import_existing_file(corpus_name, gcs_uri, documents)
        
        # File doesn't exist - create new one
        print(f"JSONL file not found in GCS. Creating new ingestion file.")
        gcs_uri = create_and_upload_jsonl(documents, blob_name, gcs_bucket, corpus_name)
        
        # Import into Vertex AI RAG
        rag.import_files(corpus_name=corpus_name, paths=[gcs_uri])
        
        print(f"Import completed – Prepared {total_docs} documents in JSONL format.")
        return total_docs
        
    except Exception as e:
        print(f"Failed to import JSON to corpus: {e}")
        raise
    
def import_new_files_to_corpus(corpus_name: str, file_paths: list, chunk_size: int = 512, chunk_overlap: int = 100):
    """Import files to the RAG corpus if they don't already exist."""
    try:
        # Check existing files first - convert pager to list
        existing_files_pager = rag.list_files(corpus_name=corpus_name)
        existing_files = list(existing_files_pager)
        
        # Print existing files information
        print(f"\n📁 Found {len(existing_files)} existing files in corpus:")
        existing_sources = set()
        
        if existing_files:
            for i, file in enumerate(existing_files, 1):
                print(f"  {i}. {file.display_name} (ID: {file.name})")
                
                # Get source information from the correct attributes
                source_info = "Unknown source"
                file_source = None
                
                try:
                    if hasattr(file, 'google_drive_source') and file.google_drive_source:
                        # Extract the file ID from google_drive_source for comparison
                        file_source = str(file.google_drive_source)
                        source_info = f"Google Drive: {file_source}"
                        existing_sources.add(file_source)
                    elif hasattr(file, 'gcs_source') and file.gcs_source:
                        file_source = str(file.gcs_source)
                        source_info = f"GCS: {file_source}"
                        existing_sources.add(file_source)
                    elif hasattr(file, 'direct_upload_source') and file.direct_upload_source:
                        file_source = str(file.direct_upload_source)
                        source_info = f"Direct upload: {file_source}"
                        existing_sources.add(file_source)
                except Exception as e:
                    logger.warning(f"Could not extract source from file {file.display_name}: {e}")
                
                print(f"     📎 {source_info}")
                if hasattr(file, 'create_time'):
                    print(f"     ⏰ Created: {file.create_time}")
        else:
            print("  (No existing files)")
        
        # Show what we're trying to import
        print(f"\n📥 Files we want to import:")
        for i, path in enumerate(file_paths, 1):
            print(f"  {i}. {path}")
        
        # Check for duplicates by comparing Google Drive file IDs
        files_to_import = []
        for path in file_paths:
            is_duplicate = False
            
            # Extract file ID from Google Drive URL
            if 'drive.google.com' in path and '/d/' in path:
                try:
                    file_id = path.split('/d/')[1].split('/')[0]
                    print(f"  🔍 Extracted file ID: {file_id} from {path}")
                    
                    # Check if this file ID exists in any of the existing sources
                    for existing_source in existing_sources:
                        if file_id in existing_source:
                            print(f"  ⚠️  Duplicate found! File ID {file_id} already exists")
                            is_duplicate = True
                            break
                except Exception as e:
                    logger.warning(f"Could not extract file ID from {path}: {e}")
            
            if not is_duplicate:
                files_to_import.append(path)
        
        print(f"\n📊 Summary:")
        print(f"  • Total files requested: {len(file_paths)}")
        print(f"  • Already in corpus: {len(file_paths) - len(files_to_import)}")
        print(f"  • New files to import: {len(files_to_import)}")
        
        if not files_to_import:
            logger.info("All files already exist in corpus, skipping import")
            print("✅ All files already imported, skipping...")
            return
        
        logger.info(f"Importing {len(files_to_import)} new files to RAG corpus")
        print(f"🚀 Starting import of {len(files_to_import)} new files...")
        
        # Continue with the actual import logic...
        rag.import_files(
            corpus_name,
            files_to_import,
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
            ),
            max_embedding_requests_per_min=1000,
        )
        logger.info("New files imported successfully to RAG corpus")
        print("✅ Import completed!")

    except Exception as e:
        logger.error(f"Failed to import files to RAG corpus: {str(e)}")
        print(f"Error importing files: {str(e)}")
        raise
    
def check_corpus_status(corpus_name: str, max_wait_time: int = 300, check_interval: int = 30):
    """Check if corpus indexing is complete before querying."""
    try:
        logger.info(f"Checking corpus status for: {corpus_name}")
        print(f"🔍 Checking corpus indexing status...")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get corpus information
                corpus = rag.get_corpus(name=corpus_name)
                
                # Check if corpus has a state/status attribute
                if hasattr(corpus, 'state'):
                    corpus_state = corpus.state
                    logger.info(f"Corpus state: {corpus_state}")
                    print(f"📊 Corpus state: {corpus_state}")
                    
                    # Check for various possible state values that indicate readiness
                    ready_states = ['ACTIVE', 'READY', 'AVAILABLE', 'COMPLETED']
                    if any(state in str(corpus_state).upper() for state in ready_states):
                        logger.info("Corpus is ready for querying")
                        print("✅ Corpus indexing is complete and ready for querying!")
                        return True
                        
                    # Check for states that indicate processing
                    processing_states = ['CREATING', 'UPDATING', 'INDEXING', 'PROCESSING', 'PENDING']
                    if any(state in str(corpus_state).upper() for state in processing_states):
                        logger.info(f"Corpus is still processing. Waiting {check_interval} seconds...")
                        print(f"⏳ Corpus is still indexing (state: {corpus_state}). Waiting {check_interval} seconds...")
                        time.sleep(check_interval)
                        continue
                
                # If no state attribute, try checking files and their status
                try:
                    files = list(rag.list_files(corpus_name=corpus_name))
                    if files:
                        # Check if files have status information
                        all_ready = True
                        for file in files:
                            if hasattr(file, 'state'):
                                file_state = str(file.state).upper()
                                if any(state in file_state for state in ['CREATING', 'UPDATING', 'INDEXING', 'PROCESSING', 'PENDING']):
                                    all_ready = False
                                    break
                        
                        if all_ready:
                            logger.info("All files in corpus appear to be ready")
                            print("✅ All files in corpus are indexed and ready!")
                            return True
                        else:
                            logger.info("Some files are still being processed")
                            print(f"⏳ Some files are still being indexed. Waiting {check_interval} seconds...")
                            time.sleep(check_interval)
                            continue
                    else:
                        logger.warning("No files found in corpus")
                        print("⚠️ No files found in corpus")
                        return False
                        
                except Exception as file_check_error:
                    logger.warning(f"Could not check file status: {file_check_error}")
                    # If we can't check file status, assume corpus is ready after a reasonable wait
                    logger.info("Cannot determine file status, assuming corpus is ready")
                    print("✅ Assuming corpus is ready (status check unavailable)")
                    return True
                
                # If we can't determine state, wait and try again
                logger.info(f"Cannot determine corpus state, waiting {check_interval} seconds...")
                print(f"⏳ Cannot determine corpus state, waiting {check_interval} seconds...")
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error checking corpus status: {e}")
                print(f"⚠️ Error checking corpus status: {e}")
                # Continue trying until timeout
                time.sleep(check_interval)
        
        # Timeout reached
        elapsed_time = int(time.time() - start_time)
        logger.warning(f"Corpus status check timed out after {elapsed_time} seconds")
        print(f"⚠️ Corpus status check timed out after {elapsed_time} seconds. Proceeding anyway...")
        return False
        
    except Exception as e:
        logger.error(f"Failed to check corpus status: {str(e)}")
        print(f"Error checking corpus status: {str(e)}")
        # Don't raise exception, just warn and continue
        print("⚠️ Continuing with queries despite status check failure...")
        return False
    
def create_rag_retrieval_config(top_k: int = 3, similarity_threshold: float = 0.5, metadataFilterField: str = "", metadataFilterValue: str = ""):
    """Create RAG retrieval configuration."""
    try:
        logger.info("Setting up RAG retrieval configuration")
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k = top_k,  # Optional
            filter = rag.Filter(
                # Vector similarity threshold
                vector_similarity_threshold = similarity_threshold,  # Optional
                # Metadata filtering (limited options)
                metadata_filter = f'metadata.{metadataFilterField} == {metadataFilterValue}'
            )
        )


        logger.info("RAG retrieval configuration created successfully")
        return rag_retrieval_config
    except Exception as e:
        logger.error(f"Failed to create RAG retrieval configuration: {str(e)}")
        print(f"Error creating RAG retrieval configuration: {str(e)}")
        raise

def perform_direct_retrieval(rag_corpus, query_text: str, rag_retrieval_config):
    """Perform direct context retrieval."""
    try:
        logger.info("Performing direct context retrieval")
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,
                    # Optional: supply IDs from `rag.list_files()`.
                     rag_file_ids=["json_doc_1", "json_doc_5"],
                )
            ],
            text=query_text,
            rag_retrieval_config=rag_retrieval_config,
        )
        logger.info("Direct context retrieval completed successfully")
        return response
    except Exception as e:
        logger.error(f"Failed to perform direct context retrieval: {str(e)}")
        print(f"Error in direct context retrieval: {str(e)}")
        raise

def get_actual_rag_file_ids(corpus_name: str):
    """Get the real file IDs that Vertex AI RAG Engine created"""
    try:
        # List all files in the corpus to see their actual IDs
        files_pager = rag.list_files(corpus_name=corpus_name)
        
        # Convert pager to list
        files_list = list(files_pager)
        
        print(f"Found {len(files_list)} files in corpus:")
        actual_file_ids = []
        
        for i, file_info in enumerate(files_list):
            print(f"\n--- File {i+1} ---")
            print(f"Full Resource Name: {file_info.name}")
            print(f"Display Name: {file_info.display_name}")
            
            # Check if these attributes exist
            if hasattr(file_info, 'size_bytes'):
                print(f"Size: {file_info.size_bytes} bytes")
            if hasattr(file_info, 'create_time'):
                print(f"Create Time: {file_info.create_time}")
            
            # Extract just the file ID part (last component of the name)
            file_id = file_info.name.split('/')[-1]
            actual_file_ids.append(file_id)
            print(f"Usable File ID: {file_id}")
            
            # Print all available attributes for debugging
            print(f"Available attributes: {dir(file_info)}")
        
        print(f"\nAll usable file IDs: {actual_file_ids}")
        return actual_file_ids
        
    except Exception as e:
        print(f"Error listing files: {e}")
        return []
    
def create_rag_retrieval_tool(rag_corpus, rag_retrieval_config):
    """Create a RAG retrieval tool."""
    try:
        logger.info("Creating RAG retrieval tool")
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[
                        rag.RagResource(
                            rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                            # Optional: supply IDs from `rag.list_files()`.
                            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                        )
                    ],
                    rag_retrieval_config=rag_retrieval_config,
                ),
            )
        )
        logger.info("RAG retrieval tool created successfully")
        return rag_retrieval_tool
    except Exception as e:
        logger.error(f"Failed to create RAG retrieval tool: {str(e)}")
        print(f"Error creating RAG retrieval tool: {str(e)}")
        raise

def create_rag_model(model_name: str, rag_retrieval_tool, system_prompt: str):
    """Create a Gemini model instance with RAG retrieval tool."""
    try:
        logger.info("Creating Gemini model instance")
        rag_model = GenerativeModel(
            model_name=model_name, tools=[rag_retrieval_tool], system_instruction=system_prompt
        )
        logger.info("Gemini model instance created successfully")
        return rag_model
    except Exception as e:
        logger.error(f"Failed to create Gemini model instance: {str(e)}")
        print(f"Error creating Gemini model: {str(e)}")
        raise

def generate_rag_content(rag_model, query_text: str):
    """Generate content using RAG-enhanced model."""
    try:
        logger.info("Generating content with RAG-enhanced model")
        response = rag_model.generate_content(query_text)
        logger.info("Content generated successfully")
        return response
    except Exception as e:
        logger.error(f"Failed to generate content: {str(e)}")
        print(f"Error generating content: {str(e)}")
        raise

def main():
    """Main function to orchestrate the RAG workflow."""
    # Configuration
    REGION = "europe-west4"
    PROJECT_ID = "gapmaanim"
    display_name = "json_corpus"    
    # data_file = "files/data.json"
    data_file = "files/data.txt"
    # json_file_path = "https://drive.google.com/file/d/1zQJ6Kx1XmnD9tBQJT9KKsqQ5kymR0aqD/view?usp=drive_link"
    # paths = ["https://drive.google.com/file/d/1lIU-DKYKOAbd6vslKP6xFVZiyPGIHQgX/view?usp=sharing"] 
    model_name = "gemini-2.0-flash-lite"
    #query_text = "מחפשת מענה בשם: ימי שיא, שנמצא תחת אוכלוסיה: מחוז, population: מחוז" # ענה תקין החזיר מענה של מחוז 
    #query_text = "מחפשת מענה בשם: ימי שיא, שנמצא תחת אוכלוסיה: מוסד, population: מוסד" # ענה תקין שלא נמצאו תוצאות עבור חיפוש של מוסד 
    #query_text = "מחפשת מענה בשם: ימי שיא, אוכלוסיה: מוסד, population: מוסד" #לא ענה תקין, הביא תוצאה של מחוז
    #query_text = "מחפשת מענה בשם: ימי שיא"
    #query_text = "מחפשת מענה שנמצא תחת אוכלוסיה: מוסד"
    query_text = "מה זה אפקט הול"
    gcs_bucket = "gap-jsonl-rag-bucket"

    # query_text = "מה זה PVD?"
    
    # Initialize Vertex AI
    initialize_vertex_ai(PROJECT_ID, REGION)
    
    # Create embedding model configuration
    embedding_model_config = create_embedding_model_config()
    
    # Get or create RAG corpus
    rag_corpus = get_or_create_rag_corpus(display_name, embedding_model_config)
    
    # Import JSON file to corpus (each item as separate document)
    uploaded_count  = import_json_to_corpus_jsonl(rag_corpus.name, data_file, gcs_bucket)
    
    # Import PDF files to corpus
    # import_new_files_to_corpus(rag_corpus.name, paths)
        # Check corpus status before querying (NEW ADDITION)
    print("\n" + "="*60)
    print("🔍 CHECKING CORPUS INDEXING STATUS")
    print("="*60)
    check_corpus_status(rag_corpus.name, max_wait_time=600, check_interval=3)
    print("="*60 + "\n")

    get_actual_rag_file_ids(rag_corpus.name)

    # Create RAG retrieval configuration
    rag_retrieval_config = create_rag_retrieval_config(3,0.4,"population","principle")
    
    # Perform direct context retrieval
    direct_response = perform_direct_retrieval(rag_corpus, query_text, rag_retrieval_config)
    print("Direct retrieval response:")
    print(direct_response)
    
    # Create RAG retrieval tool
    rag_retrieval_tool = create_rag_retrieval_tool(rag_corpus, rag_retrieval_config)
    
    system_prompt = """
        אתה מומחה במציאת מענים בעברית לפי שאילתת המשתמש.
        תענה תמיד בעברית.
        כאשר המשתמש שואל על מענה ספציפי, תן עדיפות לתוצאות עם התאמה מדויקת של השם.
        תמיד הצג בבירור: שם המענה, קוד המענה, התקציבים והאוכלוסיה.
        אם המשתמש מחפש מענה בשם מסוים, חפש בדיוק את אותו שם.

        - השתמש אך ורק במידע מהמסמכים המצורפים
        - אל תמציא מידע שלא קיים במסמכים
        - ענה קונקרטי לפי המידע שיש ברשותך, אל תתן הסבר או פירוט שלא קיים במידע
        - אסור להמליץ או להעדיף מענה אחד על פני השני!! אלא אך ורק למצוא את המענה המתאים ביותר לצורך המשתמש 
        - אם השאילתא לא ממקדת למענה מסוים, אלא מתאימה לרוב המענים, הסבר את זה למשתמש ותתן סתם כמה מענים ראשונים
        - אם יש כמה פריטים מתאימים - החזר כמה שיותר - ועד חמש פריטים

        **תקפיד לבדוק את האוכלוסיה של המענה כך שתתאים בדיוק לmetadata שהוזנה בשאילתא תחת השם population.**
        
        **אם אין מידע מתאים:** 
        "לא מצאתי מענים מתאימים לשאלתך. אנא דייק את החיפוש."
        """    

    # Create RAG model
    rag_model = create_rag_model(model_name, rag_retrieval_tool, system_prompt)
    
    # Generate enhanced response
    enhanced_response = generate_rag_content(rag_model, query_text)
    print("Enhanced RAG response:")
    print(enhanced_response.text)

if __name__ == "__main__":
    main()