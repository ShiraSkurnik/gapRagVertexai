import os
import json
import logging
from typing import List, Dict, Any
import tempfile
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

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
                publisher_model="publishers/google/models/text-embedding-005"
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

def load_json_as_documents(json_file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and convert each item to a document structure."""
    try:
        logger.info(f"Loading JSON file: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        for i, item in enumerate(data):
            # Determine population based on index (as per your logic)
            if i < 1:
                population = "מוסד"
            elif i < 20:
                population = "רשות"
            else:
                population = "מחז"
            
            # Convert item to JSON string for content
            content = json.dumps(item, indent=2, ensure_ascii=False)
            
            # Create metadata
            metadata = {
                "population": population,
                "type": "json",
                "code_maane": str(item.get("קוד_מענה", "")),
                "index": i,
                "source": json_file_path
            }
            
            documents.append({
                "content": content,
                "metadata": metadata
            })
        
        logger.info(f"Loaded {len(documents)} documents from JSON file")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_file_path}: {str(e)}")
        print(f"Error loading JSON file: {str(e)}")
        raise

def import_json_to_corpus(corpus_name: str, json_file_path: str):
    """Import JSON data to RAG corpus with each item as a separate document."""
    try:
        logger.info(f"Starting import of JSON file to corpus: {corpus_name}")
        
        # Load JSON as documents
        documents = load_json_as_documents(json_file_path)
        
        # Import documents to corpus in batches
        batch_size = 100  # Adjust based on your needs
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_end = min(i + batch_size, total_docs)
            
            logger.info(f"Importing batch {i//batch_size + 1}: documents {i+1} to {batch_end}")
            
            # Prepare RAG files for this batch
            rag_files = []
            for j, doc in enumerate(batch):
                # Create a unique name for each document
                doc_name = f"json_doc_{i+j+1}"
                
                # Create RagFile with content and metadata
                rag_file = rag.RagFile(
                    display_name=doc_name,
                    # Use direct text content
                )
                
                # For Vertex AI RAG, we need to upload the content
                # This approach uses the import_files method
                rag_files.append({
                    "display_name": doc_name,
                    "direct_upload_source": {
                        "mime_type": "text/plain",
                        "contents": doc["content"].encode('utf-8')
                    },
                    "rag_file_parsing_config": {
                        "use_advanced_pdf_parsing": False
                    }
                })
            
            # Import files to corpus
            try:
                response = rag.import_files(
                    corpus_name=corpus_name,
                    paths=[],  # Empty since we're using direct upload
                    chunk_size=512,
                    chunk_overlap=100,
                    max_embedding_requests_per_min=1000,
                    import_files_config=rag_files
                )
                logger.info(f"Successfully imported batch {i//batch_size + 1}")
                
            except Exception as batch_error:
                logger.error(f"Failed to import batch {i//batch_size + 1}: {batch_error}")
                # Try individual import for this batch
                for j, doc in enumerate(batch):
                    try:
                        doc_name = f"json_doc_{i+j+1}"
                        single_file = [{
                            "display_name": doc_name,
                            "direct_upload_source": {
                                "mime_type": "text/plain",
                                "contents": doc["content"].encode('utf-8')
                            }
                        }]
                        
                        rag.import_files(
                            corpus_name=corpus_name,
                            paths=[],
                            import_files_config=single_file
                        )
                        logger.info(f"Successfully imported individual document: {doc_name}")
                        
                    except Exception as doc_error:
                        logger.error(f"Failed to import document {doc_name}: {doc_error}")
        
        logger.info(f"Completed import of {total_docs} documents to corpus")
        
    except Exception as e:
        logger.error(f"Failed to import JSON to corpus: {e}")
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

def create_rag_retrieval_config(top_k: int = 3, distance_threshold: float = 0.5):
    """Create RAG retrieval configuration."""
    try:
        logger.info("Setting up RAG retrieval configuration")
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=top_k,  # Optional
            filter=rag.Filter(vector_distance_threshold=distance_threshold),  # Optional
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
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
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

def create_rag_model(model_name: str, rag_retrieval_tool):
    """Create a Gemini model instance with RAG retrieval tool."""
    try:
        logger.info("Creating Gemini model instance")
        rag_model = GenerativeModel(
            model_name=model_name, tools=[rag_retrieval_tool]
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
    display_name = "test_corpus"    
    # data_file = "files/data.json"
    # json_file_path = "https://drive.google.com/file/d/1zQJ6Kx1XmnD9tBQJT9KKsqQ5kymR0aqD/view?usp=drive_link"
    paths = ["https://drive.google.com/file/d/1lIU-DKYKOAbd6vslKP6xFVZiyPGIHQgX/view?usp=sharing"] 
    model_name = "gemini-2.0-flash-lite"
    # query_text = "מידע על שירותים דיגיטליים"
    query_text = "מה זה PVD?"
    
    # Initialize Vertex AI
    initialize_vertex_ai(PROJECT_ID, REGION)
    
    # Create embedding model configuration
    embedding_model_config = create_embedding_model_config()
    
    # Get or create RAG corpus
    rag_corpus = get_or_create_rag_corpus(display_name, embedding_model_config)
    
    # Import JSON file to corpus (each item as separate document)
    # import_json_to_corpus(rag_corpus.name, data_file)
    
    # Import PDF files to corpus
    import_new_files_to_corpus(rag_corpus.name, paths)
    
    # Create RAG retrieval configuration
    rag_retrieval_config = create_rag_retrieval_config()
    
    # Perform direct context retrieval
    direct_response = perform_direct_retrieval(rag_corpus, query_text, rag_retrieval_config)
    print("Direct retrieval response:")
    print(direct_response)
    
    # Create RAG retrieval tool
    rag_retrieval_tool = create_rag_retrieval_tool(rag_corpus, rag_retrieval_config)
    
    # Create RAG model
    rag_model = create_rag_model(model_name, rag_retrieval_tool)
    
    # Generate enhanced response
    enhanced_response = generate_rag_content(rag_model, query_text)
    print("Enhanced RAG response:")
    print(enhanced_response.text)

if __name__ == "__main__":
    main()