"""
Clean Vertex AI Search Implementation
Replaces Vertex AI RAG Engine to enable proper metadata filtering
"""

from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage
import json
import logging
import time
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexAISearchManager:
    """Manager class for Vertex AI Search operations"""
    
    def __init__(self, project_id: str, location: str = "global"):
        self.project_id = project_id
        self.location = location
        self.parent = f"projects/{project_id}/locations/{location}/collections/default_collection"
        
        # Initialize clients
        try:
            self.datastore_client = discoveryengine.DataStoreServiceClient()
            self.engine_client = discoveryengine.EngineServiceClient()
            self.document_client = discoveryengine.DocumentServiceClient()
            self.search_client = discoveryengine.SearchServiceClient()
            self.generation_client = discoveryengine.GroundedGenerationServiceClient()
            logger.info("Successfully initialized Vertex AI Search clients")
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise

    def datastore_exists(self, datastore_id: str) -> bool:
        """Check if datastore already exists"""
        try:
            datastore_name = f"{self.parent}/dataStores/{datastore_id}"
            self.datastore_client.get_data_store(name=datastore_name)
            logger.info(f"Datastore {datastore_id} already exists")
            return True
        except Exception:
            logger.info(f"Datastore {datastore_id} does not exist")
            return False

    def engine_exists(self, engine_id: str) -> bool:
        """Check if engine already exists"""
        try:
            engine_name = f"{self.parent}/engines/{engine_id}"
            self.engine_client.get_engine(name=engine_name)
            logger.info(f"Engine {engine_id} already exists")
            return True
        except Exception:
            logger.info(f"Engine {engine_id} does not exist")
            return False

  
    def import_documents_to_datastore(self, datastore_id: str, gcs_uri: str) -> bool:
        """Import documents from GCS to Vertex AI Search datastore"""
        try:
            logger.info(f"Importing documents from {gcs_uri} to datastore {datastore_id}")
            
            parent = f"{self.parent}/dataStores/{datastore_id}/branches/default_branch"
            
            request = discoveryengine.ImportDocumentsRequest(
                parent=parent,
                gcs_source=discoveryengine.GcsSource(
                    input_uris=[gcs_uri],
                    data_schema="content"
                ),
                reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
            )
            
            operation = self.document_client.import_documents(request)
            
            # Fixed: Use operation metadata or just log operation type
            logger.info(f"Import operation started: {type(operation).__name__}")
            
            # Wait for completion with progress updates
            start_time = time.time()
            while not operation.done():
                elapsed = time.time() - start_time
                logger.info(f"Import in progress... ({elapsed:.0f}s elapsed)")
                time.sleep(30)  # Check every 30 seconds
                
                if elapsed > 1800:  # 30 minute timeout
                    logger.error("Import operation timeout")
                    return False
            
            result = operation.result()
            logger.info(f"Import completed successfully: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import documents: {e}")
            return False
  

    def create_datastore(self, datastore_id: str, display_name: str) -> str:
        """Create Vertex AI Search data store or return existing one"""
        try:
            datastore_name = f"{self.parent}/dataStores/{datastore_id}"
            
            # Check if already exists
            if self.datastore_exists(datastore_id):
                logger.info(f"Using existing datastore: {datastore_id}")
                return datastore_name
            
            logger.info(f"Creating new datastore: {datastore_id}")
            
            data_store = discoveryengine.DataStore(
                display_name=display_name,
                industry_vertical=discoveryengine.IndustryVertical.GENERIC,
                solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
                content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
            )
            
            request = discoveryengine.CreateDataStoreRequest(
                parent=self.parent,
                data_store=data_store,
                data_store_id=datastore_id,
            )
            
            operation = self.datastore_client.create_data_store(request)
            logger.info(f"Datastore creation started: {type(operation).__name__}")
            
            # Wait for completion
            result = operation.result(timeout=300)
            logger.info(f"Datastore created successfully: {result.name}")
            return result.name
            
        except Exception as e:
            logger.error(f"Failed to create datastore {datastore_id}: {e}")
            raise

    def create_search_engine(self, engine_id: str, datastore_id: str, display_name: str) -> str:
        """Create search engine/app or return existing one"""
        try:
            engine_name = f"{self.parent}/engines/{engine_id}"
            
            # Check if already exists
            if self.engine_exists(engine_id):
                logger.info(f"Using existing engine: {engine_id}")
                return engine_name
            
            logger.info(f"Creating new search engine: {engine_id}")
            
            engine = discoveryengine.Engine(
                display_name=display_name,
                solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
                data_store_ids=[datastore_id],
                search_engine_config=discoveryengine.Engine.SearchEngineConfig(
                    search_tier=discoveryengine.SearchTier.SEARCH_TIER_STANDARD,
                ),
            )
            
            request = discoveryengine.CreateEngineRequest(
                parent=self.parent,
                engine=engine,
                engine_id=engine_id,
            )
            
            operation = self.engine_client.create_engine(request)
            logger.info(f"Search engine creation started: {type(operation).__name__}")
            
            result = operation.result(timeout=300)
            logger.info(f"Search engine created successfully: {result.name}")
            return result.name
            
        except Exception as e:
            logger.error(f"Failed to create search engine {engine_id}: {e}")
            raise

    def setup_complete_infrastructure(self, base_name: str) -> tuple[str, str]:
        """Create both datastore and search engine (or use existing ones)"""
        try:
            datastore_id = f"{base_name}-datastore"
            engine_id = f"{base_name}-engine"
            
            logger.info("Starting complete infrastructure setup")
            logger.info(f"Target datastore ID: {datastore_id}")
            logger.info(f"Target engine ID: {engine_id}")
            
            # Create or get existing data store
            datastore_name = self.create_datastore(
                datastore_id=datastore_id,
                display_name=f"{base_name.title()} Data Store"
            )
            
            # Create or get existing search engine
            engine_name = self.create_search_engine(
                engine_id=engine_id,
                datastore_id=datastore_id,
                display_name=f"{base_name.title()} Search Engine"
            )
            
            logger.info(f"Infrastructure setup complete. DataStore: {datastore_id}, Engine: {engine_id}")
            return datastore_id, engine_id
            
        except Exception as e:
            logger.error(f"Failed to setup infrastructure: {e}")
            raise

    def convert_documents_to_search_format(self, documents, output_file: str) -> bool:
        """Convert Document objects to Vertex AI Search JSONL format"""
        try:
            logger.info(f"Converting {len(documents)} documents to search format")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, doc in enumerate(documents, 1):
                    try:
                        # Create document in Vertex AI Search format
                        search_doc = {
                            "id": f"physics_doc_{i}",
                            "structData": {
                                "population": str(doc.metadata.get("population", "")),
                                "type": str(doc.metadata.get("type", "")),
                                "chunk_index": str(doc.metadata.get("chunk_index", i)),
                                "source": str(doc.metadata.get("source", "")),
                                "code_maane": str(doc.metadata.get("code_maane", ""))
                            },
                            "content": {
                                "mimeType": "text/plain",
                                "representation": "CONTENT_TEXT",
                                "textContent": doc.page_content
                            }
                        }
                        
                        f.write(json.dumps(search_doc, ensure_ascii=False) + '\n')
                        
                    except Exception as e:
                        logger.warning(f"Failed to process document {i}: {e}")
                        continue
            
            logger.info(f"Successfully created search JSONL file: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert documents: {e}")
            return False

    def upload_to_gcs(self, local_file: str, bucket_name: str, blob_name: str) -> str:
        """Upload file to Google Cloud Storage"""
        try:
            logger.info(f"Uploading {local_file} to gs://{bucket_name}/{blob_name}")
            
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(local_file)
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            
            logger.info(f"Successfully uploaded to: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise

  
    def search_with_filters(self, engine_id: str, query: str, filters: dict = None, max_results: int = 10) -> list:
        """Search documents with optional metadata filters"""
        try:
            logger.info(f"Searching for: '{query}' with filters: {filters}")
            
            # Build filter query
            filter_parts = []
            if filters:
                for field, value in filters.items():
                    if value:  # Only add non-empty values
                        filter_parts.append(f'{field}: ANY("{value}")')
            
            filter_query = " AND ".join(filter_parts) if filter_parts else ""
            logger.info(f"Filter query: '{filter_query}'")
            
            serving_config = f"{self.parent}/engines/{engine_id}/servingConfigs/default_search"
            
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                filter=filter_query,
                page_size=max_results,
            )
            
            response = self.search_client.search(request)
            
            results = []
            for result in response.results:
                try:
                    doc_data = {
                        'id': result.id,
                        'content': '',
                        'metadata': {},
                        'score': 0
                    }
                    
                    # Extract metadata from structData
                    if hasattr(result, 'document') and result.document:
                        if hasattr(result.document, 'struct_data') and result.document.struct_data:
                            doc_data['metadata'] = dict(result.document.struct_data)
                        
                        # Extract content
                        if hasattr(result.document, 'derived_struct_data'):
                            derived = result.document.derived_struct_data
                            if 'extractive_answers' in derived and derived['extractive_answers']:
                                doc_data['content'] = derived['extractive_answers'][0].get('content', '')
                            elif 'snippets' in derived and derived['snippets']:
                                doc_data['content'] = derived['snippets'][0].get('snippet', '')
                    
                    # Get score
                    if hasattr(result, 'model_scores') and result.model_scores:
                        doc_data['score'] = result.model_scores.get('quality_score', 0)
                    
                    results.append(doc_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def generate_grounded_response(self, engine_id: str, query: str, filters: dict = None, max_context: int = 5) -> str:
        """Generate grounded response using search results"""
        try:
            logger.info(f"Generating grounded response for: '{query}' with filters: {filters}")
            
            # Build filter query
            filter_parts = []
            if filters:
                for field, value in filters.items():
                    if value:
                        filter_parts.append(f'{field}: ANY("{value}")')
            
            filter_query = " AND ".join(filter_parts) if filter_parts else ""
            serving_config = f"{self.parent}/engines/{engine_id}/servingConfigs/default_search"
            
            request = discoveryengine.GenerateGroundedContentRequest(
                location=f"projects/{self.project_id}/locations/{self.location}",
                
                generation_spec=discoveryengine.GenerateGroundedContentRequest.GenerationSpec(
                    model_id="gemini-1.5-flash-001",
                ),
                
                contents=[
                    discoveryengine.GroundedGenerationContent(
                        role="user",
                        parts=[
                            discoveryengine.GroundedGenerationContent.Part(text=query)
                        ],
                    )
                ],
                
                grounding_spec=discoveryengine.GenerateGroundedContentRequest.GroundingSpec(
                    grounding_sources=[
                        discoveryengine.GenerateGroundedContentRequest.GroundingSource(
                            search_source=discoveryengine.GenerateGroundedContentRequest.GroundingSource.SearchSource(
                                serving_config=serving_config,
                                filter=filter_query,
                                max_result_count=max_context,
                            ),
                        ),
                    ]
                ),
            )
            
            response = self.generation_client.generate_grounded_content(request)
            
            if response.candidates and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
                logger.info("Successfully generated grounded response")
                return generated_text
            else:
                logger.warning("No content generated in response")
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Failed to generate grounded response: {e}")
            return f"Error generating response: {str(e)}"

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
  

# Usage example
def main():
    """Example usage of VertexAI Search"""
    try:
        PROJECT_ID = "gapmaanim"
        BUCKET_NAME = "gap-jsonl-vertexsearch-bucket"
        data_file = "files/data.txt"

        # Initialize manager
        search_manager = VertexAISearchManager(PROJECT_ID)
        
        # Step 1: Setup infrastructure
        logger.info("Setting up Vertex AI Search infrastructure")
        datastore_id, engine_id = search_manager.setup_complete_infrastructure("my-docs")
        
        # Step 2: Convert and upload documents
        logger.info("Converting and uploading documents")
        
        # Load your documents (using your existing function)
        documents = load_documents(data_file)
        
        # Convert to search format
        success = search_manager.convert_documents_to_search_format(
            documents, 
            "search_documents.jsonl"
        )
        
        if not success:
            logger.error("Failed to convert documents")
            return
        
        # Upload to GCS
        gcs_uri = search_manager.upload_to_gcs(
            "search_documents.jsonl",
            BUCKET_NAME,
            "search_documents.jsonl"
        )
        
        # Import to datastore
        import_success = search_manager.import_documents_to_datastore(datastore_id, gcs_uri)
        if not import_success:
            logger.error("Failed to import documents")
            return
        
        # Step 3: Test searching with filters
        logger.info("Testing search functionality")
        
        # Search with population filter
        results = search_manager.search_with_filters(
            engine_id=engine_id,
            query="פיזיקה קוונטית",
            filters={"population": "מוסד"},
            max_results=5
        )
        
        print(f"Found {len(results)} results for מוסד:")
        for i, result in enumerate(results, 1):
            print(f"{i}. ID: {result['id']}")
            print(f"   Population: {result['metadata'].get('population', 'Unknown')}")
            print(f"   Content: {result['content'][:100]}...")
            print()
        
        # Test grounded generation
        logger.info("Testing grounded generation")
        response = search_manager.generate_grounded_response(
            engine_id=engine_id,
            query="מהם עקרונות היסוד של פיזיקה קוונטית?",
            filters={"population": "מוסד"}
        )
        
        print(f"Grounded Response:\n{response}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()