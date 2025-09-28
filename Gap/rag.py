from vertexai import rag
import vertexai
import os
from google.cloud import storage
import json
from google.oauth2 import service_account
from graph_state import GraphState

# הגדרות קבועות
os.environ["http_proxy"] = "http://10.20.14.35:8080"
os.environ["https_proxy"] = "http://10.20.14.35:8080"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rag-service-account-key.json"

PROJECT_ID = "ai-gap-470906"
REGION = "europe-west4"
DISPLAY_NAME = "maanim_gap_corpus"
SERVICE_ACCOUNT_JSON = "rag-service-account-key.json"
FILES_PATH = "files/"
FILE_NAME = "maanim_desc"# "maanim"
BUCKET_NAME = "gap-maanim-bucket"
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON
)

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

def upload_file_to_gcs(file_path: str) -> str:
    try:        
        # Get or create bucket
        try:
            bucket = storage_client.bucket(BUCKET_NAME)
            bucket.reload()  # Check if bucket exists
        except Exception:
            bucket = storage_client.create_bucket(BUCKET_NAME, location="europe-west4")
            print(f"📦 Created new GCS bucket: {BUCKET_NAME}")
        
        # Upload file
        blob = bucket.blob( f"{FILE_NAME}.jsonl")
        blob.upload_from_filename(file_path)
        
        gcs_uri = f"gs://{BUCKET_NAME}/{FILE_NAME}.jsonl"
        print(f"📤 Uploaded to GCS: {gcs_uri}")
        
        return gcs_uri
        
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        raise

def create_rag_corpus(gcs_path):
    """יוצר Corpus חדש ומייבא קובץ JSONL. מחזיר את השם המלא של ה־Corpus"""
    
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-multilingual-embedding-002"
        )
    )

    rag_corpus = rag.create_corpus(
        display_name=DISPLAY_NAME,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )

    print(f"RAG Corpus נוצר: {rag_corpus.name}")

    # ייבוא הקובץ ל־Corpus
    rag.import_files(
        rag_corpus.name,
        [gcs_path],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=0
            ),
        ),
        max_embedding_requests_per_min=1000,
    )

    print("קובץ JSONL יובא - כל שורה היא chunk נפרד")

    return rag_corpus.name

def query_rag(state: GraphState) -> GraphState:
    
    search_query = state["search_query"]
    print(f"start query_rag for {search_query}")
    corpus_name="projects/220128409398/locations/europe-west4/ragCorpora/5685794529555251200"
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=25,
        filter=rag.Filter(vector_distance_threshold=0.9),
    )
    print("after config")
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=search_query,
        rag_retrieval_config=rag_retrieval_config,
    )

    print("תוצאות השאילתה:")
    print(response)
    texts = [ctx.text for ctx in response.contexts.contexts]
    print(texts)
    state["retrieved_docs"]=texts
    return state

def create_rag() -> str:
    jsonl_file = f"{FILES_PATH}{FILE_NAME}.jsonl" #create_jsonl_file()
    gcs_path = upload_file_to_gcs(jsonl_file)
    corpus_name = create_rag_corpus(gcs_path)
 
if __name__ == "__main__":
    # create_rag()
    state = {"search_query":"תרבות יהודית"}
    res = query_rag(state)

