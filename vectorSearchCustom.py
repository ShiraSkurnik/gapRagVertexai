# embedding work, but search faild on proxy with gRPC  its work only with http1
# error:
# <_InactiveRpcError of RPC that terminated with:
#         status = StatusCode.UNAVAILABLE
#         details = "failed to connect to all addresses; last error: UNKNOWN: ipv4:10.20.14.35:8080: Trying to connect an http1.x server (HTTP status 500)"
#         debug_error_string = "UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.20.14.35:8080: Trying to connect an http1.x server (HTTP status 500)"}"
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from vertexai.language_models import TextEmbeddingModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import os
import json
import hashlib
from datetime import datetime
from typing import List, Optional,Dict
import time
 
# אתחול GCP
load_dotenv()
 
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gapmaanim")
location = "europe-west4"
aiplatform.init(project=project_id, location=location)
 
# אתחול המודל של Vertex AI
model = TextEmbeddingModel.from_pretrained("text-embedding-005")
 
# אתחול חיבור ל-Cloud Storage
storage_client = storage.Client()
 
# הגדרות Vertex AI Vector Search
BUCKET_NAME = "gap-vector-search"
INDEX_DISPLAY_NAME = "gap-rag-index"
INDEX_ENDPOINT_DISPLAY_NAME = "gap-rag-endpoint"
DIMENSIONS = 768  # גודל הוקטורים של text-embedding-005
 
class VertexAIVectorSearch:
    """מחלקה לניהול Vertex AI Vector Search"""
    
    def __init__(self, project_id: str, location: str, bucket_name: str):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.index = None
        self.index_endpoint = None
        self.deployed_index_id = None
 
    def prepare_embeddings_for_index(self, documents: List[Document], embeddings: List) -> str:
        """
        הכנת קובץ embeddings בפורמט הנדרש עבור Vertex AI Vector Search
        
        פורמט נדרש: JSONL עם id, embedding, ו-restricts/crowding_tag
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"index_data/embeddings_{timestamp}.jsonl"
        
        # יצירת bucket אם לא קיים
        bucket = self.storage_client.bucket(self.bucket_name)
        
        # הכנת הנתונים
        data_lines = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # המרת metadata לפורמט restricts
            restricts = []
            
            # הוספת population כ-restrict
            if 'population' in doc.metadata:
                restricts.append({
                    "namespace": "population",
                    "allow": [doc.metadata['population']]
                })
            
            # הוספת code_maane כ-restrict
            if 'code_maane' in doc.metadata:
                restricts.append({
                    "namespace": "code_maane",
                    "allow": [doc.metadata['code_maane']]
                })
            
            # יצירת אובייקט לאינדקס
            index_item = {
                "id": str(i),
                "embedding": embedding.values,  # הוקטור
                "restricts": restricts,  # metadata לסינון
                "crowding_tag": doc.metadata.get('population', 'general'),  # לפיזור תוצאות
                # שמירת התוכן והמטאדאטה המלא בשדה נוסף
                "original_data": {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            
            data_lines.append(json.dumps(index_item))
        
        # העלאת הקובץ ל-GCS
        blob = bucket.blob(output_path)
        blob.upload_from_string('\n'.join(data_lines))
        
        print(f"Uploaded {len(data_lines)} embeddings to gs://{self.bucket_name}/{output_path}")
        return f"gs://{self.bucket_name}/{output_path}"
 
    def create_index(self, embeddings_gcs_uri: str) -> MatchingEngineIndex:
        """יצירת אינדקס חדש"""
        
        print("Creating new Vector Search index...")
        
        # יצירת האינדקס עם הפרמטרים הנכונים
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=INDEX_DISPLAY_NAME,
            contents_delta_uri=embeddings_gcs_uri,
            dimensions=DIMENSIONS,
            approximate_neighbors_count=100,  # מספר השכנים לחיפוש
            distance_measure_type="COSINE_DISTANCE",  # או "DOT_PRODUCT_DISTANCE"
            leaf_node_embedding_count=1000,  # מספר embeddings לכל צומת
            leaf_nodes_to_search_percent=7,  # אחוז הצמתים לסריקה
            description="RAG index for GAP documents with metadata filtering",
            labels={
                "environment": "production",
                "use_case": "rag"
            }
        )
        
        print(f"Index created: {index.resource_name}")
        self.index = index
        return index
    
    def create_index_endpoint(self) -> MatchingEngineIndexEndpoint:
        """יצירת endpoint לאינדקס"""
        
        print("Creating index endpoint...")
        
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=INDEX_ENDPOINT_DISPLAY_NAME,
            description="Endpoint for GAP RAG index",
            public_endpoint_enabled=True,  # אפשר גישה ציבורית
            labels={
                "environment": "production",
                "use_case": "rag"
            }
        )
        
        print(f"Index endpoint created: {index_endpoint.resource_name}")
        self.index_endpoint = index_endpoint
        return index_endpoint
 
    def deploy_index(self, index: MatchingEngineIndex, index_endpoint: MatchingEngineIndexEndpoint):
        """פריסת האינדקס ל-endpoint"""
    
        print("Deploying index to endpoint...")
        
        # בדיקה אם האינדקס כבר פרוס
        deployed_indexes = index_endpoint.deployed_indexes
        for deployed in deployed_indexes:
            if deployed.index == index.resource_name:
                print(f"Index is already deployed with ID: {deployed.id}")
                self.deployed_index_id = deployed.id
                return
        
        # יצירת ID ייחודי לפריסה
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deployed_id = f"deployed_{INDEX_DISPLAY_NAME.replace('-', '_')}_{timestamp}"
        
        # אם יש כבר פריסות אחרות באותו endpoint, בדיקה של ה-IDs שלהן
        existing_ids = [d.id for d in deployed_indexes]
        if existing_ids:
            print(f"Existing deployed IDs: {existing_ids}")
        
        # סוגי מכונות לנסות
        machine_types = [
            "e2-standard-16",   # מומלץ ל-MEDIUM shard
            "e2-highmem-16",    # אלטרנטיבה עם יותר זיכרון
            "e2-standard-8",    # אם המכונות הגדולות לא זמינות
            "n2d-standard-8",   # מכונה מבוססת AMD
        ]
        
        for machine_type in machine_types:
            try:
                print(f"Attempting to deploy with machine type: {machine_type}")
                
                deployed_index = index_endpoint.deploy_index(
                    index=index,
                    deployed_index_id=deployed_id,
                    display_name=f"Deployed {INDEX_DISPLAY_NAME}",
                    machine_type=machine_type,
                    min_replica_count=1,
                    max_replica_count=2,
                )
                
                self.deployed_index_id = deployed_id
                print(f"✓ Index deployed successfully!")
                print(f"  - Deployed ID: {deployed_id}")
                print(f"  - Machine type: {machine_type}")
                print(f"  - Status: Deployment initiated")
                
                # המתנה קצרה ובדיקת סטטוס
                print("\nWaiting for deployment to be ready...")
                for i in range(12):  # בדיקה כל 30 שניות למשך 6 דקות
                    time.sleep(30)
                    
                    # רענון מידע על ה-endpoint
                    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                        index_endpoint_name=index_endpoint.resource_name
                    )
                    
                    # בדיקה אם הפריסה הושלמה
                    for deployed in index_endpoint.deployed_indexes:
                        if deployed.id == deployed_id:
                            print(f"✓ Deployment confirmed! Index is ready for queries.")
                            return
                    
                    print(f"  ... still deploying ({i+1}/12)")
                
                print("⚠ Deployment is taking longer than expected, but should complete soon.")
                return
                
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Failed with {machine_type}: {error_msg}")
                
                # אם הבעיה היא ID כפול, ננסה ID אחר
                if "already exists" in error_msg:
                    deployed_id = f"deployed_{INDEX_DISPLAY_NAME.replace('-', '_')}_{timestamp}_{machine_type.replace('-', '_')}"
                    print(f"  Trying with alternative ID: {deployed_id}")
                    continue
        
        raise Exception("Failed to deploy index with any machine type")
 
 
    def undeploy_existing_index(self, index_endpoint: MatchingEngineIndexEndpoint, deployed_id: str):
        """הסרת פריסה קיימת אם נדרש"""
        try:
            print(f"Undeploying existing index: {deployed_id}")
            index_endpoint.undeploy_index(deployed_index_id=deployed_id)
            
            # המתנה להשלמת ההסרה
            print("Waiting for undeploy to complete...")
            time.sleep(60)
            
            print("✓ Successfully undeployed")
        except Exception as e:
            print(f"Error undeploying: {e}")
 
 
    def check_deployment_status(self):
        """בדיקת סטטוס הפריסה"""
        if not self.index_endpoint:
            print("No endpoint found")
            return False
        
        try:
            # רענון מידע
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint.resource_name
            )
            
            deployed_indexes = endpoint.deployed_indexes
            if not deployed_indexes:
                print("No deployed indexes found")
                return False
            
            print(f"\nDeployment Status:")
            print(f"Endpoint: {endpoint.display_name}")
            print(f"Number of deployed indexes: {len(deployed_indexes)}")
            
            for deployed in deployed_indexes:
                print(f"\nDeployed Index ID: {deployed.id}")
                print(f"Index Resource: {deployed.index}")
                print(f"Display Name: {deployed.display_name}")
                
                # בדיקה אם זה האינדקס שלנו
                if self.index and deployed.index == self.index.resource_name:
                    self.deployed_index_id = deployed.id
                    print("✓ This is our index - ready for searches!")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking status: {e}")
            return False
            
    def search(self, query: str, population: Optional[str] = None, code_maane: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        חיפוש משולב - סמנטי + metadata
        
        Args:
            query: השאילתא לחיפוש
            population: סינון לפי אוכלוסייה (אופציונלי)
            code_maane: סינון לפי קוד מענה (אופציונלי)
            top_k: מספר התוצאות
            
        Returns:
            רשימת תוצאות עם ציוני דמיון
        """
        
        if not self.index_endpoint or not self.deployed_index_id:
            raise ValueError("Index endpoint not deployed")
        
        # יצירת embedding לשאילתא
        query_embedding = model.get_embeddings([query])[0].values
        
        # בניית פילטרים
        filters = []
        
        if population:
            filters.append({
                "namespace": "population",
                "allow_list": [population]
            })
            
        if code_maane:
            filters.append({
                "namespace": "code_maane",
                "allow_list": [code_maane]
            })
        
        # ביצוע החיפוש
        response = self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k,
            filter=filters if filters else None
        )
        
        # עיבוד התוצאות
        results = []
        for neighbor_list in response:
            for neighbor in neighbor_list:
                results.append({
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                    "similarity": 1 - neighbor.distance,  # המרה לציון דמיון
                })
        
        return results
 
    def get_existing_index_and_endpoint(self):
        """מציאת אינדקס ו-endpoint קיימים"""
        # חיפוש אינדקס קיים
        indexes = aiplatform.MatchingEngineIndex.list(
            filter=f'display_name="{INDEX_DISPLAY_NAME}"'
        )
        
        if indexes:
            self.index = indexes[0]
            print(f"Found existing index: {self.index.resource_name}")
        
        # חיפוש endpoint קיים
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{INDEX_ENDPOINT_DISPLAY_NAME}"'
        )
        
        if endpoints:
            self.index_endpoint = endpoints[0]
            print(f"Found existing endpoint: {self.index_endpoint.resource_name}")
            
            # בדיקה אם האינדקס פרוס
            deployed_indexes = self.index_endpoint.deployed_indexes
            if deployed_indexes:
                # חיפוש האינדקס הפרוס שלנו
                for deployed in deployed_indexes:
                    if self.index and deployed.index == self.index.resource_name:
                        self.deployed_index_id = deployed.id
                        print(f"Found deployed index: {self.deployed_index_id}")
                        print(f"Deployment is ready and operational!")
                        return
                
                # אם יש אינדקסים פרוסים אבל לא של האינדקס שלנו
                print(f"Found {len(deployed_indexes)} deployed indexes, but none match our index")
                print("Will need to deploy our index with a new ID")
            else:
                print("Endpoint found, but no indexes are deployed yet.")
                if self.index:
                    print("Will deploy the index...")
                    self.deploy_index(self.index, self.index_endpoint)
 
 
 
def get_file_hash(file_path: str) -> str:
    """Generate hash for file to track changes"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error generating file hash: {e}")
        return ""
 
 
def load_document(file_path: str) -> List[Document]:
    """Load and process document based on file type"""
    documents = []
    file_ext = Path(file_path).suffix.lower()
    try:
        if file_ext == '.json':
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for i, item in enumerate(data):
                if i < 1:
                    population = "מוסד"
                elif i < 20:
                    population = "רשות"
                else:
                    population = "מחז"
 
                content = json.dumps(item, indent=2, ensure_ascii=False)
 
                metadata = {
                    "population": population,
                    "type": "json",
                    "code_maane": str(item.get("קוד_מענה", "")),
                    "index": i
                }
 
                documents.append(Document(page_content=content, metadata=metadata))
            
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        
    return documents
 
 
def create_embeddings(documents: List[Document]):
    """Create embeddings for the documents using Vertex AI"""
    if not documents:
        raise ValueError("No documents provided")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    
    # Get embeddings using Vertex AI model
    embeddings = model.get_embeddings([doc.page_content for doc in splits])
    return splits, embeddings
 
def simple_rag_search_fixed(
    query: str,
    top_k: int = 5,
    population: Optional[str] = None,
    code_maane: Optional[str] = None
) -> List[Dict]:
    """
    פונקצית חיפוש מתוקנת ב-RAG
    """
    
    try:
        # 1. מציאת ה-endpoint הקיים
        print("מחפש את ה-endpoint...")
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{INDEX_ENDPOINT_DISPLAY_NAME}"'
        )
        
        if not endpoints:
            raise ValueError("לא נמצא endpoint. וודא שהטעינה הושלמה בהצלחה.")
        
        # יצירת אובייקט endpoint מחדש עם resource name מפורש
        endpoint_resource_name = endpoints[0].resource_name
        print(f"נמצא endpoint: {endpoint_resource_name}")
        
        # יצירת endpoint object חדש בדרך בטוחה יותר
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_resource_name)
        
        # 2. מציאת האינדקס הפרוס
        deployed_indexes = index_endpoint.deployed_indexes
        if not deployed_indexes:
            raise ValueError("לא נמצא אינדקס פרוס. וודא שהפריסה הושלמה.")
        
        deployed_index_id = deployed_indexes[0].id
        print(f"משתמש באינדקס: {deployed_index_id}")
        
        # 3. יצירת embedding לשאילתא
        print(f"יוצר embedding עבור: '{query}'")
        query_embedding = model.get_embeddings([query])[0].values
        
        # 4. בניית פילטרים אם נדרש
        filters = []
        if population:
            filters.append({
                "namespace": "population",
                "allow_list": [population]
            })
            print(f"מסנן לפי אוכלוסייה: {population}")
            
        if code_maane:
            filters.append({
                "namespace": "code_maane",
                "allow_list": [code_maane]
            })
            print(f"מסנן לפי קוד מענה: {code_maane}")
        
        # 5. ביצוע החיפוש בדרך המתוקנת
        print("מבצע חיפוש...")
        #print(index_endpoint._gca_resource.network_endpoint)

        # שימוש ב-match במקום find_neighbors
        try:
            response = index_endpoint.match(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=top_k,
                filter=filters if filters else None
            )
        except AttributeError:
            # אם match לא עובד, ננסה עם public_match
            try:
                response = index_endpoint.public_match(
                    deployed_index_id=deployed_index_id,
                    queries=[query_embedding],
                    num_neighbors=top_k,
                    filter=filters if filters else None
                )
            except AttributeError:
                # אם גם זה לא עובד, ננסה עם find_neighbors באופן ישיר
                response = index_endpoint.find_neighbors(
                    deployed_index_id=deployed_index_id,
                    queries=[query_embedding],
                    num_neighbors=top_k,
                    filter=filters if filters else None
                )
        
        # 6. עיבוד התוצאות
        results = []
        if response:
            neighbor_list = response[0] if isinstance(response, list) else response
            
            for neighbor in neighbor_list:
                similarity_score = 1 - neighbor.distance
                
                result = {
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                    "similarity": similarity_score,
                    "similarity_percent": f"{similarity_score * 100:.1f}%"
                }
                results.append(result)
        
        print(f"נמצאו {len(results)} תוצאות")
        return results
        
    except Exception as e:
        print(f"שגיאה בחיפוש: {e}")
        print(f"סוג השגיאה: {type(e)}")
        return []
 
def alternative_search_method(
    query: str,
    top_k: int = 5,
    population: Optional[str] = None,
    code_maane: Optional[str] = None
) -> List[Dict]:
    """
    שיטת חיפוש אלטרנטיבית באמצעות המחלקה שיצרת
    """
    try:
        # שימוש במחלקה המקורית שלך
        from your_original_file import VertexAIVectorSearch  # החלף לשם הקובץ שלך
        
        vector_search = VertexAIVectorSearch(
            project_id=project_id,
            location=location,
            bucket_name="gap-vector-search"
        )
        
        # מציאת המשאבים הקיימים
        vector_search.get_existing_index_and_endpoint()
        
        if not vector_search.deployed_index_id:
            raise ValueError("לא נמצא אינדקס פרוס")
        
        # ביצוע החיפוש
        results = vector_search.search(
            query=query,
            population=population,
            code_maane=code_maane,
            top_k=top_k
        )
        
        # הוספת אחוזי דמיון
        for result in results:
            result["similarity_percent"] = f"{result['similarity'] * 100:.1f}%"
        
        return results
        
    except Exception as e:
        print(f"שגיאה בשיטה האלטרנטיבית: {e}")
        return []
 
def check_dependencies():
    """בדיקת גרסאות הספריות"""
    try:
        import google.cloud.aiplatform
        print(f"google-cloud-aiplatform version: {google.cloud.aiplatform.__version__}")
        
        import vertexai
        print(f"google-cloud-aiplatform installed")
        
        # בדיקת חיבור
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
        print(f"נמצאו {len(endpoints)} endpoints")
        
        for endpoint in endpoints:
            print(f"  - {endpoint.display_name}: {endpoint.resource_name}")
            
    except Exception as e:
        print(f"שגיאה בבדיקת dependencies: {e}")
 
 
 
 
 
def main():
    """פונקציה ראשית מעודכנת"""
    
    # יצירת אינסטנס של Vector Search
    vector_search = VertexAIVectorSearch(
        project_id=project_id,
        location=location,
        bucket_name=BUCKET_NAME
    )
    
    # בדיקה אם יש אינדקס קיים
    vector_search.get_existing_index_and_endpoint()
    
    # בדיקת סטטוס הפריסה
    if vector_search.check_deployment_status():
        print("\n✓ System is ready! Index is deployed and operational.")
        # בדיקת גרסאות
        print("בודק dependencies...")
        check_dependencies()
        
        print("\n" + "="*50)
        print("מבצע חיפוש...")
        
        # ניסיון ראשון - הפונקציה המתוקנת
        results = simple_rag_search_fixed("מידע על שירותים דיגיטליים", top_k=3)
        
        if results:
            print("החיפוש הצליח!")
            for i, result in enumerate(results, 1):
                print(f"  {i}. ID: {result['id']}, דמיון: {result['similarity_percent']}")
        else:
            print("החיפוש לא הצליח, מנסה שיטה אלטרנטיבית...")
            # ננסה את השיטה האלטרנטיבית
            # results = alternative_search_method("מידע על שירותים דיגיטליים", top_k=3)    
    else:
        # אם אין אינדקס, יצירת חדש
        if not vector_search.index:
            print("\nNo existing index found. Creating new index...")
            
            # טעינת המסמכים
            data_file = "files/data.json"
            documents = load_document(data_file)
            print(f"Loaded {len(documents)} documents")
            
            # יצירת embeddings
            splits, embeddings = create_embeddings(documents)
            
            # הכנת הנתונים לאינדקס
            embeddings_uri = vector_search.prepare_embeddings_for_index(splits, embeddings)
            
            # יצירת האינדקס
            index = vector_search.create_index(embeddings_uri)
            vector_search.index = index
        
        # יצירת endpoint אם אין
        if not vector_search.index_endpoint:
            index_endpoint = vector_search.create_index_endpoint()
            vector_search.index_endpoint = index_endpoint
        
        # פריסת האינדקס אם לא פרוס
        if not vector_search.deployed_index_id and vector_search.index and vector_search.index_endpoint:
            vector_search.deploy_index(vector_search.index, vector_search.index_endpoint)
    
        # בדיקה סופית שהכל מוכן
        if not vector_search.deployed_index_id:
            print("\n⚠ Warning: Index deployment may still be in progress.")
            print("You can run the script again in a few minutes to check status.")
            
            # אפשרות לבדוק שוב
            user_input = input("\nDo you want to wait and check again? (y/n): ")
            if user_input.lower() == 'y':
                print("Waiting 2 minutes before checking again...")
                time.sleep(120)
                if vector_search.check_deployment_status():
                    print("\n✓ Deployment completed successfully!")
                else:
                    print("\n⚠ Deployment still in progress. Please try again later.")
                    return
            else:
                return
 
        
if __name__ == "__main__":
    main()
 
 