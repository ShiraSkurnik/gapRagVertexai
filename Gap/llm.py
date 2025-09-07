
import json
import yaml
from vertexai.generative_models import GenerativeModel
from graph_state import GraphState
from dotenv import load_dotenv
from google.auth import default
from google.cloud import aiplatform
from google.oauth2 import service_account

import os

load_dotenv()
os.environ["http_proxy"] = "http://10.20.14.35:8080"
os.environ["https_proxy"] = "http://10.20.14.35:8080"

# טוענים את קובץ ה-YAML
def load_prompts():
    file_path = "Gap/prompts.yaml"
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_prompts()

PROJECT_ID = "ai-gap-470906"
REGION = "europe-west4"
SERVICE_ACCOUNT_JSON = "Gap/rag-service-account-key.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON
)

aiplatform.init(
    project=PROJECT_ID,
    location=REGION, 
    credentials=credentials
)

# פונקציה כללית לפנייה ל-LLM
def invoke_llm(prompt):
    try:
        model = GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        print("תגובה התקבלה")
        print(response.text)
        return response.text
        
    except Exception as e:
        print(f"שגיאה בהפעלת LLM : {e}")

def clean_json(str_data):
    return str_data.replace("json","").replace("```","")

def classify_message(state: GraphState) -> GraphState:
    """Classify the incoming message"""
    question = state["question"]
    prompt = prompts["classify_prompt"]
    full_prompt = prompt.format(question=question)
    response = invoke_llm(full_prompt)
    print(response)
    response = clean_json(response)
    data = json.loads(response)
    if data:
        message_type = data.get("message_type", "search_msg")       
        answer = data.get("answer", "")   
        print(f"Parsed classification: {message_type}, answer: {answer}")
        return {**state,  "message_type": message_type,  "answer": answer}
    else:
        print("Failed to parse classification response")
        return {**state, "message_type": "search_msg", "answer": ""}

def process_user_query(state: GraphState) -> GraphState:
    """Process user query and generate a search query"""
    question = state["question"]
    # TODO: call llm to generate search query to RAG
    print(f"Processing user query: {question}")
    return {**state, "search_query": question} 

def generate_answer(state: GraphState) -> GraphState:
    """Generate answer using retrieved documents"""
    question = state["question"]
    user_info = state["user_info"]
    print(f"Generating answer for question: {question}")
    chat_history = state.get("chat_history", [])
    print(f"Generating answer for chat_history: {chat_history}")
    # יצירת הקשר של היסטוריית צ'אט
    history_context = ""
    if chat_history:
        history_context = "היסטוריית השיחה:\n" + "\n".join([
            f"- {item}" for item in chat_history[-3:]  # רק 3 הרשומות האחרונות
        ]) + "\n\n"
    
    #TODO: add it when the rag work,and delete the next
    documents = state["retrieved_docs"]
    # with open("files/short_long_maanim.json", 'r', encoding='utf-8') as f:
    #     documents = json.load(f)
    prompt = prompts["generate_answer_prompt"]
    full_prompt = prompt.format(question=question,context=documents,user_info=user_info,history_context=history_context)
    answer = invoke_llm(full_prompt)
    return {**state, "answer": answer}


def create_summary(state: GraphState) -> GraphState:
    """יצירת תמצית קצרה של השאלה והתשובה"""
    question = state["question"]
    answer = state["answer"]
    prompt = prompts["summary_prompt"]
    full_prompt = prompt.format(question=question,answer=answer)
    summary = invoke_llm(full_prompt)
    return {**state, "summary": summary}

if __name__ == "__main__":
    state = {"question": "מה תוצאה של 5 + 5?","answer":"התוצאה של התרגיל החשבוני הזה היא 10, מוסיפים את 2 המספרים אחד לשני וככה מקבלים את התוצאה הטובה, אפשר לבדוק את זה במחשבון פשוט או חכם לא משנה איזה העיקר שיודע לחשב"}
    # classify_message(state)
    create_summary(state)


