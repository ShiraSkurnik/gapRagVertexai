
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
    file_path = "prompts.yaml"
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_prompts()

PROJECT_ID = "ai-gap-470906"
REGION = "europe-west4"
SERVICE_ACCOUNT_JSON = "rag-service-account-key.json"

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
        model = GenerativeModel("gemini-2.5-flash-lite", generation_config={"response_mime_type": "application/json"})
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
    history = get_history_context(state)
    prompt = prompts["classify_prompt"]

    full_prompt = prompt.format(question=question,history=history)
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

def get_history_context(state: GraphState) -> str:
    chat_history = state.get("chat_history", [])
    # יצירת הקשר של היסטוריית צ'אט
    history_context = ""
    if chat_history:
        history_context = "היסטוריית השיחה:\n" + "\n".join([
            f"- {item}" for item in chat_history[-3:]  # רק 3 הרשומות האחרונות
        ]) + "\n\n"
    return history_context

def process_user_query(state: GraphState) -> GraphState:
    """Process user query and generate a search query"""

    budgets = state["user_info"]
    user_query = state["question"]
    chat_history = get_history_context(state)
    budgets_map = [{"code": x.get("code"), "teur": x.get("teur")} for x in budgets]
    user_fixed_budgets = [
        {"code": x.get("code"), "teur": x.get("teur")}
        for x in budgets
        if x.get("taktzivKavua") is True
    ]

    prompt = prompts["rag_query_prompt"]
    full_prompt = prompt.format(user_query=user_query,chat_history=chat_history,budgets_map=budgets_map,user_fixed_budgets=user_fixed_budgets)
    rag_query = invoke_llm(full_prompt)
    print(f"Processing user query: {rag_query}")
    return {**state, "search_query": rag_query} 

def generate_answer(state: GraphState) -> GraphState:
    """Generate answer using retrieved documents"""
    question = state["question"]
    print(f"Generating answer for question: {question}")
    history_context = get_history_context(state)
    #TODO: add it when the rag work,and delete the next
    documents = state["retrieved_docs"]
    # with open("files/maanim.json", 'r', encoding='utf-8') as f:
    #     documents = json.load(f)
    budgets = state["user_info"]
    budgets_map = [{"code": x.get("code"), "teur": x.get("teur")} for x in budgets]
    user_fixed_budgets = [
        {"code": x.get("code"), "teur": x.get("teur")}
        for x in budgets
        if x.get("taktzivKavua") is True
    ]

    prompt = prompts["generate_answer_prompt"]
    full_prompt = prompt.format(question=question,context=documents,user_info=budgets_map,history_context=history_context,fixed_user_budgets=user_fixed_budgets)
    answer = invoke_llm(full_prompt)
    return {**state, "answer": answer}


def filter_relevant_maanim(state: GraphState) -> GraphState:
    """שלב 1: סינון מענים רלוונטיים לפי תקציבי המשתמש"""
    documents = state["retrieved_docs"]
    user_budgets = state["user_info"]
    # user_budget_codes = [x.get("code") for x in user_budgets]
    user_budget_codes = [{"code": x.get("code"), "teur": x.get("teur")} for x in user_budgets]
    history = get_history_context(state)
    prompt = prompts["filter_and_select_maanim_prompt"] 
    full_prompt = prompt.format(
        documents=documents,
        user_budget_codes=user_budget_codes,
        question=state["question"],
        history = history,
        shown_maanim_ids = state["shown_maanim_ids"]
    )
    
    response = invoke_llm(full_prompt)
    response_data = clean_json(response)
    try:
        parsed_data = json.loads(response_data)
        return {
            **state, 
            "selected_maanim": parsed_data.get("maanim", []),
            "has_more": parsed_data.get("has_more", False)
        }
    except Exception as e:
        print(f"Error parsing response in filter_relevant_maanim: {e}")
        return {
            **state, 
            "selected_maanim": [],
            "has_more": False
        }

# todo:delete
def select_top_maanim(state: GraphState) -> GraphState:
    """שלב 2: בחירת עד 5 מענים הכי רלוונטיים"""
    filtered_docs = state["filtered_docs"]
    question = state["question"]
    history = get_history_context(state)
    
    prompt = prompts["select_top_maanim_prompt"]  # פרומפט קצר וממוקד
    full_prompt = prompt.format(
        filtered_docs=filtered_docs,
        question=question,
        history=history
    )
    
    response = invoke_llm(full_prompt)
    selected_data = clean_json(response)
    parsed_data = json.loads(selected_data)
    print("top_maanim@@@@@@@@@@@@@@@@@@@@@")
    print(parsed_data.get("maanim", []))
    print(parsed_data.get("has_more", False))
    return {
        **state, 
        "selected_maanim": parsed_data.get("maanim", []),
        "has_more": parsed_data.get("has_more", False)
    }

def format_final_answer(state: GraphState) -> GraphState:
    """שלב 3: יצירת התשובה הסופית"""
    selected_maanim = state["selected_maanim"]
    has_more = state["has_more"]
    question = state["question"]
    history = get_history_context(state)

    prompt = prompts["format_answer_prompt"]  # פרומפט קצר וממוקד
    full_prompt = prompt.format(
        selected_maanim=selected_maanim,
        has_more=has_more,
        question=question,
        history=history
    )
    
    response = invoke_llm(full_prompt)
    print("format_final_answer@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
    print(response)
    return {**state, "answer": response}



# עדכון התהליך הראשי
def generate_answer_improved(state: GraphState) -> GraphState:
    """תהליך משופר ליצירת תשובה - 3 שלבים נפרדים"""
    
    # שלב 1: סינון
    state = filter_relevant_maanim(state)
    
    # שלב 2: בחירה
    # state = select_top_maanim(state)
    
    # שלב 3: עיצוב התשובה
    state = format_final_answer(state)
    
    return state









def create_summary(state: GraphState) -> GraphState:
    """יצירת תמצית קצרה של השאלה והתשובה"""
    ans = json.loads(state.get('answer', {}))
    state['shown_maanim_ids'] = ",".join(x for x in [state.get('shown_maanim_ids', ''), str(ans.get('maanim', ''))] if x)
    state['summary'] = f"שאלה: {state.get('question','')}. תשובה: {ans.get('answer','')}"
    return state

    # question = state["question"]
    # answer = state["answer"]
    # # prompt = prompts["summary_prompt"]
    # # full_prompt = prompt.format(question=question,answer=answer)
    # shown_maanim_ids = state['answer']['maanim']
    # state['shown_maanim_ids'] = f"{state['shown_maanim_ids']},{shown_maanim_ids}"
    # answer = state['answer']['answer']
    # summary = f"שאלה:{state['question']}. תשובה: {answer}"#invoke_llm(full_prompt)
    # return {**state, "summary": summary}

if __name__ == "__main__":
    state = {"question": "מה תוצאה של 5 + 5?","answer":"התוצאה של התרגיל החשבוני הזה היא 10, מוסיפים את 2 המספרים אחד לשני וככה מקבלים את התוצאה הטובה, אפשר לבדוק את זה במחשבון פשוט או חכם לא משנה איזה העיקר שיודע לחשב"}
    # classify_message(state)
    create_summary(state)


