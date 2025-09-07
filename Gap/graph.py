from langgraph.graph import StateGraph, END
from graph_state import GraphState
from llm import generate_answer,process_user_query,classify_message,create_summary
from rag import query_rag

def create_workflow():
    workflow = StateGraph(GraphState)
    # הוספת שלבים
    workflow.add_node("classify", classify_message)
    workflow.add_node("process_query", process_user_query)
    workflow.add_node("retrieve", query_rag)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("create_summary", create_summary)  # שלב חדש ליצירת תמצית

    # נקודת כניסה
    workflow.set_entry_point("classify")

    def route_after_classification(state: GraphState) -> str:
        message_type = state["message_type"]
        print(f"Routing based on message_type: {message_type}")
        return message_type

    # מעבר לפי סיווג
    workflow.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "general_msg": END,
            "search_msg": "process_query"
        }
    )

    # הסתעפות לאחר הסיווג
    workflow.add_edge("process_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "create_summary") 
    workflow.add_edge("create_summary", END)

    return workflow.compile()

app_graph = create_workflow()