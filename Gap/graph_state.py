
from typing import Annotated, Dict, List, Sequence, TypedDict

class GraphState(TypedDict):
    # messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    message_type: str  # "general_msg" or "search_item"
    search_query: str
    # vectorstore: object
    retrieved_docs: object
    answer: str
    user_info: str
    chat_history: List[Dict[str, str]]  # רשימה של תמצות שאלות ותשובות
    summary: str  # תמצית של השאלה והתשובה הנוכחית
    selected_maanim: object
    shown_maanim_ids: str # מענים שחזרו בתשובות קודמות 