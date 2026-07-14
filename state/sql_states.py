from typing import TypedDict, Any
from pydantic import BaseModel
from typing import Optional,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class SQLState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sql: str
    result: str
    answer: str
    
class chatrequest(BaseModel):
    message : str
    session_id: Optional[str] = None

