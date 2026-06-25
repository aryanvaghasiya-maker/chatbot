from typing_extensions import TypedDict
from pydantic import BaseModel,Field
from typing import Annotated,Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]
    client_name: str | None
    client_phone: str | None
    service: str | None
    preferred_datetime: str | None
    
class chatrequest(BaseModel):
    message : str
    session_id : str

