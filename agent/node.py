from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dependencies.llm import llm_with_tools
from agent.state import AgentState,Booking_state
from langgraph.graph import END

def agent_call(state):
    messages = state["messages"]

    response = llm_with_tools.invoke(
        [SystemMessage(content="You are helpful.")]
        + messages
    )

    return {"messages": [response]}


def llm_call_with_tool(state: AgentState) -> str:
    messages = state["messages"][-1]

    if hasattr(messages, "tool_calls") and messages.tool_calls:
        return "agent_call"
    return END