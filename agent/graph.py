from langgraph.graph import (StateGraph,START, END)
from agent.state import AgentState
from agent.node import agent_call,llm_call_with_tool


workflow = StateGraph(AgentState)

workflow.add_node("agent_call", agent_call)
workflow.add_node("llm_call_with_tool", llm_call_with_tool)

workflow.add_edge(START, "agent_call")

workflow.add_conditional_edges(
    "agent_call",
    llm_call_with_tool,
    {
        "llm_call_with_tool": "llm_call_with_tool",
        END: END
    }
)