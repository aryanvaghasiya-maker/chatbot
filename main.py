from langgraph.graph import StateGraph, START , END
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Weather(BaseModel):
    location : str = Field()

@tool("weather_tool", args_schema=Weather)
def weather_tool(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."

tool_list = [weather_tool]
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm_bind_tool = llm.bind_tools(tool_list)


class AgentState(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]


def agent_call(state: AgentState) -> dict:
    messages = state["messages"]
    chat_history = [SystemMessage(content="You are a helpful assistant tasked.")] + messages

    result = llm_bind_tool.invoke(chat_history)

    if hasattr(result, "tool_calls") and result.tool_calls:
        result.tool_calls = [weather_tool]

    return {
        "messages": [result],
    }

def llm_call_with_tool(state: AgentState) -> str:
    messages = state["messages"][-1]

    if hasattr(messages, "tool_calls") and messages.tool_calls:
        return "agent_call"
    return END

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

workflow.add_edge("llm_call_with_tool", "agent_call")

agent = workflow.compile()

messages = [HumanMessage(content="What is the weather in New York?")]
output_state = agent.invoke({"messages": messages})
print(output_state["messages"][-1].content)