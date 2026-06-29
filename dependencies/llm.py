from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.tools import (extract_booking_details,create_booking,check_availability,rag_search)
from agent.state import AgentState
from prompt.prompt import system_propmt
class Agent:

    def __init__(self):
        
        self.llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=config("OPENAI_API_KEY"))
        self.tools = [rag_search,extract_booking_details,check_availability,create_booking]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def agent(self, state: AgentState):

        messages = [SystemMessage(content=system_propmt)] + state["messages"]
        response = await self.llm_with_tools.ainvoke(messages)

        return {
            "messages": [response]
        }

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_call"
        return END

    def build_graph(self,checkpointer):

        workflow = StateGraph(AgentState)

        workflow.add_node("agent",self.agent)
        workflow.add_node("tool_call",ToolNode(self.tools))
        workflow.add_edge( START,"agent")

        workflow.add_conditional_edges("agent",self.should_continue, 
                                       {"tool_call": "tool_call",
                                        END: END})

        workflow.add_edge("tool_call","agent")
        graph = workflow.compile(checkpointer=checkpointer)
        return graph