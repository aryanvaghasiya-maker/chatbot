from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.tools import sql_db_list_tables,sql_db_query,sql_db_query_checker,sql_db_schema
from agent.prompt import system_prompt
from state.sql_states import SQLState
class Agent:

    def __init__(self):
        
        self.llm = ChatOpenAI(model="gpt-5-nano",api_key=config("OPENAI_API_KEY"))
        self.tools = [sql_db_list_tables,sql_db_schema,sql_db_query,sql_db_query_checker,]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def agent(self, state: SQLState):

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await self.llm_with_tools.ainvoke(messages)

        return {
            "messages": [response]
        }

    def should_continue(self, state: SQLState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_call"
        return END

    def build_graph(self, checkpointer=None):

        workflow = StateGraph(SQLState)

        workflow.add_node("agent",self.agent)
        workflow.add_node("tool_call",ToolNode(self.tools))
        workflow.add_edge( START,"agent")

        workflow.add_conditional_edges("agent",self.should_continue, 
                                       {"tool_call": "tool_call",
                                        END: END})

        workflow.add_edge("tool_call","agent")
        graph = workflow.compile(checkpointer=checkpointer)
        return graph