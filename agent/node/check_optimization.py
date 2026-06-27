from typing import Literal
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.config.prompts import RESUME_OPTIMIZATION_PROMPT
from agent.states.states import AdvancedAgentState
from agent.schema.schema import OptimizedResumeOutput

class check_optimization_quality:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))

    def check_optimization_quality(self, state: AdvancedAgentState) -> Literal["optimize_resume", "discover_companies"]:
        if state["ats_score"] >= self.target_score or state["iterations"] >= self.max_loops:
            print(f"[Loop Controller]: Quality Passed ({state['ats_score']}/100, Loops: {state['iterations']}). Proceeding to Market Placement.")
            return "discover_companies"
        print(f"[Loop Controller]: Quality Rejected ({state['ats_score']}/100). Re-routing to Optimization Node.")
        return "optimize_resume"

class optimize_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
    
    async def optimize_resume_node(self, state: AdvancedAgentState):
        history_context = "\n".join(state.get("critique_history", []))
        prompt = RESUME_OPTIMIZATION_PROMPT
        structured_llm = self.llm.with_structured_output(OptimizedResumeOutput)
        optimized_data = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {"optimized_resume": optimized_data, "iterations": state.get("iterations", 0) + 1}
