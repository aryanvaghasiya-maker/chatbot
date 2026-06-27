from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.schema.schema import OptimizedResumeOutput
from agent.config.prompts import RESUME_OPTIMIZATION_PROMPT

class check_optimization_quality:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    def check_optimization_quality(self, state: AdvancedAgentState) -> Literal["discover_companies", "gap_analysis"]:
        if state["ats_score"] >= self.target_score or state["iterations"] >= self.max_loops:
            print(f"[Loop Controller]: Quality Passed ({state['ats_score']}/100, Loops: {state['iterations']}).")
            return "discover_companies"
        print(f"[Loop Controller]: Quality Rejected ({state['ats_score']}/100, Loops: {state['iterations']}). Re-routing to Gap Analyzer.")
        return "gap_analysis"

class optimize_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        # Explicitly initializes its own internal LLM instance variable worker
        from langchain_openai import ChatOpenAI
        from decouple import config
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
    
    # FIXED: Removed 'llm' from the parameters list to perfectly match the 2-argument (self, state) layout
    async def optimize_resume_node(self, state: AdvancedAgentState):
        history_context = "\n".join(state.get("critique_history", []))
        user_content = f"Resume: {state['raw_resume']}\nJob: {state['job_description']}\nHistory: {history_context}"
        
        messages = [
            SystemMessage(content=RESUME_OPTIMIZATION_PROMPT),
            HumanMessage(content=user_content)
        ]
        
        # FIXED: References its own internal instance variable 'self.llm' directly
        structured_llm = self.llm.with_structured_output(OptimizedResumeOutput)
        optimized_data = await structured_llm.ainvoke(messages)
        
        # Continues utilizing the LangGraph state reducer mapping to increment values safely
        return {"optimized_resume": optimized_data, "iterations": 1}
