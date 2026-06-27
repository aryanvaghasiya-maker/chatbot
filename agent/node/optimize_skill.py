from agent.states.states import AdvancedAgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from decouple import config
from agent.schema.schema import OptimizedResumeOutput

class optimize_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
    
    async def optimize_resume_node(self, state: AdvancedAgentState):
        history_context = "\n".join(state.get("critique_history", []))
        prompt = f"""
        You are an elite executive resume writer. Tailor the applicant's resume content to perfectly align with the target job profile.
        Raw Resume: {state['raw_resume']}
        Target Job: {state['job_description']}
        Previous Evaluation Feedback to Address: {history_context}
        Rewrite sections to implement the STAR/CAR structure. Inject missing keywords organically.
        """
        structured_llm = self.llm.with_structured_output(OptimizedResumeOutput)
        optimized_data = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {"optimized_resume": optimized_data, "iterations": state.get("iterations", 0) + 1}
