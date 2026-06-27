from agent.states.states import AdvancedAgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from decouple import config
from agent.schema.schema import EvaluationOutput
from agent.config.prompts import ATS_EVALUATION_PROMPT 
class evaluate_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
        
    async def evaluate_resume_node(self, state: AdvancedAgentState, llm, target_score: int):
        prompt = ATS_EVALUATION_PROMPT
        structured_llm = llm.with_structured_output(EvaluationOutput)
        eval_result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        current_history = state.get("critique_history", [])
        if eval_result.ats_score < target_score:
            current_history.append(f"Iteration {state['iterations']} Critique: {eval_result.critique}")

        return {
            "ats_score": eval_result.ats_score,
            "missing_keywords": eval_result.missing_keywords,
            "critique_history": current_history
        }
