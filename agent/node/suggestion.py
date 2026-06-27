from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import LATEX_GENERATION_PROMPT
from langchain_openai import ChatOpenAI
from decouple import config

class inject_suggestions_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))

    async def inject_suggestions_node(self, state: AdvancedAgentState, llm):
        user_content = f"Optimized Profile Data:\n{state['optimized_resume']}"
        messages = [SystemMessage(content=LATEX_GENERATION_PROMPT), HumanMessage(content=user_content)]
        response = await llm.ainvoke(messages)
        clean_latex = response.content.replace("```latex", "").replace("```", "").strip()
        return {"suggested_companies": clean_latex}
