from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import SKILL_EXTRACTION_PROMPT
from langchain_openai import ChatOpenAI
from decouple import config

class extract_skills_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
       
    async def extract_skills_node(self, state: AdvancedAgentState):
        messages = [
            SystemMessage(content=SKILL_EXTRACTION_PROMPT), 
            HumanMessage(content=state['raw_resume'])
        ]
        response = await self.llm.ainvoke(messages)
        skills = [line.strip("- ") for line in response.content.split("\n") if line]
        
        # CRITICAL FIX: Explicitly initialize state channels to prevent KeyError crashes
        return {
            "extracted_skills": skills, 
            "critique_history": [], 
            "iterations": 0,
            "ats_score": 0,
            "missing_keywords": []
        }
