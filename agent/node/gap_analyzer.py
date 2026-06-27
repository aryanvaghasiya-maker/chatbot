# agent/node/gap_analyzer.py
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import GAP_ANALYZER_PROMPT
from langchain_openai import ChatOpenAI
from decouple import config

class gap_analyzer_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))


    async def analyze_gaps_node(self, state: AdvancedAgentState, llm):
        latest_critique = state.get("critique_history", [])[-1] if state.get("critique_history") else "Review formatting optimization."
        
        user_content = f"""
        Target Job Requirements: {state['job_description']}
        Latest Evaluation Performance Feedback: {latest_critique}
        Missing Keywords Identified: {state.get('missing_keywords', [])}
        """
        
        messages = [
            SystemMessage(content=GAP_ANALYZER_PROMPT),
            HumanMessage(content=user_content)
        ]
        
        response = await llm.ainvoke(messages)
    
        current_history = state.get("critique_history", [])
        current_history.append(f"Strategic Alignment Update: {response.content}")
        
        print(f"[Gap Analyzer]: Extracted corrections tracking strategies for iteration loop.")
        return {
            "critique_history": current_history,
            "iterations": state.get("iterations", 0)  # Preserves loop tracking indices
        }
