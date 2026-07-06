# agent/node/gap_analyzer.py
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import GAP_ANALYZER_PROMPT
from agent.services.llm_factory import get_llm

class gap_analyzer_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = get_llm()

    async def analyze_gaps_node(self, state: AdvancedAgentState, llm):
        total_tokens = state.get("token_usage", {}).get("total", 0) if state.get("token_usage") else 0
        if total_tokens >= 10000:
            print(f"[Gap Analyzer]: Token budget exceeded ({total_tokens} >= 10000). Skipping LLM invocation.")
            return {}

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
        
        token_usage = {"input": 0, "output": 0, "total": 0}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input": response.usage_metadata.get("input_tokens", 0),
                "output": response.usage_metadata.get("output_tokens", 0),
                "total": response.usage_metadata.get("total_tokens", 0),
            }

        print(f"[Gap Analyzer]: Extracted corrections tracking strategies for iteration loop.")
        return {
            "critique_history": current_history,
            "token_usage": token_usage,
        }
