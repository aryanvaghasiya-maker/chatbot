from langchain_core.messages import HumanMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import MARKET_DISCOVERY_PROMPT
class placement_discovery_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        
    async def placement_discovery_node(self, state: AdvancedAgentState, llm, search_tool):
        skills_query = ", ".join(state['optimized_resume'].skills_section[:3])
        query = f"Companies hiring remote or onsite for roles requiring: {skills_query} jobs openings"
        
        search_raw = await search_tool.ainvoke(query)
        prompt = MARKET_DISCOVERY_PROMPT
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"suggested_companies": response.content}
