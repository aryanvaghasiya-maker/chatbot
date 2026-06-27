from typing import Literal
from decouple import config
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from agent.node.check_optimization import check_optimization_quality, optimize_resume_node  
from agent.node.extract_skills import extract_skills_node
from agent.node.evalute import evaluate_resume_node
from agent.node.gap_analyzer import gap_analyzer_node
from agent.node.suggestion import inject_suggestions_node
from agent.node.placement import placement_discovery_node
from agent.states.states import AdvancedAgentState

class ResumeAgent:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
        self.search_tool = DuckDuckGoSearchRun()
        
        self.extractor_instance = extract_skills_node(max_loops, target_score)
        self.optimizer_instance = optimize_resume_node(max_loops, target_score)
        self.evaluator_instance = evaluate_resume_node(max_loops, target_score)
        self.analyzer_instance = gap_analyzer_node(max_loops, target_score)
        self.suggestion_instance = inject_suggestions_node(max_loops, target_score)
        self.discoverer_instance = placement_discovery_node(max_loops, target_score)
        self.router_instance = check_optimization_quality(max_loops, target_score)

    def build_graph(self, checkpointer):
        builder = StateGraph(AdvancedAgentState)

        async def _extract_skills(state): 
            return await self.extractor_instance.extract_skills_node(state)

        # FIXED: Removed 'self.llm' to perfectly match the 2-argument signature (self, state)
        async def _optimize_resume(state): 
            return await self.optimizer_instance.optimize_resume_node(state)

        async def _evaluate_resume(state): 
            return await self.evaluator_instance.evaluate_resume_node(state, self.llm, self.target_score)

        async def _gap_analysis(state): 
            return await self.analyzer_instance.analyze_gaps_node(state, self.llm)

        async def _inject_suggestions(state): 
            return await self.suggestion_instance.inject_suggestions_node(state, self.llm)

        async def _discover_companies(state): 
            return await self.discoverer_instance.placement_discovery_node(state, self.llm, self.search_tool)

        builder.add_node("extract_skills", _extract_skills)
        builder.add_node("optimize_resume", _optimize_resume)
        builder.add_node("evaluate_resume", _evaluate_resume)
        builder.add_node("gap_analysis", _gap_analysis)
        builder.add_node("inject_suggestions", _inject_suggestions)
        builder.add_node("discover_companies", _discover_companies)

        builder.add_edge(START, "extract_skills")
        builder.add_edge("extract_skills", "optimize_resume")
        builder.add_edge("optimize_resume", "evaluate_resume")
        builder.add_edge("gap_analysis", "optimize_resume")
        builder.add_edge("inject_suggestions", "discover_companies")
        builder.add_edge("discover_companies", END)

        builder.add_conditional_edges("evaluate_resume", self.router_instance.check_optimization_quality)
        return builder.compile(checkpointer=checkpointer)
