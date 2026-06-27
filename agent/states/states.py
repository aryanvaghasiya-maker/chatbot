from typing import List, Annotated
from typing_extensions import TypedDict
import operator
from agent.schema.schema import OptimizedResumeOutput

class AdvancedAgentState(TypedDict):
    raw_resume: str
    job_description: str
    extracted_skills: List[str]
    missing_keywords: List[str]
    critique_history: List[str]
    ats_score: int
    optimized_resume: OptimizedResumeOutput
    suggested_companies: str  
    placement_market_report: str  
    
    # CRITICAL FIX: Annotated with operator.add tells LangGraph to 
    # track and increment this counter across node boundaries instead of overwriting it.
    iterations: Annotated[int, operator.add]
