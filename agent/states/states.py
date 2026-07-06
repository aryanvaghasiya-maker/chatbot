from typing import Any, Dict, List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from agent.schema.schema import OptimizedResumeOutput


def merge_tokens(old: Dict[str, int] | None, new: Dict[str, int] | None) -> Dict[str, int]:
    if not old:
        old = {"input": 0, "output": 0, "total": 0}
    if not new:
        return old
    if new.get("reset"):
        return {
            "input": new.get("input", 0),
            "output": new.get("output", 0),
            "total": new.get("total", 0),
        }
    return {
        "input": old.get("input", 0) + new.get("input", 0),
        "output": old.get("output", 0) + new.get("output", 0),
        "total": old.get("total", 0) + new.get("total", 0),
    }


class ResumeState(TypedDict):
    # ---- Input ---------------------------------------------------------------
    raw_resume: str
    resume_file_path: str
    resume_format: str
    job_description: str
    linkedin_url: str
    github_url: str

    # ---- Parsed --------------------------------------------------------------
    parsed_resume: Dict[str, Any]
    extracted_text: str
    extracted_skills: List[str]
    extracted_projects: List[str]
    extracted_experience: List[str]
    extracted_certifications: List[str]
    original_experience: List[Any]
    original_projects: List[Any]
    original_certifications: List[Any]
    original_contact: Dict[str, str]

    # ---- ATS Analysis --------------------------------------------------------
    matched_keywords: List[str]
    missing_keywords: List[str]
    ats_score: int
    ats_breakdown: Dict[str, int]           # ATSBreakdown as dict
    experience_analysis: Dict[str, int]    # ExperienceAnalysis as dict
    improvement_changes: List[str]
    grammar_score: int
    readability_score: int
    grammar_errors: List[Dict[str, Any]]    # GrammarError as list of dicts
    quality_checks: Dict[str, Any]          # QualityChecks as dict
    structured_critique: Dict[str, Any]     # StructuredCritique as dict
    feedback: List[str]
    critique_history: List[str]             # critique history (strings)
    ats_score_explanation: str
    company_specific_ats_analysis: Dict[str, str]

    # ---- Optimized Resume ----------------------------------------------------
    optimized_resume: OptimizedResumeOutput

    # ---- Generation ----------------------------------------------------------
    latex_code: str
    markdown_resume: str
    selected_template: str
    selected_theme: str

    # ---- Career Content ------------------------------------------------------
    cover_letter: str
    interview_questions: List[str]
    roadmap: List[str]
    suggestion: str
    linkedin_optimization: Dict[str, Any]   # LinkedInOptimization as dict
    github_optimization: Dict[str, Any]     # GitHubOptimization as dict

    # ---- Recommendations -----------------------------------------------------
    suggested_companies: List[Any]          # List[CompanyRecommendation dicts]
    portfolio_suggestions: List[str]

    iterations: int
    approved: bool
    version_id: str
    generate_cover_letter: bool
    generate_interview_questions: bool
    generate_roadmap: bool
    generate_suggested_projects: bool
    status: str
    errors: List[Dict[str, str]]

    # ---- Persistence ---------------------------------------------------------
    history: List[Dict[str, Any]]

    # ---- Metrics -------------------------------------------------------------
    token_usage: Annotated[Dict[str, int], merge_tokens]
    execution_time: float
    total_cost: float


AdvancedAgentState = ResumeState
