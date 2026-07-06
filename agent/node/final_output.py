from typing import Any
from agent.states.states import ResumeState

class final_output_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def generate_final_assets(self, state: ResumeState) -> dict[str, Any]:
        return {
            "optimized_resume": state.get("optimized_resume"),
            "latex_code": state.get("latex_code", ""),
            "cover_letter": state.get("cover_letter", ""),
            "interview_questions": state.get("interview_questions", []),
            "roadmap": state.get("roadmap", []),
            "linkedin_optimization": state.get("linkedin_optimization", {}),
            "github_optimization": state.get("github_optimization", {}),
            "suggested_companies": state.get("suggested_companies", []),
            "ats_score": state.get("ats_score", 0),
            "ats_breakdown": state.get("ats_breakdown", {}),
            "experience_analysis": state.get("experience_analysis", {}),
            "matched_keywords": state.get("matched_keywords", []),
            "missing_keywords": state.get("missing_keywords", []),
            "grammar_errors": state.get("grammar_errors", []),
            "quality_checks": state.get("quality_checks", {}),
            "structured_critique": state.get("structured_critique", {}),
            "critique_history": state.get("critique_history", []),
            "improvement_changes": state.get("improvement_changes", []),
            "suggestion": state.get("suggestion", ""),
            "approved": state.get("approved", False),
        }
