from agent.states.states import AdvancedAgentState
from agent.services.latex_renderer import render_latex


class inject_suggestions_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def inject_suggestions_node(self, state: AdvancedAgentState, llm):
        optimized = state["optimized_resume"]

        # Render LaTeX from the structured schema — no LLM involved.
        # The template is fixed so no job description can leak into the output.
        latex_code = render_latex(optimized)

        # Build the suggestion text from the now-populated skills
        all_skills = optimized.all_skills()  # method on the updated schema
        if all_skills:
            top = ", ".join(all_skills[:5])
            suggestion = (
                f"Focus on the candidate's strongest skills: {top}. "
                "Ensure the final resume highlights measurable results and ATS keywords."
            )
        else:
            suggestion = (
                "Review the optimized resume and emphasize the candidate's "
                "top skills and impact metrics for the target role."
            )

        return {"latex_code": latex_code, "suggestion": suggestion}
