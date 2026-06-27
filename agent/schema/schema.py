from typing import List
from pydantic import BaseModel, Field

class EvaluationOutput(BaseModel):
    ats_score: int = Field(description="The matching score from 0-100 between the resume and job.")
    missing_keywords: List[str] = Field(description="Critical terms found in the job description but missing from the resume.")
    critique: str = Field(description="Specific, actionable feedback on how to fix the gaps.")

class OptimizedResumeOutput(BaseModel):
    summary: str = Field(description="A tailored, high-impact professional summary.")
    bullet_points: List[str] = Field(description="Tailored work history bullets optimized using the STAR/CAR format.")
    skills_section: List[str] = Field(description="Categorized target keywords injected naturally.")
