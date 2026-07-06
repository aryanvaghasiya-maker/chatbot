from typing import List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.schema.schema import ExperienceEntry, ProjectEntry, Certification, ContactInfo
from agent.services.llm_factory import get_llm


class OriginalResumeExtraction(BaseModel):
    skills: List[str] = Field(default_factory=list, description="List of technical skills, languages, tools, frameworks, and databases present in the resume.")
    companies: List[str] = Field(default_factory=list, description="List of company/employer names present in the work experience section (e.g. 'TechHive Solutions').")
    projects: List[str] = Field(default_factory=list, description="List of project names present in the projects section (e.g. 'AI Resume Optimizer').")
    certifications: List[str] = Field(default_factory=list, description="List of certification names present in the certifications section.")
    
    # Full structures
    experience: List[ExperienceEntry] = Field(default_factory=list, description="All experience entries parsed from the raw resume.")
    projects_list: List[ProjectEntry] = Field(default_factory=list, description="All project entries parsed from the raw resume.")
    certifications_list: List[Certification] = Field(default_factory=list, description="All certifications parsed from the raw resume.")
    contact: ContactInfo = Field(default_factory=ContactInfo, description="All contact info details parsed from the raw resume.")


EXTRACTION_SYSTEM_PROMPT = """
You are an advanced AI Technical Sourcer and Resume Parsing Engine.
Your mission is to parse the candidate's raw resume and extract the structured content:
1. skills: Technical skills, programming languages, frameworks, databases, and DevOps tools.
2. companies: Clean company names present in the work experience section.
3. projects: Clean project names present in the projects section.
4. certifications: Clean certification names present in the certifications section.
5. experience: Detailed experience list containing title, company, start_date, end_date, location, and bullet points.
6. projects_list: Detailed projects containing name, description, and tech_stack.
7. certifications_list: Detailed certifications containing name, issuer, and date.
8. contact: Contact details containing name, email, phone, location (city and state/country), linkedin (domain/URL), github (domain/URL), and portfolio.

Ensure all fields are fully populated based on the raw resume text. Do not invent any content; extract it exactly as written.
"""


class extract_skills_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = get_llm()
       
    async def extract_skills_node(self, state: AdvancedAgentState):
        import json
        parsed_resume = state.get("parsed_resume", {})
        if parsed_resume:
            content = json.dumps(parsed_resume, indent=2)
        else:
            content = state.get('raw_resume', '')

        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT), 
            HumanMessage(content=content)
        ]

        
        structured_llm = self.llm.with_structured_output(OriginalResumeExtraction, method="function_calling")
        res = await structured_llm.ainvoke(messages)
        
        # Initialise all state channels to prevent KeyError on first run
        return {
            "extracted_skills": res.skills,
            "extracted_experience": res.companies,
            "extracted_projects": res.projects,
            "extracted_certifications": res.certifications,
            "original_experience": [e.model_dump() for e in res.experience],
            "original_projects": [p.model_dump() for p in res.projects_list],
            "original_certifications": [c.model_dump() for c in res.certifications_list],
            "original_contact": res.contact.model_dump(),
            "critique_history": [],
            "iterations": 0,
            "ats_score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "ats_breakdown": {},
            "experience_analysis": {},
            "improvement_changes": [],
            "grammar_errors": [],
            "quality_checks": {},
            "structured_critique": {},
            "cover_letter": "",
            "interview_questions": [],
            "roadmap": [],
            "linkedin_optimization": {},
            "github_optimization": {},
            "suggested_companies": [],
            "token_usage": {"input": 0, "output": 0, "total": 0},
        }
