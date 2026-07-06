from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------

class ATSBreakdown(BaseModel):
    overall: int = Field(default=0, description="Overall ATS match score 0-100.")
    keyword_match: int = Field(default=0, description="Keyword density match score 0-100.")
    skills: int = Field(default=0, description="Technical skills alignment score 0-100.")
    grammar: int = Field(default=0, description="Grammar and readability score 0-100.")
    formatting: int = Field(default=0, description="Resume structure and formatting score 0-100.")
    experience: int = Field(default=0, description="Experience relevance and impact score 0-100.")
    projects: int = Field(default=0, description="Projects section relevance score 0-100.")
    certifications: int = Field(default=0, description="Certifications credentials relevance score 0-100.")
    role_seniority: int = Field(default=0, description="Role title and seniority match score 0-100.")
    metrics_density: int = Field(default=0, description="Quantified business outcomes and metrics density score 0-100.")


class ExperienceAnalysis(BaseModel):
    action_verbs: int = Field(default=0, description="% of bullets starting with strong action verbs (0-100).")
    metrics: int = Field(default=0, description="% of bullets containing quantifiable metrics (0-100).")
    impact: int = Field(default=0, description="Overall impact and clarity score (0-100).")
    leadership: int = Field(default=0, description="Leadership and ownership signals score (0-100).")
    
    # Rich metrics (Issue 7)
    years_required: int = Field(default=3, description="Years of experience required by the JD.")
    years_found: int = Field(default=3, description="Years of experience found in the resume.")
    gap: int = Field(default=0, description="Gap in years between required and found (0 if found >= required).")
    job_hopping: str = Field(default="low", description="Job hopping risk level (low, medium, high).")


class CritiqueRecommendations(BaseModel):
    critical: List[str] = Field(default_factory=list, description="Critical improvements that must be fixed (e.g. missing required keywords).")
    recommended: List[str] = Field(default_factory=list, description="Recommended improvements (e.g. missing preferred keywords).")
    optional: List[str] = Field(default_factory=list, description="Optional improvements (e.g. suggested formatting tweaks, extra sections).")


class StructuredCritique(BaseModel):
    strengths: List[str] = Field(default_factory=list, description="Top strengths of the resume.")
    weaknesses: List[str] = Field(default_factory=list, description="Top weaknesses or areas for improvement.")
    recommendations: CritiqueRecommendations = Field(default_factory=CritiqueRecommendations, description="Priority-based recommendations.")


class GrammarError(BaseModel):
    error: str = Field(description="The specific grammatical error or warning identified.")
    context: str = Field(description="The sentence or snippet where the error was found.")
    suggestion: str = Field(description="How to correct or rewrite the snippet.")


class QualityChecks(BaseModel):
    duplicate_skills: List[str] = Field(default_factory=list, description="Duplicate technical skills found.")
    passive_voice: List[str] = Field(default_factory=list, description="Experience bullets using passive voice.")
    spelling_errors: List[str] = Field(default_factory=list, description="Spelling or typo errors found.")
    weak_verbs: List[str] = Field(default_factory=list, description="Weak or non-action verbs used.")
    
    # Resume quality checks (Issue 20)
    one_page: bool = Field(default=True, description="Whether the resume is optimized for a single page layout.")
    sections_present: bool = Field(default=True, description="Whether all standard sections (Summary, Skills, Experience, Education) are present.")
    contact_complete: bool = Field(default=True, description="Whether essential contact details are fully populated.")
    bullet_consistency: int = Field(default=100, description="Percentage consistency in bullet point punctuation and capitalization (0-100).")
    date_consistency: int = Field(default=100, description="Percentage consistency in date formatting, e.g. Month Year or Year-Year (0-100).")
    tense_consistency: int = Field(default=100, description="Percentage consistency in action verb tense across current/past jobs (0-100).")
    ats_safe_format: bool = Field(default=True, description="Whether the resume formatting is ATS-safe.")


class EvaluationOutput(BaseModel):
    ats_breakdown: ATSBreakdown = Field(
        default_factory=ATSBreakdown,
        description="Detailed ATS scoring across 7 dimensions.",
    )
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords from the JD that ARE present in the resume.",
    )
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="Critical keywords from the JD that are missing from the resume.",
    )
    structured_critique: StructuredCritique = Field(
        default_factory=StructuredCritique,
        description="Strengths, weaknesses, and recommendations to improve the resume.",
    )
    experience_analysis: ExperienceAnalysis = Field(
        default_factory=ExperienceAnalysis,
        description="Quality analysis of the experience bullet points.",
    )
    improvement_changes: List[str] = Field(
        default_factory=list,
        description="List of specific changes made since the previous iteration.",
    )
    grammar_errors: List[GrammarError] = Field(
        default_factory=list,
        description="List of grammatical errors, typos, or style improvements.",
    )
    quality_checks: QualityChecks = Field(
        default_factory=QualityChecks,
        description="Detailed checks for spelling, passive voice, weak verbs, and duplicates.",
    )
    ats_score_explanation: str = Field(
        default="",
        description="Detailed explanation of the overall ATS score, explaining what criteria caused deductions.",
    )
    company_specific_ats_analysis: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of top tech company names (e.g. Google, Amazon, Microsoft, Netflix) to brief analysis and matching score for this resume.",
    )


# ---------------------------------------------------------------------------
# Resume content models
# ---------------------------------------------------------------------------

class ContactInfo(BaseModel):
    name: str = Field(default="", description="Candidate's full name.")
    email: str = Field(default="", description="Candidate's email address.")
    phone: str = Field(default="", description="Candidate's phone number.")
    location: str = Field(default="", description="City and country or state.")
    linkedin: str = Field(default="", description="LinkedIn URL (domain only, e.g. linkedin.com/in/username).")
    github: str = Field(default="", description="GitHub URL (domain only, e.g. github.com/username).")
    portfolio: str = Field(default="", description="Portfolio or personal website URL or empty string.")
    title: str = Field(default="", description="Professional title with key skills, e.g., 'Senior Backend Engineer | Python | FastAPI | AWS | Kubernetes | LangGraph'.")


class ExperienceEntry(BaseModel):
    title: str = Field(description="Job title, e.g. 'Senior Backend Engineer'.")
    company: str = Field(description="Company name.")
    start_date: str = Field(default="", description="Start date, e.g. 'Jan 2022'.")
    end_date: str = Field(default="Present", description="End date or 'Present'.")
    location: str = Field(default="", description="City and state/country.")
    bullets: List[str] = Field(default_factory=list, description="STAR-formatted achievement bullet points.")


class EducationEntry(BaseModel):
    degree: str = Field(description="Degree and field, e.g. 'B.Sc. Computer Science'.")
    institution: str = Field(description="University or college name.")
    graduation_date: str = Field(default="", description="Graduation year or month-year, e.g. 'May 2020'.")
    gpa: str = Field(default="", description="GPA or CGPA, e.g., '3.8/4.0' or '8.8/10'.")


class ProjectEntry(BaseModel):
    name: str = Field(description="Project name.")
    description: str = Field(description="One-line description with tech stack and measurable outcome.")
    tech_stack: List[str] = Field(default_factory=list, description="Key technologies used.")
    github_url: str = Field(default="", description="Link to the GitHub repository, or empty string.")
    live_demo_url: str = Field(default="", description="Link to the live demo, or empty string.")


class SuggestedProject(BaseModel):
    name: str = Field(description="Suggested project name.")
    description: str = Field(description="What to build and how it aligns with the job target. Max 2 sentences.")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies to use. Max 6 items.")
    resume_bullet: str = Field(default="", description="One STAR-format bullet for the resume once built. Max 200 chars.")
    architecture: str = Field(default="", description="Brief system architecture. Max 100 chars.")
    difficulty: str = Field(default="Intermediate", description="Beginner, Intermediate, or Advanced.")
    estimated_time: str = Field(default="2 weeks", description="Estimated completion time.")
    learning_outcome: str = Field(default="", description="Key learning outcome. Max 100 chars.")
    # These fields are filled by post_process templates, never by the LLM structured call
    github_readme: str = Field(default="", description="README template (filled post-LLM, not by model).")
    github_repo_structure: str = Field(default="", description="Repo structure (filled post-LLM, not by model).")


class Certification(BaseModel):
    name: str = Field(description="Certification name, e.g. 'AWS Solutions Architect'.")
    issuer: str = Field(default="", description="Issuing organization.")
    date: str = Field(default="", description="Issue date or year.")


class SkillsSection(BaseModel):
    languages: List[str] = Field(default_factory=list, description="Programming languages.")
    backend: List[str] = Field(default_factory=list, description="Backend frameworks, libraries, runtime environments, e.g. FastAPI, Node.js.")
    ai_llm: List[str] = Field(default_factory=list, description="AI/LLM technologies, e.g. LangGraph, OpenAI API, Vector DBs, Prompt Engineering, RAG.")
    databases: List[str] = Field(default_factory=list, description="Databases and caching systems.")
    cloud: List[str] = Field(default_factory=list, description="Cloud platforms and services, e.g. AWS, GCP, Azure, ECS.")
    devops: List[str] = Field(default_factory=list, description="DevOps, containerization, CI/CD tools, e.g. Docker, Kubernetes, Helm, GitLab CI.")
    messaging: List[str] = Field(default_factory=list, description="Message brokers and event queues, e.g. RabbitMQ, Kafka, Celery.")
    monitoring: List[str] = Field(default_factory=list, description="Observability, logging, and monitoring, e.g. Prometheus, Grafana, ELK.")
    testing: List[str] = Field(default_factory=list, description="Testing libraries and tools, e.g. pytest, unittest.")
    architecture: List[str] = Field(default_factory=list, description="Architectural patterns and methodologies, e.g. Distributed Systems, Microservices, REST APIs.")
    other: List[str] = Field(default_factory=list, description="Other relevant technical skills.")


# ---------------------------------------------------------------------------
# Company recommendation model
# ---------------------------------------------------------------------------

class CompanyRecommendation(BaseModel):
    name: str = Field(description="Company name.")
    role: str = Field(description="Exact open role title, e.g. 'Senior Backend Engineer'.")
    match: int = Field(default=0, description="Match percentage 0-100 based on skills overlap.")
    location: str = Field(default="", description="'Remote', 'Hybrid - City', or 'Onsite - City'.")
    tech_overlap: List[str] = Field(default_factory=list, description="Overlapping technologies.")
    reason: str = Field(default="", description="2 sentences explaining why this candidate fits.")


# ---------------------------------------------------------------------------
# LinkedIn and GitHub Profile Optimization
# ---------------------------------------------------------------------------

class LinkedInOptimization(BaseModel):
    headline: str = Field(default="", description="Optimized, SEO-friendly LinkedIn headline.")
    about: str = Field(default="", description="Compelling, keyword-rich LinkedIn summary/about section.")
    skills: List[str] = Field(default_factory=list, description="Top skills to pin to the profile.")
    featured_projects: List[str] = Field(default_factory=list, description="Projects to showcase under the Featured section.")


class GitHubOptimization(BaseModel):
    bio: str = Field(default="", description="Short, punchy developer bio.")
    pinned_repositories: List[str] = Field(default_factory=list, description="Suggested repository names to pin.")
    readme_suggestions: List[str] = Field(default_factory=list, description="Specific suggestions to improve the GitHub profile README.")


# ---------------------------------------------------------------------------
# Primary optimized resume output
# ---------------------------------------------------------------------------

class OptimizedResumeOutput(BaseModel):
    contact: ContactInfo = Field(
        default_factory=ContactInfo,
        description="Candidate contact information extracted verbatim from the resume.",
    )
    summary: str = Field(
        default="",
        description="A tailored, high-impact professional summary (3-4 sentences).",
    )
    skills: SkillsSection = Field(
        default_factory=SkillsSection,
        description="Categorized technical skills.",
    )
    experience: List[ExperienceEntry] = Field(
        default_factory=list,
        description="Work experience entries with STAR bullet points.",
    )
    education: List[EducationEntry] = Field(
        default_factory=list,
        description="Education history.",
    )
    projects: List[ProjectEntry] = Field(
        default_factory=list,
        description="Projects from the resume (empty list if none found).",
    )
    certifications: List[Certification] = Field(
        default_factory=list,
        description="Certifications from the resume.",
    )
    spoken_languages: List[str] = Field(
        default_factory=list,
        description="Spoken/written human languages, e.g. ['English', 'Hindi'].",
    )
    suggested_projects: List[SuggestedProject] = Field(
        default_factory=list,
        description="2-3 AI-suggested portfolio project ideas to fill skill gaps. Only include when explicitly requested. Each project is compact with no README content.",
    )
    recommended_certifications: List[str] = Field(
        default_factory=list,
        description="Recommended certifications (e.g. AWS CCP, CKA, Terraform Associate) if original certifications are empty.",
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="3-4 quantified achievements strictly supported by the original resume.",
    )
    core_competencies: List[str] = Field(
        default_factory=list,
        description="8 core competencies based on actual technical capabilities.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def all_bullets(self) -> List[str]:
        return [b for exp in self.experience for b in exp.bullets]

    def all_skills(self) -> List[str]:
        s = self.skills
        return (
            (s.languages or []) +
            (s.backend or []) +
            (s.ai_llm or []) +
            (s.databases or []) +
            (s.cloud or []) +
            (s.devops or []) +
            (s.messaging or []) +
            (s.monitoring or []) +
            (s.testing or []) +
            (s.architecture or []) +
            (s.other or [])
        )


class SlimOptimizedResumeOutput(BaseModel):
    """Reduced schema used when generate_suggested_projects=False.
    Identical to OptimizedResumeOutput but without suggested_projects,
    preventing the model from wasting tokens on that field.
    """
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = Field(default="")
    skills: SkillsSection = Field(default_factory=SkillsSection)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    spoken_languages: List[str] = Field(default_factory=list)
    recommended_certifications: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    core_competencies: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def all_bullets(self) -> List[str]:
        return [b for exp in self.experience for b in exp.bullets]

    def all_skills(self) -> List[str]:
        s = self.skills
        return (
            (s.languages or []) + (s.backend or []) + (s.ai_llm or []) +
            (s.databases or []) + (s.cloud or []) + (s.devops or []) +
            (s.messaging or []) + (s.monitoring or []) + (s.testing or []) +
            (s.architecture or []) + (s.other or [])
        )

    def to_full(self) -> "OptimizedResumeOutput":
        """Convert back to full OptimizedResumeOutput with empty suggested_projects."""
        return OptimizedResumeOutput(
            contact=self.contact,
            summary=self.summary,
            skills=self.skills,
            experience=self.experience,
            education=self.education,
            projects=self.projects,
            certifications=self.certifications,
            spoken_languages=self.spoken_languages,
            recommended_certifications=self.recommended_certifications,
            achievements=self.achievements,
            core_competencies=self.core_competencies,
            suggested_projects=[],
        )





# ---------------------------------------------------------------------------
# API-level schemas
# ---------------------------------------------------------------------------

class ParsedResumeResponse(BaseModel):
    filename: str = Field(description="Original uploaded file name.")
    text: str = Field(description="Extracted resume content.")
    ocr_used: bool = Field(default=False, description="Whether OCR was used to extract text.")
    mime_type: str = Field(description="Resolved MIME type of the uploaded resume.")

    # Structured JSON fields for token efficiency
    contact: Dict[str, Any] = Field(default_factory=dict, description="Structured contact details.")
    summary: str = Field(default="", description="Structured summary section.")
    experience: List[Any] = Field(default_factory=list, description="Structured work experience lines.")
    projects: List[Any] = Field(default_factory=list, description="Structured projects lines.")
    skills: Dict[str, Any] = Field(default_factory=dict, description="Structured skills map.")
    education: List[str] = Field(default_factory=list, description="Structured education lines.")
    languages: str = Field(default="", description="Structured languages section.")
    raw_text_length: int = Field(default=0, description="Length of extracted and cleaned raw text.")


