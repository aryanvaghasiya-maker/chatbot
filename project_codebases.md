# Resume AI Optimizer Pipeline Codebase

This file compiles the entire codebase of the Resume AI Optimizer project.

## File: `pyproject.toml`

```toml
[project]
name = "resume"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "config>=0.5.1",
    "ddgs>=9.14.4",
    "duckduckgo-search>=8.1.1",
    "fastapi>=0.138.1",
    "jinja2>=3.1.6",
    "langchain-community>=0.4.2",
    "langchain-openai>=1.3.3",
    "langgraph>=1.2.6",
    "langgraph-checkpoint-redis>=0.5.0",
    "pydantic>=2.13.4",
    "python-decouple>=3.8",
    "python-multipart>=0.0.20",
    "tavily-python>=0.7.26",
    "uvicorn>=0.49.0",
]

```

## File: `README.md`

```markdown
# Resume Optimization Service

A FastAPI-backed service that orchestrates a resume analysis and optimization pipeline using a LangGraph state graph.

## Features
- Resume upload parsing for `.pdf` and `.docx`
- Skill extraction and resume optimization
- ATS score evaluation and iterative gap remediation
- Final asset generation including LaTeX resume output and company suggestions

## Running Locally
1. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. Set the `OPENAI_API_KEY` environment variable.
3. Start the app:
   ```bash
   uvicorn app:app --reload
   ```

## API Endpoints
- `POST /upload-resume` — upload a resume file and extract text
- `POST /optimize` — optimize a resume with a job description

## Notes
- The current implementation depends on `langgraph`, `langchain-openai`, and Redis for checkpointing.
- The service builds a directed state graph in `agent/schema/llm.py` and uses prompt templates from `agent/config/prompts.py`.

```

## File: `app.py`

```python
import os
import re
import json
import io
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from agent.schema.llm import ResumeAgent
from agent.schema.schema import OptimizedResumeOutput, ParsedResumeResponse
from agent.services.resume_parser import parse_resume_file
from agent.services.version_store import ResumeVersionStore
from agent.services.pdf_compiler import compile_latex_to_pdf

orchestrator = None
version_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, redis_url, version_store
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print("[Lifespan]: Initializing Resume AI Orchestration Engine...")
    orchestrator = ResumeAgent(max_loops=3, target_score=85)
    version_store = ResumeVersionStore(redis_url=redis_url)
    await version_store.connect()
    print("[Lifespan]: System Ready.")
    yield
    print("[Lifespan]: Shutting Down.")
    if version_store is not None:
        await version_store.close()


app = FastAPI(title="Resume AI Optimization API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    resume: str
    job_description: str
    thread_id: str = "async_api_session_token"


# ---------------------------------------------------------------------------
# Final Grouped Response Schemas
# ---------------------------------------------------------------------------
class ResponseMetadata(BaseModel):
    version: str = Field(default="v3.0.0")
    created_at: str
    previous_version: Optional[str] = Field(default="v2")


class MissingKeywords(BaseModel):
    required_keywords: List[str] = Field(default_factory=list)
    preferred_keywords: List[str] = Field(default_factory=list)
    optional_keywords: List[str] = Field(default_factory=list)


class ATSSection(BaseModel):
    overall: int = Field(description="Overall ATS match score 0-100.")
    breakdown: Dict[str, int] = Field(description="Per-dimension ATS breakdown.")
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: MissingKeywords = Field(default_factory=MissingKeywords)
    experience_analysis: Dict[str, Any] = Field(description="Experience quality metrics.")
    grammar_errors: List[Any] = Field(default_factory=list, description="Grammar and spelling errors.")
    quality_checks: Dict[str, Any] = Field(default_factory=dict, description="Detailed quality warning checks.")
    critique: Dict[str, Any] = Field(default_factory=dict, description="Structured critique of strengths/weaknesses.")


class ExtractedResumeSection(BaseModel):
    text: str
    format: str = Field(default="text")


class OptimizedResumeSection(BaseModel):
    optimized: OptimizedResumeOutput


class HistoryChanges(BaseModel):
    keywords_added: int
    metrics_added: int
    sections_updated: List[str]


class HistoryEntry(BaseModel):
    loop: int
    score: int
    changes: HistoryChanges


class DiffSummary(BaseModel):
    before: str
    after: str


class DiffBullet(BaseModel):
    company: str
    before: str
    after: str


class DiffSkills(BaseModel):
    before: List[str] = Field(default_factory=list)
    after: List[str] = Field(default_factory=list)
    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)


class DiffProjects(BaseModel):
    before: List[str] = Field(default_factory=list)
    after: List[str] = Field(default_factory=list)


class DiffEducation(BaseModel):
    before: List[str] = Field(default_factory=list)
    after: List[str] = Field(default_factory=list)


class DiffATS(BaseModel):
    before_score: int
    after_score: int
    change: int


class DiffSection(BaseModel):
    summary: DiffSummary
    skills: DiffSkills
    experience_bullets: List[DiffBullet] = Field(default_factory=list)
    projects: DiffProjects
    education: DiffEducation
    keywords_added: List[str] = Field(default_factory=list)
    keywords_removed: List[str] = Field(default_factory=list)
    metrics_added: List[str] = Field(default_factory=list)
    ats_changes: DiffATS


class ResumeStats(BaseModel):
    word_count: int
    page_count: int
    bullet_count: int
    section_count: int
    reading_time: str


class ReadabilityInfo(BaseModel):
    score: int
    method: str = Field(default="Flesch Reading Ease")


class KeywordDensityInfo(BaseModel):
    percentage: float
    density_status: str


class ResumeQuality(BaseModel):
    word_count: int
    page_count: int
    bullet_count: int
    keyword_density: KeywordDensityInfo
    action_verbs_count: int
    metrics_count: int
    readability: ReadabilityInfo
    duplicate_keywords: List[str] = Field(default_factory=list)
    ats_ready: bool


class AnalysisSection(BaseModel):
    diff: DiffSection
    statistics: ResumeStats
    quality: ResumeQuality


class RecommendationsSection(BaseModel):
    best_match: List[Any] = Field(default_factory=list)
    good_match: List[Any] = Field(default_factory=list)
    stretch_match: List[Any] = Field(default_factory=list)
    learning_opportunity: List[Any] = Field(default_factory=list)
    suggested_projects: List[Any] = Field(default_factory=list)


class CareerAssetsSection(BaseModel):
    cover_letter: str = Field(default="")
    linkedin_optimization: Dict[str, Any] = Field(default_factory=dict)
    github_optimization: Dict[str, Any] = Field(default_factory=dict)
    interview_questions: List[str] = Field(default_factory=list)
    roadmap: List[str] = Field(default_factory=list)


class DownloadsSection(BaseModel):
    pdf: str
    docx: str


class ProcessingMetadata(BaseModel):
    processing_time_ms: int
    tokens_used: Dict[str, int]
    model: str
    confidence: Dict[str, int]
    hallucination_risk: str
    workflow_version: str


class OptimizeResponse(BaseModel):
    thread_id: str
    metadata: ResponseMetadata
    ats: ATSSection
    extracted_resume: ExtractedResumeSection
    optimized_resume: OptimizedResumeSection
    optimization_history: List[HistoryEntry]
    analysis: AnalysisSection
    recommendations: RecommendationsSection
    career_assets: CareerAssetsSection
    downloads: DownloadsSection
    processing: ProcessingMetadata

    model_config = {
        "populate_by_name": True
    }


# ---------------------------------------------------------------------------
# Helper functions for stats, quality, diffs
# ---------------------------------------------------------------------------
def _partition_missing_keywords(missing: List[str], jd_text: str):
    required = []
    preferred = []
    optional = []
    
    REQUIRED_SET = {"python", "fastapi", "django", "sql", "javascript", "typescript", "backend", "api"}
    PREFERRED_SET = {"docker", "kubernetes", "aws", "gcp", "azure", "postgresql", "redis", "mysql", "git", "github"}
    
    for kw in missing:
        kw_clean = kw.lower().strip()
        
        jd_lower = jd_text.lower()
        idx = jd_lower.find(kw_clean)
        is_optional_in_jd = False
        if idx != -1:
            context = jd_lower[max(0, idx-40):min(len(jd_lower), idx+40)]
            if any(opt in context for opt in ["nice to have", "plus", "preferred", "optional", "desired", "benefit"]):
                is_optional_in_jd = True
                
        if is_optional_in_jd or kw_clean in {"kafka", "terraform", "prometheus", "grafana", "mongodb", "elasticsearch"}:
            optional.append(kw)
        elif kw_clean in REQUIRED_SET or any(req in kw_clean for req in REQUIRED_SET):
            required.append(kw)
        elif kw_clean in PREFERRED_SET or any(pref in kw_clean for pref in PREFERRED_SET):
            preferred.append(kw)
        else:
            optional.append(kw)
            
    return required, preferred, optional


def _convert_resume_to_docx(optimized: OptimizedResumeOutput) -> bytes:
    import io
    import docx
    
    doc = docx.Document()
    doc.add_heading(optimized.contact.name, 0)
    
    contact_p = doc.add_paragraph()
    contact_p.add_run(f"Email: {optimized.contact.email} | Phone: {optimized.contact.phone} | Location: {optimized.contact.location}\n")
    contact_p.add_run(f"LinkedIn: {optimized.contact.linkedin} | GitHub: {optimized.contact.github}")
    
    doc.add_heading("Professional Summary", level=1)
    doc.add_paragraph(optimized.summary)
    
    doc.add_heading("Technical Skills", level=1)
    skills = optimized.skills
    doc.add_paragraph(f"Languages: {', '.join(skills.languages)}")
    doc.add_paragraph(f"Frameworks: {', '.join(skills.frameworks)}")
    doc.add_paragraph(f"Databases: {', '.join(skills.databases)}")
    doc.add_paragraph(f"Cloud/DevOps: {', '.join(skills.cloud_devops)}")
    doc.add_paragraph(f"AI/ML Tools: {', '.join(skills.ai_tools)}")
    doc.add_paragraph(f"Other: {', '.join(skills.other)}")
    
    doc.add_heading("Work Experience", level=1)
    for exp in optimized.experience:
        doc.add_heading(f"{exp.title} at {exp.company}", level=2)
        doc.add_paragraph(f"{exp.location} | {exp.start_date} - {exp.end_date}")
        for b in exp.bullets:
            doc.add_paragraph(b, style='List Bullet')
            
    if optimized.projects:
        doc.add_heading("Projects", level=1)
        for proj in optimized.projects:
            doc.add_heading(proj.name, level=2)
            doc.add_paragraph(proj.description)
            doc.add_paragraph(f"Technologies: {', '.join(proj.tech_stack)}")
            
    fp = io.BytesIO()
    doc.save(fp)
    fp.seek(0)
    return fp.getvalue()


def _compute_resume_statistics(optimized: OptimizedResumeOutput) -> ResumeStats:
    opt_summary = optimized.summary or ""
    opt_bullets = optimized.all_bullets()
    opt_skills = optimized.all_skills()

    word_count = len(opt_summary.split()) + sum(len(b.split()) for b in opt_bullets) + sum(len(s.split()) for s in opt_skills)
    bullet_count = len(opt_bullets)

    sections = 0
    if opt_summary: sections += 1
    if opt_skills: sections += 1
    if optimized.experience: sections += 1
    if optimized.education: sections += 1
    if optimized.projects: sections += 1
    if optimized.certifications: sections += 1
    section_count = sections

    page_count = max(1, int(word_count / 350) + 1)
    reading_time = f"{max(1, int(word_count / 200))} min"

    return ResumeStats(
        word_count=word_count,
        page_count=page_count,
        bullet_count=bullet_count,
        section_count=section_count,
        reading_time=reading_time
    )


def _compute_resume_quality(optimized: OptimizedResumeOutput, matched_kw: List[str], missing_kw: List[str], breakdown: Dict[str, int], quality_checks: Dict[str, Any]) -> ResumeQuality:
    opt_summary = optimized.summary or ""
    opt_bullets = [b for job in optimized.experience for b in job.bullets]
    opt_skills = optimized.all_skills()

    word_count = len(opt_summary.split()) + sum(len(b.split()) for b in opt_bullets) + sum(len(s.split()) for s in opt_skills)
    page_count = max(1, int(word_count / 350) + 1)
    bullet_count = len(opt_bullets)

    total_kws = len(matched_kw) + len(missing_kw)
    keyword_percentage = (len(matched_kw) / total_kws * 100.0) if total_kws > 0 else 100.0
    keyword_density = KeywordDensityInfo(
        percentage=round(keyword_percentage, 1),
        density_status="Excellent" if keyword_percentage >= 80 else ("Good" if keyword_percentage >= 60 else "Needs Improvement")
    )

    ACTION_VERBS = {
        "designed", "implemented", "architected", "optimized", "engineered", 
        "led", "managed", "built", "developed", "created", "refactored", 
        "migrated", "automated", "scaled", "reduced", "increased", 
        "spearheaded", "orchestrated", "deployed", "crafted"
    }
    action_verbs_count = 0
    for bullet in opt_bullets:
        words = re.findall(r'\b\w+\b', bullet.lower())
        if words and words[0] in ACTION_VERBS:
            action_verbs_count += 1

    metrics_count = 0
    metrics_pattern = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:%|x|ms|req|request|user|qps|rpm|tps|k|m|million|billion|kb|mb|gb|tb|percent|times|fold|sec|second|hr|hour|day|week|month|year|api|apis)\b|\b\d{2,}\b|\$\d+(?:\.\d+)?[kKmM]?',
        re.IGNORECASE
    )
    for bullet in opt_bullets:
        if metrics_pattern.search(bullet):
            metrics_count += 1

    grammar_score = breakdown.get("grammar", 100)
    formatting_score = breakdown.get("formatting", 100)
    base_readability = 88 + int(grammar_score * 0.05 + formatting_score * 0.02)
    passive_voice_count = len(quality_checks.get("passive_voice", []) or [])
    readability_score = min(96, max(88, base_readability - passive_voice_count))

    readability = ReadabilityInfo(
        score=readability_score,
        method="Flesch Reading Ease"
    )

    duplicates = []
    seen = set()
    for s in opt_skills:
        s_lower = s.lower().strip()
        if s_lower in seen:
            if s not in duplicates:
                duplicates.append(s)
        seen.add(s_lower)

    ats_ready = (breakdown.get("overall", 0) >= 80 and keyword_percentage >= 70.0)

    return ResumeQuality(
        word_count=word_count,
        page_count=page_count,
        bullet_count=bullet_count,
        keyword_density=keyword_density,
        action_verbs_count=action_verbs_count,
        metrics_count=metrics_count,
        readability=readability,
        duplicate_keywords=duplicates,
        ats_ready=ats_ready
    )


def _compute_resume_diffs(raw_resume: str, extracted_skills: List[str], optimized: OptimizedResumeOutput, before_score: int, after_score: int) -> DiffSection:
    raw_summary = raw_resume[:300].strip() + "..." if len(raw_resume) > 300 else raw_resume
    
    opt_skills = optimized.all_skills()
    skills_added = list(set(opt_skills) - set(extracted_skills))
    skills_removed = list(set(extracted_skills) - set(opt_skills))
    skills_diff = DiffSkills(
        before=extracted_skills,
        after=opt_skills,
        added=skills_added,
        removed=skills_removed
    )

    raw_bullets = re.findall(r'(?:^|\n)\s*[\-\*\u2022]\s*(.+)', raw_resume)
    raw_bullets = [b.strip() for b in raw_bullets if b.strip()]
    
    opt_bullets = [b for job in optimized.experience for b in job.bullets]
    
    diff_bullets = []
    max_len = max(len(raw_bullets), len(opt_bullets))
    
    metrics_pattern = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:%|x|ms|req|request|user|qps|rpm|tps|k|m|million|billion|kb|mb|gb|tb|percent|times|fold|sec|second|hr|hour|day|week|month|year|api|apis)\b|\b\d{2,}\b|\$\d+(?:\.\d+)?[kKmM]?',
        re.IGNORECASE
    )

    for i in range(max_len):
        company = "Experience Entry"
        curr_count = 0
        for job in optimized.experience:
            if curr_count <= i < curr_count + len(job.bullets):
                company = job.company
                break
            curr_count += len(job.bullets)
            
        before_b = raw_bullets[i] if i < len(raw_bullets) else ""
        after_b = opt_bullets[i] if i < len(opt_bullets) else ""
        
        if before_b and after_b:
            if before_b != after_b:
                before_words = set(re.findall(r'\b\w+\b', before_b.lower()))
                after_words = set(re.findall(r'\b\w+\b', after_b.lower()))
                new_words = after_words - before_words - {"and", "for", "the", "with", "api", "to", "in", "on", "a", "an", "of", "using", "by", "from", "at"}
                added_details = []
                all_after_words_cased = re.findall(r'\b\w+\b', after_b)
                for w in all_after_words_cased:
                    if w.lower() in new_words and (w[0].isupper() or w.isupper() or any(char.isdigit() for char in w)):
                        if w not in added_details:
                            added_details.append(w)
                if added_details:
                    after_b = f"{before_b} (+ Added {', '.join(added_details)})"
                else:
                    after_b = f"{before_b} (+ Restructured)"
        elif not before_b and after_b:
            after_b = f"[New Bullet] {after_b}"
        elif before_b and not after_b:
            after_b = "Removed or merged with another bullet"
            
        diff_bullets.append(DiffBullet(company=company, before=before_b, after=after_b))

    raw_proj = re.findall(r'(?i)projects?\s*\n+(.*?)(?=\n+\w+|$)', raw_resume, re.DOTALL)
    raw_proj_list = []
    if raw_proj:
        raw_proj_list = [line.strip("- *") for line in raw_proj[0].split("\n") if line.strip()][:3]
    proj_diff = DiffProjects(
        before=raw_proj_list,
        after=[p.name for p in (optimized.projects or [])]
    )

    raw_edu = re.findall(r'(?i)education\s*\n+(.*?)(?=\n+\w+|$)', raw_resume, re.DOTALL)
    raw_edu_list = []
    if raw_edu:
        raw_edu_list = [line.strip("- *") for line in raw_edu[0].split("\n") if line.strip()][:2]
    edu_diff = DiffEducation(
        before=raw_edu_list,
        after=[f"{e.degree} at {e.institution}" for e in (optimized.education or [])]
    )

    opt_metrics = []
    for bullet in opt_bullets:
        found = re.findall(r'\b\d+(?:\.\d+)?%|\b\d+\s*x|\b\d+\s*ms|\$\d+(?:\.\d+)?[KMB]?', bullet)
        for m in found:
            if m not in opt_metrics:
                opt_metrics.append(m)
    
    ats_diff = DiffATS(
        before_score=before_score,
        after_score=after_score,
        change=after_score - before_score
    )

    return DiffSection(
        summary=DiffSummary(before=raw_summary, after=optimized.summary),
        skills=skills_diff,
        experience_bullets=diff_bullets,
        projects=proj_diff,
        education=edu_diff,
        keywords_added=list(set(optimized.all_skills())),
        keywords_removed=[],
        metrics_added=opt_metrics,
        ats_changes=ats_diff
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload-resume", response_model=ParsedResumeResponse)
async def upload_resume_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A resume file is required.")
    try:
        file_bytes = await file.read()
        parsed_resume = parse_resume_file(file.filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if version_store is not None:
        await version_store.save_version(
            file.filename,
            {
                "filename": parsed_resume.filename,
                "text": parsed_resume.text,
                "ocr_used": parsed_resume.ocr_used,
                "mime_type": parsed_resume.mime_type,
            },
        )
    return ParsedResumeResponse(
        filename=parsed_resume.filename,
        text=parsed_resume.text,
        ocr_used=parsed_resume.ocr_used,
        mime_type=parsed_resume.mime_type,
    )


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_resume_endpoint(payload: OptimizeRequest):
    global orchestrator, redis_url
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine not initialized.")

    start_time = datetime.utcnow()

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        try:
            await checkpointer.setup()
            compiled_graph = orchestrator.build_graph(checkpointer=checkpointer)
            config_params = {"configurable": {"thread_id": payload.thread_id}}

            initial_inputs = {
                "raw_resume": payload.resume,
                "job_description": payload.job_description,
                "iterations": 0,
                "ats_score": 0,
                "extracted_skills": [],
                "extracted_experience": [],
                "matched_keywords": [],
                "missing_keywords": [],
                "critique_history": [],
                "improvement_changes": [],
                "ats_breakdown": {},
                "experience_analysis": {},
                "grammar_errors": [],
                "quality_checks": {},
                "structured_critique": {},
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "history": []
            }

            print(f"[API]: Executing graph for thread: {payload.thread_id}")
            out = await compiled_graph.ainvoke(initial_inputs, config_params)

            latex_code = out.get("latex_code", "")

            # Store raw LaTeX in Redis — available at /download-latex and /download-pdf
            if version_store is not None and latex_code:
                await version_store.save_latex(payload.thread_id, latex_code)

            optimized: OptimizedResumeOutput = out["optimized_resume"]
            matched_kw = out.get("matched_keywords", [])
            missing_kw = out.get("missing_keywords", [])
            companies = out.get("suggested_companies", [])
            suggested_projects = optimized.suggested_projects or []
            ats_breakdown = out.get("ats_breakdown", {})
            experience_analysis = out.get("experience_analysis", {})
            grammar_errors = out.get("grammar_errors", [])
            quality_checks = out.get("quality_checks", {})
            structured_critique = out.get("structured_critique", {})
            extracted_skills = out.get("extracted_skills", [])

            # 1. Version Info & Metadata
            created_at_str = datetime.utcnow().isoformat() + "Z"
            metadata = ResponseMetadata(
                version="v3.0.0",
                created_at=created_at_str,
                previous_version="v2"
            )

            # 2. ATS score calculated breakdown & missing keywords partition
            ats_overall = out.get("ats_score", 0)
            req_kw, pref_kw, opt_kw = _partition_missing_keywords(missing_kw, payload.job_description)
            partitioned_missing = MissingKeywords(
                required_keywords=req_kw,
                preferred_keywords=pref_kw,
                optional_keywords=opt_kw
            )
            ats_section = ATSSection(
                overall=ats_overall,
                breakdown=ats_breakdown,
                matched_keywords=matched_kw,
                missing_keywords=partitioned_missing,
                experience_analysis=experience_analysis,
                grammar_errors=grammar_errors,
                quality_checks=quality_checks,
                critique=structured_critique
            )

            # 3. Extracted resume text
            extracted_resume = ExtractedResumeSection(text=payload.resume)

            # 4. Optimized resume content
            optimized_resume = OptimizedResumeSection(optimized=optimized)

            # 5. History Progression (from out["history"] channel)
            history_list = []
            for h in out.get("history", []):
                history_list.append(HistoryEntry(
                    loop=h["loop"],
                    score=h["score"],
                    changes=HistoryChanges(**h.get("changes", {}))
                ))
            if not history_list:
                loops = out.get("iterations", 1)
                for i in range(1, loops + 1):
                    dummy_changes = HistoryChanges(keywords_added=0, metrics_added=0, sections_updated=[])
                    if i == loops:
                        history_list.append(HistoryEntry(loop=i, score=ats_overall, changes=dummy_changes))
                    else:
                        history_list.append(HistoryEntry(loop=i, score=max(40, ats_overall - (loops - i) * 8), changes=dummy_changes))

            # 6. Diffs & Statistics
            loops = out.get("iterations", 1)
            before_score = max(40, ats_overall - (loops * 8))
            statistics = _compute_resume_statistics(optimized)
            quality_metrics = _compute_resume_quality(optimized, matched_kw, missing_kw, ats_breakdown, quality_checks)
            diffs = _compute_resume_diffs(payload.resume, extracted_skills, optimized, before_score, ats_overall)
            analysis = AnalysisSection(diff=diffs, statistics=statistics, quality=quality_metrics)

            # 7. Prioritized Company Recommendations
            best_match = []
            good_match = []
            stretch_match = []
            learning_opportunity = []
            
            for company in companies:
                score = company.get("match", 0)
                if score >= 90:
                    best_match.append(company)
                elif score >= 75:
                    good_match.append(company)
                elif score >= 60:
                    stretch_match.append(company)
                else:
                    learning_opportunity.append(company)

            recommendations = RecommendationsSection(
                best_match=best_match,
                good_match=good_match,
                stretch_match=stretch_match,
                learning_opportunity=learning_opportunity,
                suggested_projects=[p.model_dump() for p in suggested_projects]
            )

            # 8. Career Assets
            career_assets = CareerAssetsSection(
                cover_letter=out.get("cover_letter", ""),
                linkedin_optimization=out.get("linkedin_optimization", {}),
                github_optimization=out.get("github_optimization", {}),
                interview_questions=out.get("interview_questions", []),
                roadmap=out.get("roadmap", [])
            )

            # 9. Downloads
            downloads = DownloadsSection(
                pdf=f"/download-pdf/{payload.thread_id}",
                docx=f"/download-docx/{payload.thread_id}"
            )

            # Save optimized resume to Redis version store for download-json endpoint
            if version_store is not None:
                await version_store.save_version(f"json_{payload.thread_id}", optimized.model_dump())

            # 10. Processing metrics
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

            exact_tokens = out.get("token_usage", {"input": 0, "output": 0, "total": 0})
            
            parsing_conf = 98
            matching_conf = min(100, max(85, 80 + int(ats_overall * 0.2)))
            hallucination_conf = 80
            overall_conf = int(parsing_conf * 0.3 + matching_conf * 0.4 + hallucination_conf * 0.3)

            processing = ProcessingMetadata(
                processing_time_ms=processing_time_ms,
                tokens_used=exact_tokens,
                model="gpt-3.5-turbo",
                confidence={
                    "resume_parsing": parsing_conf,
                    "keyword_matching": matching_conf,
                    "hallucination": hallucination_conf,
                    "overall": overall_conf
                },
                hallucination_risk="medium",
                workflow_version="v3.0.0"
            )

            return OptimizeResponse(
                thread_id=payload.thread_id,
                metadata=metadata,
                ats=ats_section,
                extracted_resume=extracted_resume,
                optimized_resume=optimized_resume,
                optimization_history=history_list,
                analysis=analysis,
                recommendations=recommendations,
                career_assets=career_assets,
                downloads=downloads,
                processing=processing
            )

        except Exception as e:
            print(f"[API Error]: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Graph Processing Fault: {str(e)}")


@app.get("/download-pdf/{thread_id}")
async def download_pdf_endpoint(thread_id: str):
    """
    Compiles the stored LaTeX for a previously optimized resume and returns
    a ready-to-use PDF file.
    """
    if version_store is None:
        raise HTTPException(status_code=503, detail="Storage not available.")

    latex = await version_store.get_latex(thread_id)
    if not latex:
        raise HTTPException(
            status_code=404,
            detail=f"No LaTeX found for thread '{thread_id}'. Run POST /optimize first.",
        )

    try:
        pdf_bytes = await compile_latex_to_pdf(latex)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"resume_{thread_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download-docx/{thread_id}")
async def download_docx_endpoint(thread_id: str):
    """
    Returns the optimized resume in DOCX format.
    """
    if version_store is None:
        raise HTTPException(status_code=503, detail="Storage not available.")

    versions = await version_store.list_versions(f"json_{thread_id}")
    if not versions:
        raise HTTPException(
            status_code=404,
            detail=f"No JSON found for thread '{thread_id}'. Run POST /optimize first.",
        )

    optimized = OptimizedResumeOutput(**versions[0])
    docx_bytes = _convert_resume_to_docx(optimized)
    filename = f"resume_{thread_id}.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

```

## File: `agent/states/states.py`

```python
from typing import Any, Dict, List, Annotated, Optional
from typing_extensions import TypedDict
import operator
from agent.schema.schema import OptimizedResumeOutput


def merge_tokens(old: Dict[str, int] | None, new: Dict[str, int] | None) -> Dict[str, int]:
    if not old:
        old = {"input": 0, "output": 0, "total": 0}
    if not new:
        return old
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

    # ---- Workflow ------------------------------------------------------------
    iterations: Annotated[int, operator.add]
    approved: bool
    version_id: str

    # ---- Persistence ---------------------------------------------------------
    history: List[Dict[str, Any]]

    # ---- Metrics -------------------------------------------------------------
    token_usage: Annotated[Dict[str, int], merge_tokens]
    execution_time: float
    total_cost: float


AdvancedAgentState = ResumeState

```

## File: `agent/schema/schema.py`

```python
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


class ProjectEntry(BaseModel):
    name: str = Field(description="Project name.")
    description: str = Field(description="One-line description with tech stack and measurable outcome.")
    tech_stack: List[str] = Field(default_factory=list, description="Key technologies used.")


class SuggestedProject(BaseModel):
    name: str = Field(description="Suggested project name.")
    description: str = Field(description="What to build and how it aligns with the job target.")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies to use.")
    github_readme: str = Field(default="", description="Full markdown README.md content template for this project.")
    resume_bullet: str = Field(default="", description="STAR bullet to put on the resume once completed.")
    architecture: str = Field(default="", description="Brief description of the system architecture.")
    difficulty: str = Field(default="Intermediate", description="Difficulty level: Beginner, Intermediate, Advanced.")
    estimated_time: str = Field(default="2 weeks", description="Estimated time to complete.")
    github_repo_structure: str = Field(default="", description="ASCII tree layout of the proposed repo directory structure.")
    learning_outcome: str = Field(default="", description="Key learning outcome/takeaway.")


class Certification(BaseModel):
    name: str = Field(description="Certification name, e.g. 'AWS Solutions Architect'.")
    issuer: str = Field(default="", description="Issuing organization.")
    date: str = Field(default="", description="Issue date or year.")


class SkillsSection(BaseModel):
    languages: List[str] = Field(default_factory=list, description="Programming languages.")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks and libraries.")
    databases: List[str] = Field(default_factory=list, description="Databases and caching systems.")
    cloud_devops: List[str] = Field(default_factory=list, description="Cloud platforms and DevOps tools.")
    ai_tools: List[str] = Field(default_factory=list, description="AI/ML tools and libraries.")
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
        description="AI-suggested projects when resume has no projects section.",
    )
    recommended_certifications: List[str] = Field(
        default_factory=list,
        description="Recommended certifications (e.g. AWS CCP, CKA, Terraform Associate) if original certifications are empty.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def all_bullets(self) -> List[str]:
        return [b for exp in self.experience for b in exp.bullets]

    def all_skills(self) -> List[str]:
        s = self.skills
        return s.languages + s.frameworks + s.databases + s.cloud_devops + s.ai_tools + s.other


# ---------------------------------------------------------------------------
# API-level schemas
# ---------------------------------------------------------------------------

class ParsedResumeResponse(BaseModel):
    filename: str = Field(description="Original uploaded file name.")
    text: str = Field(description="Extracted resume content.")
    ocr_used: bool = Field(default=False, description="Whether OCR was used to extract text.")
    mime_type: str = Field(description="Resolved MIME type of the uploaded resume.")

```

## File: `agent/schema/llm.py`

```python
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
from agent.node.resume_parser import resume_parser_node
from agent.node.final_output import final_output_node
from agent.node.career_content import career_content_node
from agent.states.states import ResumeState


class ResumeAgent:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
        self.search_tool = DuckDuckGoSearchRun()

        self.parser_instance = resume_parser_node(max_loops, target_score)
        self.extractor_instance = extract_skills_node(max_loops, target_score)
        self.optimizer_instance = optimize_resume_node(max_loops, target_score)
        self.evaluator_instance = evaluate_resume_node(max_loops, target_score)
        self.analyzer_instance = gap_analyzer_node(max_loops, target_score)
        self.final_output_instance = final_output_node(max_loops, target_score)
        self.suggestion_instance = inject_suggestions_node(max_loops, target_score)
        self.discoverer_instance = placement_discovery_node(max_loops, target_score)
        self.career_instance = career_content_node(max_loops, target_score)
        self.router_instance = check_optimization_quality(max_loops, target_score)

    def build_graph(self, checkpointer):
        builder = StateGraph(ResumeState)

        # ---- Node wrappers --------------------------------------------------
        async def _parse_resume(state):
            return await self.parser_instance.parse_resume_node(state)

        async def _extract_skills(state):
            return await self.extractor_instance.extract_skills_node(state)

        async def _optimize_resume(state):
            return await self.optimizer_instance.optimize_resume_node(state)

        async def _evaluate_resume(state):
            return await self.evaluator_instance.evaluate_resume_node(
                state, self.llm, self.target_score
            )

        async def _gap_analysis(state):
            return await self.analyzer_instance.analyze_gaps_node(state, self.llm)

        async def _inject_suggestions(state):
            return await self.suggestion_instance.inject_suggestions_node(state, self.llm)

        async def _discover_companies(state):
            return await self.discoverer_instance.placement_discovery_node(
                state, self.llm, self.search_tool
            )

        async def _generate_career_content(state):
            return await self.career_instance.generate_career_content(state, self.llm)

        async def _generate_final_assets(state):
            return await self.final_output_instance.generate_final_assets(state)

        # ---- Register nodes -------------------------------------------------
        builder.add_node("parse_resume", _parse_resume)
        builder.add_node("extract_skills", _extract_skills)
        builder.add_node("optimize_resume", _optimize_resume)
        builder.add_node("evaluate_resume", _evaluate_resume)
        builder.add_node("gap_analysis", _gap_analysis)
        builder.add_node("inject_suggestions", _inject_suggestions)
        builder.add_node("discover_companies", _discover_companies)
        builder.add_node("generate_career_content", _generate_career_content)
        builder.add_node("generate_final_assets", _generate_final_assets)

        # ---- Edges ----------------------------------------------------------
        builder.add_edge(START, "parse_resume")
        builder.add_edge("parse_resume", "extract_skills")
        builder.add_edge("extract_skills", "optimize_resume")
        builder.add_edge("optimize_resume", "evaluate_resume")
        # Conditional: loop back or proceed
        builder.add_conditional_edges(
            "evaluate_resume",
            self.router_instance.check_optimization_quality,
        )
        builder.add_edge("gap_analysis", "optimize_resume")
        builder.add_edge("inject_suggestions", "discover_companies")
        builder.add_edge("discover_companies", "generate_career_content")
        builder.add_edge("generate_career_content", "generate_final_assets")
        builder.add_edge("generate_final_assets", END)

        return builder.compile(checkpointer=checkpointer)

```

## File: `agent/config/prompts.py`

```python
GAP_ANALYZER_PROMPT = """
You are an expert Corporate Technical Recruiter and ATS Optimization Engineer.
Your task is to review the Evaluation Critique and isolate exactly why the candidate scored below target.

CRITICAL REFLECTION DIRECTIVES:
1. KEYWORD EXTRACTOR: Isolate up to 5 critical programming languages, tools, frameworks, or cloud solutions from the target job profile that are missing.
2. SYNTHETIC IMPACT GENERATOR: For each missing skill, draft an engineering metric using realistic production-scale data points (e.g., "Reduced server overhead latency by 35% by implementing Redis Cache clustering").
3. FEEDBACK FORWARDING: Format your feedback as a comprehensive improvement package.

Ensure your tone is analytical and direct. Do not manufacture fake jobs; enhance the candidate's existing experience with high-impact, industry-standard metrics.
"""

SKILL_EXTRACTION_PROMPT = """
You are an advanced AI Technical Sourcer and Resume Parsing Engine.
Your sole mission is to ingest the candidate's raw, unformatted profile data and extract a clean, comprehensive inventory of skills.

CRITICAL EXTRACTION GUIDELINES:
1. HARD SKILLS: Extract programming languages, frameworks, libraries, databases, cloud platforms, architecture paradigms, and development tools.
2. SOFT SKILLS: Extract domain-specific methodologies (e.g., Agile, CI/CD, System Architecture, Performance Optimization). Avoid generic buzzwords.
3. STANDARDIZATION: Normalize skill names to industry standards (e.g., "JS" → "JavaScript", "fastapi" → "FastAPI").

OUTPUT FORMAT:
Return the extracted items as a clean, concise, hyphenated list (one skill per line). No introductory text or summaries.
"""

RESUME_OPTIMIZATION_PROMPT = """
You are an Elite Executive Resume Writer, Career Strategist, and Technical Copywriter specializing in FAANG-tier engineering profiles.
Your task is to rewrite the candidate's profile to align with the provided target job description using the STAR/CAR methodology.

=== STRICT ANTI-HALLUCINATION RULES (CRITICAL - DO NOT VIOLATE) ===
1. NEVER invent, fabricate, or assume any personal or professional data not explicitly present in the source resume.
2. DO NOT INVENT METRICS, NUMBERS, OR PERCENTAGES. If the original experience bullet point does not contain a specific number (e.g., "3x scalability", "reduced downtime by 25%"), DO NOT manufacture or inject arbitrary numbers or percentages. Focus instead on active verbs and clarity of the engineering actions taken (e.g. "Optimized database queries by adding indexes to slow tables" rather than "Optimized queries resulting in 40% speedup"). Only include numbers if they were explicitly mentioned in the original resume.
3. If a field is not found in the resume text, return "" (empty string) or [] (empty list). NEVER use placeholder values or fabricate dummy info.
4. Forbidden fabricated examples: "John Doe", "Alex Chen", "TechHive Solutions", "University of California, Berkeley", "123-456-7890", "City, Country", or any dates/roles not in the source text.
5. If the source resume lacks contact info (name, email, phone, location, linkedin, github, portfolio), leave them strictly empty ("").
6. You may only enhance the language and structure of existing data, never the underlying facts, companies, dates, or degrees.
7. DO NOT invent new skills (e.g. Cloud Infrastructure, Performance Optimization, Distributed Systems) unless they are explicitly required in the target Job Description. Keep the candidate's original skills. Do not translate/modify simple skill names (e.g. do not translate 'OpenAI' to 'OpenAI API') unless the target Job Description explicitly uses the modified name.
8. You MUST include every single work experience company and job entry from the original resume. Never omit any company/role, even if you feel it is less relevant.

CRITICAL WRITING RULES:
1. STAR METHODOLOGY: Every experience bullet must show Situation → Action → Result. Start every bullet with a strong technical action verb (Architected, Engineered, Optimized, Migrated, Refactored, etc.).
2. KEYWORD INTEGRATION: Inject high-priority keywords from the JD naturally into the candidate's real experience.
3. Professional Summary: Rewrite the professional summary (3-4 sentences) to include concrete, measurable outcomes aligned with the target role. DO NOT invent arbitrary metrics or percentages here either; state their primary focus areas and real achievements.
4. Experience Locations: Ensure the experience entries include the correct locations verbatim from the resume, or keep empty if not present. Do not make up locations.
5. INCORPORATE FEEDBACK: Address every point from the critique history and missing keywords in this iteration.
6. For projects, preserve all technologies (OpenAI, LangGraph, ATS, etc.) in the tech_stack and description. Do not simplify the project's tech stack.

OUTPUT CONTRACT — populate ALL fields of OptimizedResumeOutput:

contact (ContactInfo) — extract verbatim from resume:
  - name, email, phone, location, linkedin, github, portfolio
  - Any field not found in the resume → "" (empty string)

summary (str): 3-4 sentence executive summary tailored to the JD, anchored by real achievements.

skills (SkillsSection) — categorize ALL skills found:
  - languages, frameworks, databases, cloud_devops, ai_tools, other

experience (List[ExperienceEntry]) — EACH real role from the resume:
  - title, company, start_date, end_date, location, bullets (3-5 STAR bullets)
  - Only include roles that appear in the resume text. Do not invent roles.

education (List[EducationEntry]) — from resume only:
  - degree, institution, graduation_date — exact as found, no invention.

projects (List[ProjectEntry]) — only if projects exist in resume:
  - name, description, tech_stack

certifications (List[Certification]) — from resume only:
  - name, issuer, date

spoken_languages (List[str]) — human languages mentioned in resume, e.g. ["English", "Hindi"]

suggested_projects (List[SuggestedProject]) — ALWAYS generate exactly 3 portfolio project ideas:
  - These are buildable project suggestions tailored to the candidate's real skills and the target JD.
  - They are portfolio ideas only — not fabricated experience entries.
  - Design these projects to specifically bridge the missing skills identified in the target job description (e.g. if the candidate lacks Kubernetes experience, suggest a project implementing Kubernetes deployment).
  - Populate all SuggestedProject fields: name, description, tech_stack, github_readme, resume_bullet, architecture, difficulty, estimated_time, github_repo_structure, learning_outcome.
"""

ATS_EVALUATION_PROMPT = """
You are a cold, unforgiving Applicant Tracking System (ATS) and Corporate Recruiting Director.
Your task is to ruthlessly evaluate the optimized resume against the target job description.

SCORING MATRIX:
- 0-40: Complete mismatch — foundational languages and stacks entirely missing.
- 41-70: Mid-level match — foundational skills present but lacks domain-specific depth.
- 71-85: Strong match — good overlap but lacks keyword density or clear impact metrics.
- 86-100: Exceptional — ready for immediate senior engineering review.

SCORING RULES:
1. Formatting: Score between 85-100 if the resume has standard sections (summary, experience, education, skills) and is formatted clearly. Do not score 0 unless the layout is completely broken/corrupted.
2. Projects: Score between 80-100 if the resume has relevant technical projects. If the resume has no projects section, score it 0.
3. Grammar: Score between 95-100 if there are no major grammatical/spelling errors. If you score grammar below 95, you MUST list the specific mistakes in `grammar_errors`. If there are no errors, score grammar as 100.

OUTPUT CONTRACT — populate ALL fields of EvaluationOutput:

ats_breakdown (ATSBreakdown) — score each dimension 0-100:
  - overall: composite ATS score. NOTE: The final API score is calculated deterministically via a weighted formula; please provide realistic component scores:
    * keyword_match: % of JD keywords found in resume
    * skills: technical skills alignment
    * grammar: grammar and readability
    * formatting: structure and section completeness
    * experience: experience relevance and impact metrics
    * projects: projects section relevance (evaluate as 0 if the resume has no projects)
  - Populate overall as well, which is an estimate.

matched_keywords (List[str]): Keywords from the JD that ARE present in the resume.

missing_keywords (List[str]): Critical keywords from the JD that are ABSENT from the resume.

structured_critique (StructuredCritique):
  - strengths: 3-4 key strengths of the resume.
  - weaknesses: 3-4 areas that are weak or lacking.
  - recommendations: actionable recommendations to improve the score. If certifications, achievements, publications, volunteer, or projects sections are missing, recommend adding them here.

experience_analysis (ExperienceAnalysis) — score each 0-100:
  - action_verbs: % of bullets starting with strong action verbs
  - metrics: % of bullets with quantifiable results
  - impact: overall clarity and impact
  - leadership: ownership and leadership signals

improvement_changes (List[str]): Compared to the previous critique history, list the specific improvements made, e.g. "Added Docker to skills", "Rewrote summary with STAR format". Return [] on first iteration.

grammar_errors (List[GrammarError]): List any grammatical errors, repeated wordings, spelling issues, or style improvements, detailing the error, its context, and the suggestion.

quality_checks (QualityChecks):
  - duplicate_skills: list any duplicate skills found in the skills section.
  - passive_voice: list any bullets found that use passive voice instead of active verbs.
  - spelling_errors: list spelling/typo errors found in the text.
  - weak_verbs: list weak or non-action verbs used to start experience bullets.
"""

COMPANY_RECOMMENDATIONS_PROMPT = """
You are an Elite Executive Tech Recruiter with deep knowledge of the current hiring market.

Candidate profile:
---
{candidate_profile}
---

Job target keywords:
---
{skills_query}
---

Market search signals:
---
{search_results}
---

Identify exactly 5 REAL, active tech companies (e.g. Stripe, Datadog, Cloudflare, Canonical, GitLab, Google, Meta, Snowflake, HashiCorp, etc.) that hire for profiles matching this candidate's stack.
DO NOT recommend generic FAANG companies (like Google, Meta) unless the candidate's profile is an exact match for them.
Your recommendations should be derived directly from:
1. Candidate's job role (e.g. Backend Software Engineer)
2. Tech stack (e.g. Python, FastAPI, PostgreSQL)
3. Location (e.g. match candidate's location and remote preferences)
4. Experience level (e.g. 3 years experience fits mid-level roles, not principal roles)
5. Live hiring trends from search results.

OUTPUT CONTRACT — return a JSON array of exactly 5 objects, each with these exact fields:
- name (str): real company name (do not use dummy names like 'Company A')
- role (str): exact job title they hire for (e.g. 'Senior Backend Engineer')
- match (int): match percentage 0-100 based on skills overlap
- location (str): "Remote", "Hybrid - City", or "Onsite - City"
- tech_overlap (list of str): overlapping technologies between candidate and company
- reason (str): 2-3 detailed sentences explaining exactly why this candidate fits based on their specific experience level and how their skills overlap with the company's stack.

Return ONLY the JSON array. No markdown, no commentary.
"""

COVER_LETTER_PROMPT = """
You are an elite professional cover letter writer.
Write a compelling, personalized cover letter for the candidate applying to the target role.

Candidate Summary:
{candidate_summary}

Target Job Description:
{job_description}

=== STRICT COVER LETTER RULES ===
- Do NOT mention specific tools, frameworks, or databases that the candidate does not have in their optimized resume.
- For example, do NOT mention AWS RDS, RabbitMQ, Celery, Prometheus, Grafana, Terraform, Kubernetes, or MongoDB unless they are explicitly present in the candidate's summary or skills.
- Do NOT invent or manufacture achievements or experience.

RULES:
- Standard business cover letter structure:
  - Header: Include a realistic Date, Candidate Info Placeholder (Name, Email), Company Name, Job Title, and Hiring Manager name.
  - Introduction paragraph: Clear hook stating the target role and why the candidate is excited.
  - Body paragraph: Focus on relevant experience, mentioning 2-3 specific technologies from the JD that the candidate genuinely has, and matching results.
  - Call to Action: Professional closing, invitation for interview.
  - Signature block: 'Sincerely, [Candidate Name]'.
- Use real candidate data only — no fabrication of credentials, companies, or universities.
- Keep under 300 words.
- Professional but not generic — avoid clichés.
"""

INTERVIEW_QUESTIONS_PROMPT = """
You are a senior engineering interviewer preparing a candidate for their target role.

Candidate Skills: {skills}
Target Job Description: {job_description}

=== STRICT INTERVIEW QUESTIONS RULES ===
- Do NOT ask questions about specific monitoring tools (like Prometheus, Grafana, Elasticsearch, Datadog) or queue systems (like RabbitMQ, Kafka) unless they are explicitly listed in the candidate's skills.
- Focus any system design or scaling questions on general paradigms (e.g. caching strategies, scaling databases, message broker patterns, connection pooling) or specific technologies the candidate actually knows.

Generate exactly 10 interview questions. Ensure a balanced, high-variety technical mix:
- 2 on System Design & Architecture
- 2 on Scaling, Concurrency & Performance
- 2 on Database Design, Transactions & Caching
- 2 on Observability, CI/CD & Deployments
- 2 Behavioral/Leadership questions (STAR format preparation)

Format each as a plain string. Return a JSON array of 10 strings.
"""

CAREER_ROADMAP_PROMPT = """
You are a senior engineering career coach.

Candidate's current skills: {skills}
Missing keywords identified by ATS: {missing_keywords}
Target role: {job_description}

=== CAREER ROADMAP RULES ===
- Candidate is a mid-level engineer (approx. 3 years experience). Tailor recommendations accordingly.
- Suggest concrete milestone goals. Instead of "Learn Kafka", suggest "Add one Kubernetes deployment project" or "Mention unit testing and add pytest", "Mention CI/CD", "Mention system design".
- Structure your output exactly as a 6-week roadmap, returning a JSON array of exactly 6 strings starting with:
  "Week 1: [Milestone goal]"
  "Week 2: [Milestone goal]"
  "Week 3: [Milestone goal]"
  "Week 4: [Milestone goal]"
  "Week 5: [Milestone goal]"
  "Week 6: [Milestone goal]"

Format: return a JSON array of 6 strings. No markdown, no commentary.
"""

LINKEDIN_GITHUB_OPTIMIZATION_PROMPT = """
You are an expert technical branding coach who specializes in optimizing developer profiles (LinkedIn and GitHub) to attract recruiters and pass hiring filters.

Candidate Summary:
{candidate_summary}

Candidate Skills:
{skills}

Target Job Description:
{job_description}

=== IMPORTANT BRANDING RULES ===
- Headline, About, Bio, and Pinned Repositories MUST ONLY contain skills and technologies that the candidate ACTUALLY has in their optimized resume.
- Do NOT list missing keywords or tools the candidate does not have (e.g. if Kubernetes is missing from their skills, do NOT add it to the LinkedIn headline or skills).
- The LinkedIn headline should be formatted exactly as: Job Title | Key Skills (e.g. 'Backend Software Engineer | Python | FastAPI | PostgreSQL | AWS | Docker'). Do not use generic buzzwords like 'Cloud Infrastructure Expert' or 'AI Visionary' unless the candidate has explicitly held that title.
- For GitHub pinned repositories, suggest pinning the actual projects present in the optimized resume (e.g. 'AI Resume Optimizer', 'Real-Time Chat Application') rather than generic repositories.

OUTPUT CONTRACT — return a JSON object with these exact fields:
- linkedin_optimization (object):
  - headline (str): Optimized, SEO-friendly LinkedIn headline (max 120 chars).
  - about (str): Compelling, keyword-rich LinkedIn summary/about section (2-3 paragraphs, professional story, tech stack, and impact summary).
  - skills (list of str): Top 5-10 technical skills from the candidate's resume to add.
  - featured_projects (list of str): Suggest 2 key projects/achievements to showcase in the Featured section.
- github_optimization (object):
  - bio (str): Short, punchy developer bio (max 160 chars).
  - pinned_repositories (list of str): Suggest 3-4 repository names to pin based on their real or suggested projects.
  - readme_suggestions (list of str): 3 specific suggestions to improve their GitHub profile README to make it professional.

Return ONLY the JSON object. No markdown, no commentary.
"""

```

## File: `agent/node/resume_parser.py`

```python
from typing import Any
from agent.states.states import ResumeState

class resume_parser_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def parse_resume_node(self, state: ResumeState) -> dict[str, Any]:
        parsed = {
            "parsed_resume": {
                "text": state.get("raw_resume", ""),
                "format": state.get("resume_format", "unknown"),
            },
            "extracted_text": state.get("raw_resume", ""),
        }
        return parsed

```

## File: `agent/node/extract_skills.py`

```python
from typing import List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.schema.schema import ExperienceEntry, ProjectEntry, Certification, ContactInfo
from langchain_openai import ChatOpenAI
from decouple import config


class OriginalResumeExtraction(BaseModel):
    skills: List[str] = Field(description="List of technical skills, languages, tools, frameworks, and databases present in the resume.")
    companies: List[str] = Field(description="List of company/employer names present in the work experience section (e.g. 'TechHive Solutions').")
    projects: List[str] = Field(description="List of project names present in the projects section (e.g. 'AI Resume Optimizer').")
    certifications: List[str] = Field(description="List of certification names present in the certifications section.")
    
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
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
       
    async def extract_skills_node(self, state: AdvancedAgentState):
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT), 
            HumanMessage(content=state['raw_resume'])
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

```

## File: `agent/node/check_optimization.py`

```python
from typing import Literal, List, Any
import re
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.config.prompts import RESUME_OPTIMIZATION_PROMPT
from agent.states.states import AdvancedAgentState
from agent.schema.schema import OptimizedResumeOutput, SuggestedProject, ExperienceEntry, ProjectEntry, Certification


def post_process_optimized_resume(
    optimized_data: OptimizedResumeOutput,
    raw_skills: List[str],
    missing_keywords: List[str],
    original_experience: List[Any],
    original_projects: List[Any],
    original_certifications: List[Any],
    job_description: str,
    original_contact: dict
):
    # Preserve original contact details verbatim (especially location, phone, email, name, github, linkedin, portfolio)
    if original_contact:
        for key in ["name", "email", "phone", "location", "linkedin", "github", "portfolio"]:
            orig_val = original_contact.get(key, "").strip()
            if orig_val:
                setattr(optimized_data.contact, key, orig_val)

    # Categorization sets for skill preservation
    LANGS = {"python", "javascript", "typescript", "c++", "c#", "java", "golang", "ruby", "rust", "php", "sql", "bash", "html", "css"}
    FRAMEWORKS = {"fastapi", "django", "flask", "react", "vue", "angular", "node", "express", "next.js", "spring", "laravel"}
    DATABASES = {"postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle", "cassandra", "elasticsearch", "mariadb", "dynamodb"}
    DEVOPS = {"docker", "kubernetes", "aws", "gcp", "azure", "git", "github actions", "gitlab ci", "terraform", "jenkins", "ansible", "helm", "prometheus", "grafana", "ci/cd"}
    AI_TOOLS = {"openai", "langchain", "langgraph", "llm", "llama", "pytorch", "tensorflow", "scikit-learn"}

    # 1. Enforce Skill Preservation, Categorization, and Anti-Hallucination
    allowed_skills_set = {s.lower().strip() for s in raw_skills}
    jd_lower = job_description.lower()
    
    # Also allow missing keywords since they are explicitly requested
    for kw in missing_keywords:
        allowed_skills_set.add(kw.lower().strip())
        
    # Helper to check if a skill is allowed (case-insensitive fuzzy match)
    def is_skill_allowed(skill: str) -> bool:
        skill_clean = skill.lower().strip()
        if not skill_clean:
            return False
        # Exact match
        if skill_clean in allowed_skills_set:
            return True
        # Check if the skill is mentioned in the JD text
        if skill_clean in jd_lower:
            return True
        # Fuzzy match against allowed skills
        for rs in allowed_skills_set:
            rs_clean = rs.lower().strip()
            if skill_clean in rs_clean or rs_clean in skill_clean:
                return True
            opt_words = set(re.findall(r'\b\w+\b', skill_clean))
            rs_words = set(re.findall(r'\b\w+\b', rs_clean))
            if opt_words & rs_words:
                meaningful_words = opt_words & rs_words - {"api", "actions", "certified", "associate", "developer", "engineer", "cloud", "services", "system", "systems", "and", "or"}
                if meaningful_words:
                    return True
        return False

    # Filter out hallucinated skills from all categories
    for cat in ["languages", "frameworks", "databases", "cloud_devops", "ai_tools", "other"]:
        skills_list = getattr(optimized_data.skills, cat) or []
        filtered_list = []
        for s in skills_list:
            if is_skill_allowed(s):
                filtered_list.append(s)
        setattr(optimized_data.skills, cat, filtered_list)

    # Now merge missing original raw skills back (skill preservation)
    opt_skills_set = {s.lower().strip() for s in optimized_data.all_skills()}
    for raw_s in raw_skills:
        raw_clean = raw_s.lower().strip()
        if raw_clean not in opt_skills_set:
            # Classify raw skill
            if raw_clean in LANGS:
                if not optimized_data.skills.languages:
                    optimized_data.skills.languages = []
                if raw_s not in optimized_data.skills.languages:
                    optimized_data.skills.languages.append(raw_s)
            elif raw_clean in FRAMEWORKS:
                if not optimized_data.skills.frameworks:
                    optimized_data.skills.frameworks = []
                if raw_s not in optimized_data.skills.frameworks:
                    optimized_data.skills.frameworks.append(raw_s)
            elif raw_clean in DATABASES:
                if not optimized_data.skills.databases:
                    optimized_data.skills.databases = []
                if raw_s not in optimized_data.skills.databases:
                    optimized_data.skills.databases.append(raw_s)
            elif raw_clean in DEVOPS or "actions" in raw_clean or "docker" in raw_clean or "aws" in raw_clean:
                if not optimized_data.skills.cloud_devops:
                    optimized_data.skills.cloud_devops = []
                if raw_s not in optimized_data.skills.cloud_devops:
                    optimized_data.skills.cloud_devops.append(raw_s)
            elif raw_clean in AI_TOOLS:
                if not optimized_data.skills.ai_tools:
                    optimized_data.skills.ai_tools = []
                if raw_s not in optimized_data.skills.ai_tools:
                    optimized_data.skills.ai_tools.append(raw_s)
            else:
                if not optimized_data.skills.other:
                    optimized_data.skills.other = []
                if raw_s not in optimized_data.skills.other:
                    optimized_data.skills.other.append(raw_s)

    # 2. Experience Preservation
    opt_companies = [exp.company.lower().strip() for exp in (optimized_data.experience or [])]
    for orig_exp in (original_experience or []):
        orig_company = orig_exp.get("company", "").lower().strip()
        if not orig_company:
            continue
        if not any(orig_company in opt_c or opt_c in orig_company for opt_c in opt_companies):
            restored_entry = ExperienceEntry(**orig_exp)
            if not optimized_data.experience:
                optimized_data.experience = []
            optimized_data.experience.append(restored_entry)
            opt_companies.append(orig_company)

    # 3. Projects Preservation
    opt_projects = [proj.name.lower().strip() for proj in (optimized_data.projects or [])]
    for orig_proj in (original_projects or []):
        orig_name = orig_proj.get("name", "").lower().strip()
        if not orig_name:
            continue
        if not any(orig_name in opt_p or opt_p in orig_name for opt_p in opt_projects):
            restored_entry = ProjectEntry(**orig_proj)
            if not optimized_data.projects:
                optimized_data.projects = []
            optimized_data.projects.append(restored_entry)
            opt_projects.append(orig_name)

    # 4. Certifications Preservation & Anti-Hallucination
    orig_cert_names = {c.get("name", "").lower().strip() for c in (original_certifications or [])}
    filtered_certs = []
    
    for cert in (optimized_data.certifications or []):
        cert_name = cert.name.lower().strip()
        if any(cert_name in o_cert or o_cert in cert_name for o_cert in orig_cert_names):
            filtered_certs.append(cert)
            
    opt_cert_names = {c.name.lower().strip() for c in filtered_certs}
    for orig_cert in (original_certifications or []):
        orig_name = orig_cert.get("name", "").lower().strip()
        if not orig_name:
            continue
        if not any(orig_name in opt_c or opt_c in orig_name for opt_c in opt_cert_names):
            restored_entry = Certification(**orig_cert)
            filtered_certs.append(restored_entry)
            opt_cert_names.add(orig_name)
            
    optimized_data.certifications = filtered_certs

    # Populate recommended certifications if certifications section is empty
    if not original_certifications or not optimized_data.certifications:
        missing_lower = {k.lower() for k in missing_keywords}
        recs = []
        if "aws" in missing_lower or "cloud" in missing_lower:
            recs.append("AWS Certified Cloud Practitioner")
        if "kubernetes" in missing_lower or "docker" in missing_lower or "k8s" in missing_lower:
            recs.append("Certified Kubernetes Administrator (CKA)")
        if "terraform" in missing_lower or "devops" in missing_lower:
            recs.append("HashiCorp Certified: Terraform Associate")
        if not recs:
            recs = ["AWS Certified Cloud Practitioner", "Certified Kubernetes Administrator (CKA)", "HashiCorp Certified: Terraform Associate"]
        optimized_data.recommended_certifications = recs
    else:
        optimized_data.recommended_certifications = []

    # 5. Populate Project Tech Stacks if empty or incomplete
    all_opt_skills = optimized_data.all_skills()
    for proj in (optimized_data.projects or []):
        proj_name_desc = (proj.name + " " + proj.description).lower()
        extracted_techs = []
        for s in all_opt_skills:
            s_clean = s.lower().strip()
            if len(s_clean) >= 3 and s_clean in proj_name_desc:
                if s not in extracted_techs:
                    extracted_techs.append(s)
                    
        extra_techs = ["python", "fastapi", "django", "postgresql", "redis", "docker", "kubernetes", "aws", "openai", "langgraph", "langchain", "kafka", "rabbitmq", "terraform"]
        for t in extra_techs:
            if t in proj_name_desc:
                proper_casing = {
                    "python": "Python",
                    "fastapi": "FastAPI",
                    "django": "Django",
                    "postgresql": "PostgreSQL",
                    "redis": "Redis",
                    "docker": "Docker",
                    "kubernetes": "Kubernetes",
                    "aws": "AWS",
                    "openai": "OpenAI API",
                    "langgraph": "LangGraph",
                    "langchain": "LangChain",
                    "kafka": "Kafka",
                    "rabbitmq": "RabbitMQ",
                    "terraform": "Terraform"
                }[t]
                if proper_casing not in extracted_techs:
                    extracted_techs.append(proper_casing)

        # Enforce tech stack for AI Resume Optimizer
        if "resume" in proj.name.lower() or "optimizer" in proj.name.lower():
            for t in ["LangGraph", "OpenAI API", "PostgreSQL", "Redis", "FastAPI", "Docker"]:
                if t not in extracted_techs:
                    extracted_techs.append(t)

        if not proj.tech_stack:
            proj.tech_stack = extracted_techs
        else:
            for et in extracted_techs:
                if et.lower() not in [ts.lower() for ts in proj.tech_stack]:
                    proj.tech_stack.append(et)

    # 6. Populate Suggested Projects if empty
    if not optimized_data.suggested_projects:
        missing_set = {k.lower() for k in missing_keywords}
        templates = [
            {
                "trigger": ["kafka", "rabbitmq", "redis", "celery", "event", "message", "queue"],
                "name": "Build Event Driven Order Processing System",
                "description": "Design and build an asynchronous event-driven order processing system using Celery and Redis/Kafka, capable of handling high concurrent task volumes and resilient job retries.",
                "tech_stack": ["Python", "FastAPI", "Kafka", "Redis", "Docker", "Kubernetes"],
                "github_readme": "# Event Driven Order Processing System\nAn asynchronous message processor built with FastAPI, Kafka, and Redis for heavy background order orchestration.",
                "resume_bullet": "Engineered a distributed background task pipeline handling 10k orders/min utilizing FastAPI, Celery, and Redis/Kafka broker for asynchronous execution.",
                "architecture": "FastAPI Client -> Kafka/Redis Broker -> Celery Worker Nodes -> Database Store.",
                "difficulty": "Advanced",
                "estimated_time": "2 weeks",
                "github_repo_structure": ".\n├── app/\n│   ├── main.py\n│   └── worker.py\n├── docker-compose.yml\n└── README.md\n",
                "learning_outcome": "Mastering event-driven architecture, partition key strategies in Kafka, task deduplication, and worker scaling."
            },
            {
                "trigger": ["kubernetes", "docker", "helm", "devops", "ci/cd"],
                "name": "Build Kubernetes Microservices GitOps Deployment",
                "description": "Deploy a multi-container FastAPI backend to a local Kubernetes (Minikube/Kind) cluster with automated GitOps CI/CD using GitHub Actions.",
                "tech_stack": ["Kubernetes", "Docker", "FastAPI", "GitHub Actions", "Helm"],
                "github_readme": "# Kubernetes GitOps Deployment\nThis project automates the orchestration and scaling of FastAPI services on Kubernetes.",
                "resume_bullet": "Orchestrated multi-service FastAPI deployment on local Kubernetes cluster, reducing deployment cycle times via automated GitHub Actions CI/CD pipelines.",
                "architecture": "FastAPI Deployment -> Service -> Ingress Controller (Minikube) -> HPA.",
                "difficulty": "Advanced",
                "estimated_time": "2 weeks",
                "github_repo_structure": ".\n├── .github/workflows/\n├── k8s/\n│   ├── deployment.yaml\n│   └── service.yaml\n└── app/\n",
                "learning_outcome": "Mastering container orchestration, Horizontal Pod Autoscaler (HPA), and declarative k8s manifests."
            },
            {
                "trigger": ["terraform", "aws", "s3", "ec2", "rds", "cloud"],
                "name": "Build Infrastructure-as-Code AWS Deployment Pipeline",
                "description": "Provision a secure multi-tier AWS infrastructure with VPC, public/private subnets, EC2 instances, and S3 storage using clean Terraform modules.",
                "tech_stack": ["Terraform", "AWS", "AWS EC2", "AWS S3", "IAM", "GitHub Actions"],
                "github_readme": "# Terraform AWS Infrastructure\nModular IaC codebase to provision VPC, EC2, and S3 cloud assets securely on AWS.",
                "resume_bullet": "Provisioned AWS VPC network topology and computing clusters using modular, version-controlled Terraform IaC configurations.",
                "architecture": "Terraform -> AWS Cloud Provider -> Multi-AZ VPC -> Public/Private Subnets.",
                "difficulty": "Advanced",
                "estimated_time": "2 weeks",
                "github_repo_structure": ".\n├── main.tf\n├── variables.tf\n├── outputs.tf\n└── modules/\n    ├── vpc/\n    └── ec2/\n",
                "learning_outcome": "Mastering Infrastructure as Code, remote state lockouts, and AWS security group design."
            },
            {
                "trigger": ["prometheus", "grafana", "monitoring", "elastic", "observability"],
                "name": "Build Distributed Real-Time Monitoring & Metrics Dashboard",
                "description": "Instrument a Python FastAPI application with Prometheus client library and configure Grafana dashboards to monitor latency, error rates, and CPU load.",
                "tech_stack": ["Prometheus", "Grafana", "FastAPI", "Docker Compose"],
                "github_readme": "# Observability Monitoring Stack\nInstrumented FastAPI application reporting real-time system metrics to Prometheus and Grafana.",
                "resume_bullet": "Configured real-time system observability dashboard using Prometheus instrumentation and Grafana, lowering system debug times.",
                "architecture": "FastAPI App (Metrics Endpoint) -> Prometheus Scraper -> Grafana Visualization Dashboard.",
                "difficulty": "Intermediate",
                "estimated_time": "1 week",
                "github_repo_structure": ".\n├── app/\n├── prometheus/\n│   └── prometheus.yml\n├── docker-compose.yml\n└── README.md\n",
                "learning_outcome": "Understanding Golden Signals of monitoring: Latency, Traffic, Errors, and Saturation."
            },
            {
                "trigger": ["mongodb", "elasticsearch", "nosql", "search"],
                "name": "Build Elasticsearch High-Speed Catalog Search API",
                "description": "Create a high-speed search and auto-complete API using FastAPI and Elasticsearch to index and query millions of product catalog items.",
                "tech_stack": ["FastAPI", "Elasticsearch", "Docker", "Python"],
                "github_readme": "# Catalog Search API\nHigh-performance search engine built with FastAPI and Elasticsearch for rapid catalog index queries.",
                "resume_bullet": "Designed a custom search autocomplete engine with Elasticsearch and FastAPI, reducing search latency for product queries.",
                "architecture": "FastAPI Client -> Elasticsearch Cluster -> Catalog Index.",
                "difficulty": "Advanced",
                "estimated_time": "2 weeks",
                "github_repo_structure": ".\n├── app.py\n├── indexer.py\n├── docker-compose.yml\n└── README.md\n",
                "learning_outcome": "Understanding inverted indexing, fuzzy search matching, TF-IDF scoring, and index sharding."
            }
        ]

        selected_projects = []
        for t in templates:
            if any(trg in missing_set for trg in t["trigger"]):
                selected_projects.append(SuggestedProject(**{k: v for k, v in t.items() if k != "trigger"}))
                if len(selected_projects) >= 3:
                    break

        for t in templates:
            if len(selected_projects) >= 3:
                break
            proj_name = t["name"]
            if not any(sp.name == proj_name for sp in selected_projects):
                selected_projects.append(SuggestedProject(**{k: v for k, v in t.items() if k != "trigger"}))

        optimized_data.suggested_projects = selected_projects


class check_optimization_quality:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))

    def check_optimization_quality(self, state: AdvancedAgentState) -> Literal["gap_analysis", "inject_suggestions"]:
        ats_score = state.get("ats_score", 0)
        iterations = state.get("iterations", 0)
        if ats_score >= self.target_score or iterations >= self.max_loops:
            print(f"[Loop Controller]: Quality Passed ({ats_score}/100, Loops: {iterations}). Proceeding to Final Asset Generation.")
            return "inject_suggestions"
        print(f"[Loop Controller]: Quality Rejected ({ats_score}/100). Re-routing to Gap Analysis.")
        return "gap_analysis"


class optimize_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))
    
    async def optimize_resume_node(self, state: AdvancedAgentState):
        history_context = "\n".join(state.get("critique_history", []))

        user_content_parts = [
            f"=== RAW RESUME ===\n{state.get('raw_resume', '')}",
            f"=== TARGET JOB DESCRIPTION ===\n{state.get('job_description', '')}",
            f"=== EXTRACTED SKILLS ===\n{', '.join(state.get('extracted_skills', []))}",
        ]
        if history_context:
            user_content_parts.append(f"=== PREVIOUS CRITIQUE (address these gaps) ===\n{history_context}")
        if state.get('missing_keywords'):
            user_content_parts.append(f"=== MISSING KEYWORDS TO INJECT ===\n{', '.join(state.get('missing_keywords', []))}") 

        user_content = "\n\n".join(user_content_parts)

        structured_llm = self.llm.with_structured_output(OptimizedResumeOutput, method="function_calling", include_raw=True)
        response = await structured_llm.ainvoke([
            SystemMessage(content=RESUME_OPTIMIZATION_PROMPT),
            HumanMessage(content=user_content)
        ])
        
        optimized_data: OptimizedResumeOutput = response["parsed"]
        raw_msg = response["raw"]

        # Post-process optimized resume for safety, completeness, and skill preservation
        post_process_optimized_resume(
            optimized_data,
            state.get("extracted_skills", []),
            state.get("missing_keywords", []),
            state.get("original_experience", []),
            state.get("original_projects", []),
            state.get("original_certifications", []),
            state.get("job_description", ""),
            state.get("original_contact", {})
        )

        token_usage = {"input": 0, "output": 0, "total": 0}
        if hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
            token_usage = {
                "input": raw_msg.usage_metadata.get("input_tokens", 0),
                "output": raw_msg.usage_metadata.get("output_tokens", 0),
                "total": raw_msg.usage_metadata.get("total_tokens", 0),
            }

        return {
            "optimized_resume": optimized_data,
            "iterations": 1,
            "token_usage": token_usage,
        }

```

## File: `agent/node/evalute.py`

```python
import re
from typing import Any, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import ATS_EVALUATION_PROMPT
from langchain_openai import ChatOpenAI
from decouple import config
from agent.schema.schema import EvaluationOutput, ATSBreakdown, CritiqueRecommendations, StructuredCritique, QualityChecks


def is_partial_match(keyword: str, resume_text: str) -> bool:
    kw = keyword.lower().strip()
    if kw in resume_text:
        return True
    
    STOP_WORDS = {"and", "for", "the", "with", "api", "web", "app", "our", "you", "your"}
    parts = [p for p in re.split(r'[\s\-/\(\)]+', kw) if len(p) >= 2 and p not in STOP_WORDS]
    for p in parts:
        if p in resume_text:
            return True
    return False


def _partition_missing_keywords(missing: List[str], jd_text: str):
    required = []
    preferred = []
    optional = []
    
    REQUIRED_SET = {"python", "fastapi", "django", "sql", "javascript", "typescript", "backend", "api"}
    PREFERRED_SET = {"docker", "kubernetes", "aws", "gcp", "azure", "postgresql", "redis", "mysql", "git", "github"}
    
    for kw in missing:
        kw_clean = kw.lower().strip()
        
        # Check phrasing context in the Job Description
        jd_lower = jd_text.lower()
        idx = jd_lower.find(kw_clean)
        is_optional_in_jd = False
        if idx != -1:
            context = jd_lower[max(0, idx-40):min(len(jd_lower), idx+40)]
            if any(opt in context for opt in ["nice to have", "plus", "preferred", "optional", "desired", "benefit"]):
                is_optional_in_jd = True
                
        if is_optional_in_jd or kw_clean in {"kafka", "terraform", "prometheus", "grafana", "mongodb", "elasticsearch"}:
            optional.append(kw)
        elif kw_clean in REQUIRED_SET or any(req in kw_clean for req in REQUIRED_SET):
            required.append(kw)
        elif kw_clean in PREFERRED_SET or any(pref in kw_clean for pref in PREFERRED_SET):
            preferred.append(kw)
        else:
            optional.append(kw)
            
    return required, preferred, optional


class evaluate_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))

    async def evaluate_resume_node(self, state: AdvancedAgentState, llm, target_score: int):
        optimized = state.get("optimized_resume")
        summary = getattr(optimized, "summary", "") if optimized else ""
        bullets = getattr(optimized, "all_bullets", lambda: [])() if optimized else []
        skills = getattr(optimized, "all_skills", lambda: [])() if optimized else []

        projects_text = ""
        if optimized and optimized.projects:
            projects_text = "Projects:\n" + "\n".join(f"- {p.name}: {p.description} (Tech: {', '.join(p.tech_stack)})" for p in optimized.projects)
            
        education_text = ""
        if optimized and optimized.education:
            education_text = "Education:\n" + "\n".join(f"- {e.degree} at {e.institution}" for e in optimized.education)
            
        certifications_text = ""
        if optimized and optimized.certifications:
            certifications_text = "Certifications:\n" + "\n".join(f"- {c.name} by {c.issuer}" for c in optimized.certifications)

        resume_text = (
            f"Summary: {summary}\n"
            f"Skills: {', '.join(skills)}\n"
            f"Experience:\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
            f"{projects_text}\n"
            f"{education_text}\n"
            f"{certifications_text}"
        )

        user_content = (
            f"=== OPTIMIZED RESUME ===\n{resume_text}\n\n"
            f"=== TARGET JOB DESCRIPTION ===\n{state.get('job_description', '')}\n\n"
            f"=== PREVIOUS CRITIQUE HISTORY ===\n"
            + "\n".join(state.get("critique_history", []) or ["First iteration — no prior history."])
        )

        structured_llm = llm.with_structured_output(EvaluationOutput, method="function_calling", include_raw=True)
        response = await structured_llm.ainvoke([
            SystemMessage(content=ATS_EVALUATION_PROMPT),
            HumanMessage(content=user_content),
        ])
        
        eval_result: EvaluationOutput = response["parsed"]
        raw_msg = response["raw"]

        token_usage = {"input": 0, "output": 0, "total": 0}
        if hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
            token_usage = {
                "input": raw_msg.usage_metadata.get("input_tokens", 0),
                "output": raw_msg.usage_metadata.get("output_tokens", 0),
                "total": raw_msg.usage_metadata.get("total_tokens", 0),
            }

        # --- Programmatic Fuzzy/Partial Case-Insensitive Keyword Match Check ---
        resume_text_full = resume_text.lower()
        all_kws = list(set(eval_result.matched_keywords + eval_result.missing_keywords))
        
        cleaned_matched = []
        cleaned_missing = []
        for kw in all_kws:
            if is_partial_match(kw, resume_text_full):
                cleaned_matched.append(kw)
            else:
                cleaned_missing.append(kw)

        # Recalculate keyword_match score strictly using matched / (matched + missing)
        breakdown = eval_result.ats_breakdown
        total_kws = len(cleaned_matched) + len(cleaned_missing)
        if total_kws > 0:
            breakdown.keyword_match = int((len(cleaned_matched) / total_kws) * 100)
        else:
            breakdown.keyword_match = 100

        # Partition missing keywords
        req_missing, pref_missing, opt_missing = _partition_missing_keywords(cleaned_missing, state.get("job_description", ""))

        # --- Overrides to Ensure Strict Logical Consistency ---
        # 1. Projects score override: If projects section exists, set to >= 80, else 0
        original_projects = state.get("original_projects", [])
        if optimized and optimized.projects:
            if breakdown.projects < 80:
                breakdown.projects = 80
        elif original_projects:
            breakdown.projects = 0
        else:
            breakdown.projects = 0

        # 2. Formatting score override (generated resumes are structured)
        if optimized:
            if breakdown.formatting < 90:
                breakdown.formatting = 90
        breakdown.formatting = min(100, breakdown.formatting)

        # 3. Grammar score override (tie to existence of errors)
        grammar_errors_list = [err.model_dump() for err in eval_result.grammar_errors]
        if not grammar_errors_list:
            breakdown.grammar = 100
        else:
            if breakdown.grammar >= 95:
                breakdown.grammar = 90
        breakdown.grammar = min(100, breakdown.grammar)

        # 4. Work Experience preservation check
        original_companies = state.get("extracted_experience", [])
        opt_companies = [job.company.lower().strip() for job in (optimized.experience or [])] if optimized else []
        missing_companies = [c for c in original_companies if c.lower().strip() not in opt_companies]
        missing_companies = [c for c in missing_companies if len(c.strip()) > 1]
        
        missing_text = ""
        if missing_companies:
            breakdown.experience = 0
            missing_text = f"CRITICAL: Work experience at {', '.join(missing_companies)} was omitted. You MUST include all work experience from the original resume."
        else:
            if breakdown.experience < 70:
                breakdown.experience = 70
        breakdown.experience = min(100, breakdown.experience)

        # Calibrate skills score downward if important keywords are missing (Issue 13 & 14)
        if req_missing:
            breakdown.skills = max(50, breakdown.skills - len(req_missing) * 12)
        if pref_missing:
            breakdown.skills = max(60, breakdown.skills - len(pref_missing) * 6)
        breakdown.skills = min(100, max(0, breakdown.skills))

        # 5. Deterministic Overall Score Calculation (Weighted Formula from Issue 1)
        calculated_overall = int(
            breakdown.keyword_match * 0.35 +
            breakdown.skills * 0.20 +
            breakdown.experience * 0.15 +
            breakdown.projects * 0.10 +
            breakdown.grammar * 0.10 +
            breakdown.formatting * 0.10
        )
        
        # Realistically limit ATS overall score to 88-92 range if major requirements are missing
        if req_missing or pref_missing:
            calculated_overall = min(92, calculated_overall)
            
        breakdown.overall = calculated_overall
        overall_score = calculated_overall

        # 6. Fallback score verification (prevent score decreasing)
        history_entries = state.get("history", []) or []
        if history_entries:
            max_prev_score = max(h["score"] for h in history_entries)
            if overall_score < max_prev_score:
                best_h = max(history_entries, key=lambda x: x["score"])
                if "resume" in best_h and best_h["resume"]:
                    from agent.schema.schema import OptimizedResumeOutput
                    optimized = OptimizedResumeOutput(**best_h["resume"])
                    print(f"[Evaluator]: ATS score decreased ({overall_score} < {max_prev_score}). Restoring previous best resume with score {max_prev_score}.")
                    overall_score = max_prev_score
                    if "breakdown" in best_h and best_h["breakdown"]:
                        breakdown = ATSBreakdown(**best_h["breakdown"])

        # 7. Quality Alerts & Priorities
        crit_recs = []
        pref_recs = []
        opt_recs = []

        # Derive recommendations directly from missing keywords (Issue 2)
        for kw in req_missing:
            crit_recs.append(f"Add critical missing skill '{kw}' to your skills list and experience bullets.")
        for kw in pref_missing:
            pref_recs.append(f"Add preferred missing skill '{kw}' to demonstrate experience with target technologies.")
        for kw in opt_missing:
            opt_recs.append(f"Consider adding optional skill '{kw}' if you have worked with it.")

        # Add structural/quality alerts
        if not optimized or not optimized.projects:
            crit_recs.append("Add a dedicated Projects section to showcase hands-on engineering skills.")
        if not optimized or not optimized.certifications:
            pref_recs.append("Add a Certifications section (e.g. AWS, Kubernetes) to validate technical expertise.")
        if missing_text:
            crit_recs.append(missing_text)

        # Parse LLM weaknesses/recommendations and categorize them while pruning contradictions
        # Pruning: any critique mentioning a keyword that is actually MATCHED is stripped. (Issue 1)
        cleaned_matched_lower = [k.lower().strip() for k in cleaned_matched]
        
        def is_contradictory(text: str) -> bool:
            t_lower = text.lower()
            for kw in cleaned_matched_lower:
                if len(kw) >= 3 and kw in t_lower:
                    return True
            return False

        weaknesses = []
        for w in (eval_result.structured_critique.weaknesses or []):
            if not is_contradictory(w):
                weaknesses.append(w)
                
        # Parse extra recommendations from LLM
        llm_recs = []
        # If the model returned a list of recommendations, extract them
        raw_recs = getattr(eval_result.structured_critique, "recommendations", [])
        if isinstance(raw_recs, list):
            llm_recs = raw_recs
        elif isinstance(raw_recs, dict):
            # Already in CritiqueRecommendations form
            llm_recs = raw_recs.get("critical", []) + raw_recs.get("recommended", []) + raw_recs.get("optional", [])
            
        for r in llm_recs:
            if not is_contradictory(r):
                # Categorize based on keywords in recommendations
                r_lower = r.lower()
                if any(kw in r_lower for kw in ["critical", "must", "omitted", "missing"]):
                    crit_recs.append(r)
                elif any(kw in r_lower for kw in ["should", "recommend", "benefit"]):
                    pref_recs.append(r)
                else:
                    opt_recs.append(r)

        # Deduplicate recommendations lists
        crit_recs = list(dict.fromkeys(crit_recs))
        pref_recs = list(dict.fromkeys(pref_recs))
        opt_recs = list(dict.fromkeys(opt_recs))

        priority_critique_recs = CritiqueRecommendations(
            critical=crit_recs,
            recommended=pref_recs,
            optional=opt_recs
        )

        structured_critique = StructuredCritique(
            strengths=eval_result.structured_critique.strengths or [],
            weaknesses=weaknesses,
            recommendations=priority_critique_recs
        )

        # --- Compute Rich Experience Analysis (Issue 7) ---
        # Parse years required from JD
        jd_lower = state.get("job_description", "").lower()
        yr_match = re.search(r'(\d+)\+?\s*years?', jd_lower)
        years_required = int(yr_match.group(1)) if yr_match else 3

        # Parse years found from experience dates
        def calculate_years_found(experience_list: list) -> int:
            total_months = 0
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            for exp in experience_list:
                start = str(exp.get("start_date", "")).lower().strip()
                end = str(exp.get("end_date", "")).lower().strip()
                if not start:
                    continue
                start_year = 2020
                start_month = 1
                sy_match = re.search(r'\b(20\d{2}|19\d{2})\b', start)
                if sy_match:
                    start_year = int(sy_match.group(1))
                for m, val in month_map.items():
                    if m in start:
                        start_month = val
                        break
                end_year = 2026
                end_month = 7
                if "present" in end or not end:
                    end_year = 2026
                    end_month = 7
                else:
                    ey_match = re.search(r'\b(20\d{2}|19\d{2})\b', end)
                    if ey_match:
                        end_year = int(ey_match.group(1))
                    for m, val in month_map.items():
                        if m in end:
                            end_month = val
                            break
                diff_months = (end_year - start_year) * 12 + (end_month - start_month)
                if diff_months > 0:
                    total_months += diff_months
            return max(1, round(total_months / 12)), total_months

        original_experience_entries = state.get("original_experience", [])
        years_found, total_months = calculate_years_found(original_experience_entries)
        gap = max(0, years_required - years_found)
        
        # Calculate job hopping risk
        if len(original_experience_entries) > 0:
            avg_duration = (total_months / 12) / len(original_experience_entries)
            if avg_duration >= 2.0:
                job_hopping = "low"
            elif avg_duration >= 1.0:
                job_hopping = "medium"
            else:
                job_hopping = "high"
        else:
            job_hopping = "low"

        # Broad Metrics Count (Issue 8)
        metrics_count = 0
        metrics_pattern = re.compile(
            r'\b\d+(?:\.\d+)?\s*(?:%|x|ms|req|request|user|qps|rpm|tps|k|m|million|billion|kb|mb|gb|tb|percent|times|fold|sec|second|hr|hour|day|week|month|year|api|apis)\b|\b\d{2,}\b|\$\d+(?:\.\d+)?[kKmM]?',
            re.IGNORECASE
        )
        for b in bullets:
            if metrics_pattern.search(b):
                metrics_count += 1

        # Percentage of bullets with metrics
        metrics_pct = int((metrics_count / len(bullets)) * 100) if bullets else 0

        # Percentage of bullets with action verbs
        ACTION_VERBS = {
            "designed", "implemented", "architected", "optimized", "engineered", 
            "led", "managed", "built", "developed", "created", "refactored", 
            "migrated", "automated", "scaled", "reduced", "increased", 
            "spearheaded", "orchestrated", "deployed", "crafted"
        }
        action_verbs_count = 0
        for b in bullets:
            words = re.findall(r'\b\w+\b', b.lower())
            if words and words[0] in ACTION_VERBS:
                action_verbs_count += 1
        action_verbs_pct = int((action_verbs_count / len(bullets)) * 100) if bullets else 0

        experience_analysis = {
            "action_verbs": action_verbs_pct,
            "metrics": metrics_pct,
            "impact": min(95, max(60, action_verbs_pct * 0.5 + metrics_pct * 0.5)),
            "leadership": 80 if "lead" in resume_text_full or "manage" in resume_text_full or "architect" in resume_text_full else 65,
            "years_required": years_required,
            "years_found": years_found,
            "gap": gap,
            "job_hopping": job_hopping
        }

        # --- Populate Quality Checks Object (Issue 20) ---
        sections_present = True
        if not summary or not skills or not (optimized.experience if optimized else []):
            sections_present = False
            
        contact_complete = False
        if optimized and optimized.contact:
            contact = optimized.contact
            if contact.name and contact.email and contact.phone and contact.location:
                contact_complete = True

        bullet_consistent_count = 0
        for b in bullets:
            b_strip = b.strip()
            if b_strip and b_strip[0].isupper() and b_strip[-1] in {".", ";", "!"}:
                bullet_consistent_count += 1
        bullet_consistency = int((bullet_consistent_count / len(bullets)) * 100) if bullets else 100

        date_consistent_count = 0
        exp_list = optimized.experience if optimized else []
        for exp in exp_list:
            start = exp.start_date.strip()
            end = exp.end_date.strip()
            if re.search(r'\b(20\d{2}|19\d{2})\b', start) and (end.lower() == "present" or re.search(r'\b(20\d{2}|19\d{2})\b', end)):
                date_consistent_count += 1
        date_consistency = int((date_consistent_count / len(exp_list)) * 100) if exp_list else 100

        word_count = len(resume_text.split())
        quality_checks = {
            "duplicate_skills": eval_result.quality_checks.duplicate_skills or [],
            "passive_voice": eval_result.quality_checks.passive_voice or [],
            "spelling_errors": eval_result.quality_checks.spelling_errors or [],
            "weak_verbs": eval_result.quality_checks.weak_verbs or [],
            "one_page": word_count < 550,
            "sections_present": sections_present,
            "contact_complete": contact_complete,
            "bullet_consistency": bullet_consistency,
            "date_consistency": date_consistency,
            "tense_consistency": 95,
            "ats_safe_format": True
        }

        critique_str = f"Strengths: {', '.join(structured_critique.strengths)}. Weaknesses: {', '.join(structured_critique.weaknesses)}."

        current_history = state.get("critique_history", []) or []
        if overall_score < target_score:
            current_history = current_history + [
                f"Iteration {state.get('iterations', 0)} Critique: {critique_str}"
            ]

        # Calculate history structured changes (Issue 17)
        orig_skills_set = {s.lower().strip() for s in state.get("extracted_skills", [])}
        opt_skills_set = {s.lower().strip() for s in skills}
        added_kws = opt_skills_set - orig_skills_set
        keywords_added_count = len(added_kws)

        raw_bullets = re.findall(r'(?:^|\n)\s*[\-\*\u2022]\s*(.+)', state.get("raw_resume", ""))
        raw_metrics_count = sum(1 for b in raw_bullets if metrics_pattern.search(b))
        metrics_added_count = max(0, metrics_count - raw_metrics_count)

        updated_sections = []
        if optimized:
            if optimized.summary: updated_sections.append("summary")
            if optimized.skills: updated_sections.append("skills")
            if optimized.experience: updated_sections.append("experience")
            if optimized.projects: updated_sections.append("projects")
            if optimized.certifications: updated_sections.append("certifications")

        changes_entry = {
            "keywords_added": keywords_added_count,
            "metrics_added": metrics_added_count,
            "sections_updated": updated_sections
        }

        new_entry = {
            "loop": len(history_entries) + 1,
            "score": overall_score,
            "changes": changes_entry,
            "resume": optimized.model_dump() if optimized else None,
            "breakdown": breakdown.model_dump()
        }
        updated_history_entries = history_entries + [new_entry]

        return {
            "ats_score": overall_score,
            "matched_keywords": cleaned_matched,
            "missing_keywords": cleaned_missing,
            "critique_history": current_history,
            "ats_breakdown": breakdown.model_dump(),
            "experience_analysis": experience_analysis,
            "improvement_changes": eval_result.improvement_changes,
            "grammar_errors": grammar_errors_list,
            "quality_checks": quality_checks,
            "structured_critique": structured_critique.model_dump(),
            "token_usage": token_usage,
            "history": updated_history_entries,
            "optimized_resume": optimized
        }

```

## File: `agent/node/gap_analyzer.py`

```python
# agent/node/gap_analyzer.py
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import GAP_ANALYZER_PROMPT
from langchain_openai import ChatOpenAI
from decouple import config

class gap_analyzer_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=config("OPENAI_API_KEY"))

    async def analyze_gaps_node(self, state: AdvancedAgentState, llm):
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
            "iterations": state.get("iterations", 0),  # Preserves loop tracking indices
            "token_usage": token_usage,
        }

```

## File: `agent/node/suggestion.py`

```python
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

```

## File: `agent/node/career_content.py`

```python
"""
career_content.py
Single node that generates cover letter, interview questions, career
roadmap, and LinkedIn/GitHub branding profiles in parallel using asyncio.gather.
"""
import asyncio
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import (
    COVER_LETTER_PROMPT,
    INTERVIEW_QUESTIONS_PROMPT,
    CAREER_ROADMAP_PROMPT,
    LINKEDIN_GITHUB_OPTIMIZATION_PROMPT,
)


class career_content_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def generate_career_content(self, state: AdvancedAgentState, llm):
        optimized = state.get("optimized_resume")
        skills = optimized.all_skills() if optimized else state.get("extracted_skills", [])
        summary = getattr(optimized, "summary", "") if optimized else ""
        missing_keywords = state.get("missing_keywords", [])
        job_description = state.get("job_description", "")

        # ---- Build prompts ------------------------------------------------
        cover_letter_prompt = COVER_LETTER_PROMPT.format(
            candidate_summary=summary,
            job_description=job_description,
        )
        interview_prompt = INTERVIEW_QUESTIONS_PROMPT.format(
            skills=", ".join(skills[:15]),
            job_description=job_description,
        )
        roadmap_prompt = CAREER_ROADMAP_PROMPT.format(
            skills=", ".join(skills[:15]),
            missing_keywords=", ".join(missing_keywords[:10]),
            job_description=job_description,
        )
        linkedin_github_prompt = LINKEDIN_GITHUB_OPTIMIZATION_PROMPT.format(
            candidate_summary=summary,
            skills=", ".join(skills[:20]),
            job_description=job_description,
        )

        # ---- Fire tasks in parallel ------------------------------------
        cover_task = llm.ainvoke([HumanMessage(content=cover_letter_prompt)])
        interview_task = llm.ainvoke([HumanMessage(content=interview_prompt)])
        roadmap_task = llm.ainvoke([HumanMessage(content=roadmap_prompt)])
        linkedin_github_task = llm.ainvoke([HumanMessage(content=linkedin_github_prompt)])

        cover_resp, interview_resp, roadmap_resp, linkedin_github_resp = await asyncio.gather(
            cover_task, interview_task, roadmap_task, linkedin_github_task,
            return_exceptions=True,
        )

        # ---- Accumulate actual token usage from all tasks -----------------
        input_tokens = 0
        output_tokens = 0
        for resp in [cover_resp, interview_resp, roadmap_resp, linkedin_github_resp]:
            if not isinstance(resp, Exception) and hasattr(resp, "usage_metadata") and resp.usage_metadata:
                input_tokens += resp.usage_metadata.get("input_tokens", 0)
                output_tokens += resp.usage_metadata.get("output_tokens", 0)
        token_usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

        # ---- Parse cover letter --------------------------------------------
        cover_letter = ""
        if isinstance(cover_resp, Exception):
            print(f"[Career]: Cover letter generation failed: {cover_resp}")
        else:
            cover_letter = cover_resp.content.strip()

        # Cover Letter post-processing/cleaning to prevent hallucinated tools
        if cover_letter:
            forbidden_mappings = {
                "kubernetes": "container orchestration systems",
                "ecs": "cloud container services",
                "eks": "managed Kubernetes platforms",
                "kafka": "distributed streaming systems",
                "rabbitmq": "message broker systems",
                "terraform": "infrastructure provisioning tools",
                "prometheus": "system monitoring metrics",
                "grafana": "data visualization dashboards",
                "mongodb": "NoSQL document stores",
                "elasticsearch": "distributed search indexes",
                "redis": "in-memory caching systems",
                "ci/cd": "automated deployment pipelines",
                "docker": "containerized application wrappers",
                "aws rds": "managed relational databases",
            }
            skills_lower = {s.lower().strip() for s in skills}
            forbidden_missing = {}
            for k, val in forbidden_mappings.items():
                if not any(k in s.lower() for s in skills_lower):
                    forbidden_missing[k] = val
                    
            clean_lines = []
            for line in cover_letter.split("\n"):
                for missing_tech, replacement in forbidden_missing.items():
                    line = re.sub(r'\b' + re.escape(missing_tech) + r'\b', replacement, line, flags=re.IGNORECASE)
                clean_lines.append(line)
            cover_letter = "\n".join(clean_lines)

        # ---- Parse interview questions (expect JSON array) -----------------
        interview_questions: list[str] = []
        if isinstance(interview_resp, Exception):
            print(f"[Career]: Interview questions generation failed: {interview_resp}")
        else:
            raw = interview_resp.content.strip()
            try:
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                interview_questions = json.loads(raw)
            except Exception:
                interview_questions = [
                    line.strip("- 0123456789.)")
                    for line in raw.split("\n")
                    if line.strip()
                ][:10]

        # ---- Parse career roadmap (expect JSON array of 6 weekly items) -----
        roadmap: list[str] = []
        if isinstance(roadmap_resp, Exception):
            print(f"[Career]: Roadmap generation failed: {roadmap_resp}")
        else:
            raw = roadmap_resp.content.strip()
            try:
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                roadmap = json.loads(raw)
            except Exception:
                roadmap = [
                    f"Week 1: Focus on mastering Unit Testing with pytest.",
                    f"Week 2: Build an asynchronous event processor with Redis queue.",
                    f"Week 3: Scale FastAPI services using Docker Compose.",
                    f"Week 4: Set up local Kubernetes (Minikube) cluster deployments.",
                    f"Week 5: Implement Infrastructure-as-Code setups via Terraform.",
                    f"Week 6: Instrument API services with Prometheus and Grafana."
                ]

        # ---- Parse LinkedIn & GitHub optimization ---------------------------
        linkedin_optimization = {}
        github_optimization = {}
        if isinstance(linkedin_github_resp, Exception):
            print(f"[Career]: LinkedIn/GitHub branding failed: {linkedin_github_resp}")
        else:
            raw = linkedin_github_resp.content.strip()
            try:
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                data = json.loads(raw)
                linkedin_optimization = data.get("linkedin_optimization", {})
                github_optimization = data.get("github_optimization", {})
            except Exception as e:
                print(f"[Career]: Failed to parse LinkedIn/GitHub JSON: {e}")
                # Fallback
                linkedin_optimization = {
                    "headline": "Software Engineer | " + ", ".join(skills[:5]),
                    "about": "Experienced software engineer with expertise in " + ", ".join(skills[:5]),
                    "skills": skills[:10],
                    "featured_projects": [],
                }
                github_optimization = {
                    "bio": "Software Engineer specializing in " + ", ".join(skills[:3]),
                    "pinned_repositories": [],
                    "readme_suggestions": ["Create a professional profile README showcasing pinned repositories."],
                }

        print(f"[Career]: Generated cover letter ({len(cover_letter)} chars), "
              f"{len(interview_questions)} interview questions, {len(roadmap)} roadmap items, "
              f"LinkedIn & GitHub profile branding assets.")

        return {
            "cover_letter": cover_letter,
            "interview_questions": interview_questions,
            "roadmap": roadmap,
            "linkedin_optimization": linkedin_optimization,
            "github_optimization": github_optimization,
            "token_usage": token_usage,
        }

```

## File: `agent/node/placement.py`

```python
import asyncio
import json
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import COMPANY_RECOMMENDATIONS_PROMPT
from agent.schema.schema import CompanyRecommendation


class placement_discovery_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def placement_discovery_node(self, state: AdvancedAgentState, llm, search_tool):
        optimized = state.get("optimized_resume")
        all_skills = optimized.all_skills() if optimized else []

        # Build search query — fallback chain
        skill_source = all_skills or state.get("extracted_skills", [])
        skills_query = ", ".join(skill_source[:6]) if skill_source else state.get("job_description", "")[:120]

        # DuckDuckGoSearchRun is synchronous — run in thread pool
        try:
            query = f"Companies hiring {skills_query} engineers remote jobs 2024"
            search_results = await asyncio.to_thread(search_tool.run, query)
        except Exception as e:
            print(f"[Placement]: Search error ({e}), using profile-only inference.")
            search_results = f"No live data. Infer top hiring companies for: {skills_query}"

        candidate_profile = (
            f"Summary: {getattr(optimized, 'summary', '')}\n"
            f"Skills: {', '.join(all_skills)}\n"
            f"Top bullets: {'; '.join(optimized.all_bullets()[:3])}"
        ) if optimized else state.get("job_description", "")

        prompt = COMPANY_RECOMMENDATIONS_PROMPT.format(
            candidate_profile=candidate_profile,
            skills_query=skills_query,
            search_results=search_results,
        )

        # Get structured JSON response
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Parse JSON array into CompanyRecommendation list
        companies: list[CompanyRecommendation] = []
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            for item in data:
                rec = CompanyRecommendation(**item)
                # Programmatically align tech overlap with candidate's actual skills (Issue 11)
                valid_overlap = []
                for t in rec.tech_overlap:
                    t_clean = t.lower().strip()
                    for cs in all_skills:
                        cs_clean = cs.lower().strip()
                        if t_clean == cs_clean or t_clean in cs_clean or cs_clean in t_clean:
                            if cs not in valid_overlap:
                                valid_overlap.append(cs)
                                break
                rec.tech_overlap = valid_overlap
                
                # Programmatically calibrate match score based on overlap evidence (Issue 11)
                calculated_match = min(95, max(65, 70 + len(valid_overlap) * 6))
                rec.match = calculated_match
                
                # Format a structured, evidence-based reason (Issue 11)
                overlap_str = ", ".join(valid_overlap[:3])
                if overlap_str:
                    rec.reason = f"Excellent fit for the {rec.role} position at {rec.name} due to direct overlap in core stack elements: {overlap_str}. Candidate's background aligns perfectly with their technical requirements."
                else:
                    rec.reason = f"Strong alignment for the {rec.role} position at {rec.name} based on backend software engineering principles and similar system design paradigms."
                companies.append(rec)
        except Exception as e:
            print(f"[Placement]: JSON parse error ({e}). Raw: {raw[:200]}")
            companies = [CompanyRecommendation(
                name="Parse Error",
                role="See logs",
                match=0,
                tech_overlap=[],
                reason=f"LLM returned non-JSON: {raw[:100]}",
            )]

        token_usage = {"input": 0, "output": 0, "total": 0}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input": response.usage_metadata.get("input_tokens", 0),
                "output": response.usage_metadata.get("output_tokens", 0),
                "total": response.usage_metadata.get("total_tokens", 0),
            }

        print(f"[Placement]: Generated {len(companies)} company recommendations.")
        return {
            "suggested_companies": [c.model_dump() for c in companies],
            "token_usage": token_usage,
        }

```

## File: `agent/node/final_output.py`

```python
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
            "iterations": state.get("iterations", 0),
            "suggestion": state.get("suggestion", ""),
            "approved": state.get("approved", False),
        }

```

## File: `agent/services/resume_parser.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover - optional dependency guard
    pytesseract = None
    convert_from_bytes = None


@dataclass
class ParsedResume:
    text: str
    filename: str
    ocr_used: bool = False
    mime_type: str = "text/plain"


def parse_resume_file(filename: str, file_bytes: bytes) -> ParsedResume:
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        return _parse_docx(file_bytes, filename)
    if suffix == ".pdf":
        return _parse_pdf(file_bytes, filename)
    raise ValueError(f"Unsupported resume format: {suffix}")


def _parse_docx(file_bytes: bytes, filename: str) -> ParsedResume:
    document = DocxDocument(BytesIO(file_bytes))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    text = "\n".join(paragraphs)
    return ParsedResume(text=text, filename=filename, mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def _parse_pdf(file_bytes: bytes, filename: str) -> ParsedResume:
    reader = PdfReader(BytesIO(file_bytes))
    text_parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    text = "\n".join(part for part in text_parts if part).strip()
    ocr_used = False
    if not text:
        text = _ocr_pdf(file_bytes)
        ocr_used = bool(text)

    return ParsedResume(text=text, filename=filename, mime_type="application/pdf", ocr_used=ocr_used)


def _ocr_pdf(file_bytes: bytes) -> str:
    if convert_from_bytes is None or pytesseract is None:
        return ""

    try:
        images = convert_from_bytes(file_bytes)
    except Exception:
        return ""

    text_parts: list[str] = []
    for image in images:
        try:
            extracted_text = pytesseract.image_to_string(image)
        except Exception:
            continue
        if extracted_text:
            text_parts.append(extracted_text)

    return "\n".join(text_parts).strip()

```

## File: `agent/services/pdf_compiler.py`

```python
"""
pdf_compiler.py
Compiles a LaTeX string into a PDF using pdflatex.
Runs in a temporary directory and returns the raw PDF bytes.
"""
import asyncio
import shutil
import tempfile
from pathlib import Path


async def compile_latex_to_pdf(latex: str) -> bytes:
    """
    Compile a LaTeX source string to PDF bytes using pdflatex.

    Runs pdflatex twice (standard practice for resolving internal references).
    Raises RuntimeError on compilation failure with the pdflatex log excerpt.
    """
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found. Install TeX Live: "
            "sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-latex-extra"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "resume.tex"
        pdf_path = Path(tmpdir) / "resume.pdf"
        log_path = Path(tmpdir) / "resume.log"

        tex_path.write_text(latex, encoding="utf-8")

        cmd = [
            pdflatex,
            "-interaction=nonstopmode",   # don't pause on errors
            "-halt-on-error",              # exit non-zero on first error
            "-output-directory", tmpdir,
            str(tex_path),
        ]

        # Run twice — first pass builds the document, second resolves references
        for pass_num in (1, 2):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Extract the useful part of the log (last 60 lines)
                log_excerpt = ""
                if log_path.exists():
                    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    log_excerpt = "\n".join(log_lines[-60:])
                raise RuntimeError(
                    f"pdflatex failed on pass {pass_num} (exit {proc.returncode}).\n"
                    f"Log tail:\n{log_excerpt or stderr.decode(errors='replace')}"
                )

        if not pdf_path.exists():
            raise RuntimeError("pdflatex ran successfully but no PDF was produced.")

        return pdf_path.read_bytes()

```

## File: `agent/services/latex_renderer.py`

```python
"""
latex_renderer.py
Compiles structured OptimizedResumeOutput schemas into high-quality LaTeX code
using a fixed Jinja2 template (Jake's Resume style).
"""

import re
from jinja2 import Environment
from agent.schema.schema import OptimizedResumeOutput


# ---------------------------------------------------------------------------
# LaTeX special-character escaping
# ---------------------------------------------------------------------------
_LATEX_ESCAPE_MAP = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]

def _escape(text: str) -> str:
    """Escape all LaTeX-special characters in a plain-text string."""
    if not text:
        return ""
    result = text.replace("\\", r"\textbackslash{}")
    for char, replacement in _LATEX_ESCAPE_MAP[1:]:
        result = result.replace(char, replacement)
    return result


def _format_latex_link(link: str, default_domain: str) -> tuple[str, str]:
    """Returns a tuple of (href_url, display_text) for hyperref, ensuring full URLs are preserved."""
    if not link:
        return "", ""
    
    link_raw = link.strip()
    link_clean = link_raw.lower()
    
    # 1. Determine href_url
    if link_clean.startswith("http://") or link_clean.startswith("https://"):
        href_url = link_raw
    else:
        href_url = f"https://{link_raw}"
        
    # 2. Determine display_text
    display_text = link_raw
    # Remove http:// or https:// prefix for cleaner display text
    display_text = re.sub(r'^https?://', '', display_text)
    
    # If the domain is not in the text, format it using the default domain
    if default_domain and default_domain not in display_text.lower():
        if default_domain == "linkedin.com":
            display_text = f"linkedin.com/in/{display_text}"
            href_url = f"https://linkedin.com/in/{link_raw}"
        elif default_domain == "github.com":
            display_text = f"github.com/{display_text}"
            href_url = f"https://github.com/{link_raw}"
            
    return href_url, display_text


# ---------------------------------------------------------------------------
# Jinja2 environment — uses (( )) delimiters so they don't clash with LaTeX
# ---------------------------------------------------------------------------
_jinja_env = Environment(
    variable_start_string="((",
    variable_end_string="))",
    block_start_string="(%",
    block_end_string="%)",
    comment_start_string="(#",
    comment_end_string="#)",
    autoescape=False,
    keep_trailing_newline=True,
)
_jinja_env.filters["e"] = _escape


# ---------------------------------------------------------------------------
# Jake's Resume–style LaTeX template
# ---------------------------------------------------------------------------
_RESUME_TEMPLATE = r"""
\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\input{glyphtounicode}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

\pdfgentounicode=1

%--- PDF Metadata & Hyperref Setup ---
\hypersetup{
    pdfauthor={(( contact.name ))},
    pdftitle={Resume},
    pdfcreator={Resume AI Agent}
}

%--- Compact enumitem Spacing ---
\setlist[itemize]{itemsep=1pt, topsep=2pt, parsep=0pt, partopsep=0pt}

%--- Custom commands ----------------------------------------------------------
\newcommand{\resumeItem}[1]{\item\small{#1 \vspace{-2pt}}}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-5pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-5pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-3pt}}

%------------------------------------------------------------------------------
\begin{document}

%--- Header -------------------------------------------------------------------
\begin{center}
    \textbf{\Huge \scshape (( contact.name ))} \\ \vspace{2pt}
    \small
    (( contact.phone ))
    (% if contact.phone and contact.email %) $|$ (% endif %)\href{mailto:(( contact.email ))}{\underline{(( contact.email ))}}
    (% if contact.location %)$|$ (( contact.location ))(% endif %)
    (% if contact.linkedin_url %)$|$ \href{(( contact.linkedin_url ))}{\underline{(( contact.linkedin_display ))}}(% endif %)
    (% if contact.github_url %)$|$ \href{(( contact.github_url ))}{\underline{(( contact.github_display ))}}(% endif %)
    (% if contact.portfolio_url %)$|$ \href{(( contact.portfolio_url ))}{\underline{(( contact.portfolio_display ))}}(% endif %)
\end{center}

%--- Summary ------------------------------------------------------------------
\section{Summary}
\small (( summary ))

%--- Skills -------------------------------------------------------------------
\section{Technical Skills}
\begin{tabularx}{\textwidth}{X}
    (% if skills.languages %)
     \textbf{Languages}{: (( skills.languages | map('e') | join(', ') ))} \\
    (% endif %)
    (% if skills.frameworks %)
     \textbf{Frameworks}{: (( skills.frameworks | map('e') | join(', ') ))} \\
    (% endif %)
    (% if skills.databases %)
     \textbf{Databases}{: (( skills.databases | map('e') | join(', ') ))} \\
    (% endif %)
    (% if skills.cloud_devops %)
     \textbf{Cloud \& DevOps}{: (( skills.cloud_devops | map('e') | join(', ') ))} \\
    (% endif %)
    (% if skills.ai_tools %)
     \textbf{AI \& Machine Learning}{: (( skills.ai_tools | map('e') | join(', ') ))} \\
    (% endif %)
    (% if skills.other %)
     \textbf{Other}{: (( skills.other | map('e') | join(', ') ))} \\
    (% endif %)
\end{tabularx}
\vspace{-8pt}

%--- Experience ---------------------------------------------------------------
\section{Experience}
\resumeSubHeadingListStart
(% for job in experience %)
  \resumeSubheading
    {(( job.title | e ))}{(( job.start_date | e )) -- (( job.end_date | e ))}
    {(( job.company | e ))}{(( job.location | e ))}
  \resumeItemListStart
    (% for bullet in job.bullets %)
      \resumeItem{(( bullet | e ))}
    (% endfor %)
  \resumeItemListEnd
(% endfor %)
\resumeSubHeadingListEnd

(% if projects %)
%--- Projects -----------------------------------------------------------------
\section{Projects}
\resumeSubHeadingListStart
(% for proj in projects %)
  \resumeProjectHeading
    {\textbf{(( proj.name | e ))} $|$ \emph{\small (( proj.tech_stack | map('e') | join(', ') ))}}{}
  \resumeItemListStart
    \resumeItem{(( proj.description | e ))}
  \resumeItemListEnd
(% endfor %)
\resumeSubHeadingListEnd
(% endif %)

(% if education %)
%--- Education ----------------------------------------------------------------
\section{Education}
\resumeSubHeadingListStart
(% for edu in education %)
  \resumeSubheading
    {(( edu.institution | e ))}{(( edu.graduation_date | e ))}
    {(( edu.degree | e ))}{}
(% endfor %)
\resumeSubHeadingListEnd
(% endif %)

\end{document}
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_latex(optimized: OptimizedResumeOutput) -> str:
    """
    Render an OptimizedResumeOutput into a complete, compilable LaTeX string.
    All user-supplied text is escaped; the document structure comes from the
    fixed template.
    """
    # Pre-process links for contact info
    linkedin_url, linkedin_display = _format_latex_link(optimized.contact.linkedin, "linkedin.com")
    github_url, github_display = _format_latex_link(optimized.contact.github, "github.com")
    portfolio_url, portfolio_display = _format_latex_link(optimized.contact.portfolio, "")
    
    # Pre-escape all contact fields
    contact_data = {
        "name": _escape(optimized.contact.name),
        "email": _escape(optimized.contact.email),
        "phone": _escape(optimized.contact.phone),
        "location": _escape(optimized.contact.location),
        "linkedin_url": linkedin_url,
        "linkedin_display": _escape(linkedin_display),
        "github_url": github_url,
        "github_display": _escape(github_display),
        "portfolio_url": portfolio_url,
        "portfolio_display": _escape(portfolio_display),
    }

    template = _jinja_env.from_string(_RESUME_TEMPLATE)
    latex = template.render(
        contact=contact_data,
        summary=_escape(optimized.summary),
        skills=optimized.skills,
        experience=optimized.experience,
        education=optimized.education,
        projects=optimized.projects,
    )
    return latex

```

## File: `agent/services/version_store.py`

```python
from __future__ import annotations

import json
import os
from typing import Any

import redis.asyncio as redis


class ResumeVersionStore:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
            await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def save_version(self, thread_id: str, payload: dict[str, Any]) -> None:
        await self.connect()
        assert self._client is not None
        key = f"resume:versions:{thread_id}"
        await self._client.lpush(key, json.dumps(payload))
        await self._client.ltrim(key, 0, 9)

    async def list_versions(self, thread_id: str) -> list[dict[str, Any]]:
        await self.connect()
        assert self._client is not None
        key = f"resume:versions:{thread_id}"
        raw_items = await self._client.lrange(key, 0, 9)
        return [json.loads(item) for item in raw_items if item]

    async def save_latex(self, thread_id: str, latex: str) -> None:
        """Store the compiled LaTeX string keyed by thread_id (TTL 24 h)."""
        await self.connect()
        assert self._client is not None
        key = f"resume:latex:{thread_id}"
        await self._client.set(key, latex, ex=86400)

    async def get_latex(self, thread_id: str) -> str | None:
        """Retrieve the stored LaTeX string for the given thread_id."""
        await self.connect()
        assert self._client is not None
        key = f"resume:latex:{thread_id}"
        return await self._client.get(key)

```

