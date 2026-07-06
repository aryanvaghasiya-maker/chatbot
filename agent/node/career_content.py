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
from pydantic import BaseModel, Field
from agent.schema.schema import LinkedInOptimization, GitHubOptimization

class ProfileBranding(BaseModel):
    linkedin_optimization: LinkedInOptimization = Field(default_factory=LinkedInOptimization)
    github_optimization: GitHubOptimization = Field(default_factory=GitHubOptimization)
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

        # ---- Build prompts & Conditionally Run tasks ----------------------
        tasks = {}
        generate_cover = state.get("generate_cover_letter", True)
        generate_interview = state.get("generate_interview_questions", True)
        generate_road = state.get("generate_roadmap", True)

        if generate_cover:
            cover_letter_prompt = COVER_LETTER_PROMPT.format(
                candidate_summary=summary,
                job_description=job_description,
            )
            tasks["cover"] = llm.ainvoke([HumanMessage(content=cover_letter_prompt)])

        if generate_interview:
            interview_prompt = INTERVIEW_QUESTIONS_PROMPT.format(
                skills=", ".join(skills[:15]),
                job_description=job_description,
            )
            tasks["interview"] = llm.ainvoke([HumanMessage(content=interview_prompt)])

        if generate_road:
            roadmap_prompt = CAREER_ROADMAP_PROMPT.format(
                skills=", ".join(skills[:15]),
                missing_keywords=", ".join(missing_keywords[:10]),
                job_description=job_description,
            )
            tasks["roadmap"] = llm.ainvoke([HumanMessage(content=roadmap_prompt)])

        generate_linkedin_github = (generate_cover or generate_interview or generate_road)
        if generate_linkedin_github:
            linkedin_github_prompt = LINKEDIN_GITHUB_OPTIMIZATION_PROMPT.format(
                candidate_summary=summary,
                skills=", ".join(skills[:20]),
                job_description=job_description,
            )
            structured_llm = llm.with_structured_output(ProfileBranding, method="function_calling")
            tasks["linkedin_github"] = structured_llm.ainvoke([HumanMessage(content=linkedin_github_prompt)])

        # Run tasks in parallel
        keys = list(tasks.keys())
        futures = list(tasks.values())
        results = {}
        if futures:
            resps = await asyncio.gather(*futures, return_exceptions=True)
            for k, resp in zip(keys, resps):
                results[k] = resp

        cover_resp = results.get("cover", None)
        interview_resp = results.get("interview", None)
        roadmap_resp = results.get("roadmap", None)
        linkedin_github_resp = results.get("linkedin_github", None)

        # ---- Accumulate actual token usage from all tasks -----------------
        input_tokens = 0
        output_tokens = 0
        for resp in [cover_resp, interview_resp, roadmap_resp, linkedin_github_resp]:
            if resp is not None and not isinstance(resp, Exception) and hasattr(resp, "usage_metadata") and resp.usage_metadata:
                input_tokens += resp.usage_metadata.get("input_tokens", 0)
                output_tokens += resp.usage_metadata.get("output_tokens", 0)
        token_usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

        errors = state.get("errors") or []
        status = state.get("status", "success")

        # ---- Parse cover letter --------------------------------------------
        cover_letter = ""
        if cover_resp is None:
            pass
        elif isinstance(cover_resp, Exception):
            print(f"[Career]: Cover letter generation failed: {cover_resp}")
            errors.append({"module": "cover_letter", "message": f"Cover letter generation failed: {str(cover_resp)}"})
            status = "partial_success"
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
        if interview_resp is None:
            pass
        elif isinstance(interview_resp, Exception):
            print(f"[Career]: Interview questions generation failed: {interview_resp}")
            errors.append({"module": "interview_questions", "message": f"Interview questions generation failed: {str(interview_resp)}"})
            status = "partial_success"
        else:
            raw = interview_resp.content.strip()
            try:
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            q = item.get("question", "")
                            cat = item.get("category", "")
                            if q and cat:
                                interview_questions.append(f"{cat}: {q}")
                            elif q:
                                interview_questions.append(q)
                            else:
                                interview_questions.append(str(item))
                        elif isinstance(item, str):
                            interview_questions.append(item)
                        else:
                            interview_questions.append(str(item))
                elif isinstance(parsed, dict):
                    possible_list = parsed.get("interview_questions") or parsed.get("questions") or parsed.get("data")
                    if isinstance(possible_list, list):
                        for item in possible_list:
                            if isinstance(item, dict):
                                q = item.get("question", "")
                                cat = item.get("category", "")
                                if q and cat:
                                    interview_questions.append(f"{cat}: {q}")
                                elif q:
                                    interview_questions.append(q)
                                else:
                                    interview_questions.append(str(item))
                            elif isinstance(item, str):
                                interview_questions.append(item)
                            else:
                                interview_questions.append(str(item))
                    else:
                        interview_questions = [str(parsed)]
                else:
                    interview_questions = [str(parsed)]
            except Exception as e:
                print(f"[Career]: JSON parse error in interview questions ({e})")
                errors.append({"module": "interview_questions", "message": f"JSON parsing failed for interview questions: {str(e)}"})
                status = "partial_success"
                interview_questions = [
                    line.strip("- 0123456789.)")
                    for line in raw.split("\n")
                    if line.strip()
                ][:10]

        # ---- Parse career roadmap (expect JSON array of 6 weekly items) -----
        roadmap: list[str] = []
        if roadmap_resp is None:
            pass
        elif isinstance(roadmap_resp, Exception):
            print(f"[Career]: Roadmap generation failed: {roadmap_resp}")
            errors.append({"module": "roadmap", "message": f"Roadmap generation failed: {str(roadmap_resp)}"})
            status = "partial_success"
        else:
            raw = roadmap_resp.content.strip()
            try:
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            goal = item.get("goal") or item.get("milestone") or item.get("week") or item.get("task", "")
                            week = item.get("week", "")
                            if week and goal:
                                if not str(week).lower().startswith("week"):
                                    roadmap.append(f"Week {week}: {goal}")
                                else:
                                    roadmap.append(f"{week}: {goal}")
                            elif goal:
                                roadmap.append(str(goal))
                            else:
                                roadmap.append(str(item))
                        else:
                            roadmap.append(str(item))
                elif isinstance(parsed, dict):
                    possible_list = parsed.get("roadmap") or parsed.get("milestones") or parsed.get("steps")
                    if isinstance(possible_list, list):
                        for item in possible_list:
                            if isinstance(item, dict):
                                goal = item.get("goal") or item.get("milestone") or item.get("week") or item.get("task", "")
                                week = item.get("week", "")
                                if week and goal:
                                    if not str(week).lower().startswith("week"):
                                        roadmap.append(f"Week {week}: {goal}")
                                    else:
                                        roadmap.append(f"{week}: {goal}")
                                elif goal:
                                    roadmap.append(str(goal))
                                else:
                                    roadmap.append(str(item))
                            else:
                                roadmap.append(str(item))
                    else:
                        roadmap = [str(parsed)]
                else:
                    roadmap = [str(parsed)]
            except Exception as e:
                print(f"[Career]: JSON parse error in roadmap ({e})")
                errors.append({"module": "roadmap", "message": f"JSON parsing failed for roadmap: {str(e)}"})
                status = "partial_success"
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
        if linkedin_github_resp is None:
            pass
        elif isinstance(linkedin_github_resp, Exception):
            print(f"[Career]: LinkedIn/GitHub branding failed: {linkedin_github_resp}")
            errors.append({"module": "linkedin_github", "message": f"LinkedIn/GitHub branding failed: {str(linkedin_github_resp)}"})
            status = "partial_success"
        else:
            try:
                if hasattr(linkedin_github_resp, "linkedin_optimization"):
                    linkedin_optimization = linkedin_github_resp.linkedin_optimization.model_dump()
                    github_optimization = linkedin_github_resp.github_optimization.model_dump()
                else:
                    data = linkedin_github_resp
                    if isinstance(data, dict):
                        linkedin_optimization = data.get("linkedin_optimization", {})
                        github_optimization = data.get("github_optimization", {})
            except Exception as e:
                print(f"[Career]: Failed to parse LinkedIn/GitHub object: {e}")
                errors.append({"module": "linkedin_github", "message": f"LinkedIn/GitHub parsing failed: {str(e)}"})
                status = "partial_success"

        # Fallback values if empty
        if not linkedin_optimization:
            linkedin_optimization = {
                "headline": "Software Engineer | " + " | ".join(skills[:3]),
                "about": "Experienced software engineer specializing in backend development, API design, and distributed systems.",
                "skills": skills[:10],
                "featured_projects": [],
            }
        if not github_optimization:
            github_optimization = {
                "bio": "Software Engineer specializing in backend development.",
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
            "errors": errors,
            "status": status,
        }
