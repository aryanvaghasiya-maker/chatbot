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

        # Extract optimized seniority job title and 3-4 primary technical keywords
        role_title = "Backend Engineer"
        if optimized and optimized.contact and optimized.contact.title:
            role_title = optimized.contact.title.split("|")[0].strip()
        
        primary_keywords = []
        if all_skills:
            primary_keywords = all_skills[:4]
        else:
            primary_keywords = state.get("extracted_skills", [])[:4]
        primary_kws_str = " ".join(primary_keywords)

        # Determine candidate remote/location preference
        jd_lower = state.get("job_description", "").lower()
        prefers_remote = "remote" in jd_lower or "work from home" in jd_lower or "wfh" in jd_lower
        orig_location = state.get("original_contact", {}).get("location", "") or ""
        if "remote" in orig_location.lower():
            prefers_remote = True

        remote_suffix = "remote" if prefers_remote else ""
        query = f"companies hiring {role_title} {primary_kws_str} {remote_suffix} jobs 2026".strip()

        # DuckDuckGoSearchRun is synchronous — run in thread pool
        try:
            print(f"[Placement]: Running search with query: '{query}'")
            search_results = await asyncio.to_thread(search_tool.run, query)
        except Exception as e:
            print(f"[Placement]: Search error ({e}), using profile-only inference.")
            search_results = f"No live data. Infer top hiring companies for: {role_title} {primary_kws_str}"

        candidate_profile = (
            f"Summary: {getattr(optimized, 'summary', '')}\n"
            f"Skills: {', '.join(all_skills)}\n"
            f"Top bullets: {'; '.join(optimized.all_bullets()[:3])}"
        ) if optimized else state.get("job_description", "")

        prompt = COMPANY_RECOMMENDATIONS_PROMPT.format(
            candidate_profile=candidate_profile,
            skills_query=primary_kws_str,
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
            if isinstance(data, dict):
                data = data.get("companies") or data.get("recommendations") or data.get("jobs") or data.get("data") or [data]
            if not isinstance(data, list):
                data = [data]
                
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    # Normalize keys to match CompanyRecommendation schema
                    name_val = item.get("name") or item.get("company_name") or item.get("company") or "Unknown Company"
                    role_val = item.get("role") or item.get("job_title") or item.get("title") or item.get("position") or "Backend Engineer"
                    
                    match_val = item.get("match") or item.get("match_percentage") or item.get("score") or item.get("matching_score") or 80
                    try:
                        match_val = int(str(match_val).replace("%", ""))
                    except Exception:
                        match_val = 80
                        
                    location_val = item.get("location") or item.get("job_location") or "Remote"
                    location_str = str(location_val)

                    # Strict remote filtering
                    if prefers_remote and not any(r in location_str.lower() for r in ["remote", "hybrid", "wfh", "work from home", "anywhere"]):
                        print(f"[Placement]: Discarding non-remote recommendation: {name_val} ({location_str})")
                        continue
                    
                    tech_overlap_val = item.get("tech_overlap") or item.get("tech_stack") or item.get("overlap") or item.get("technologies") or []
                    if isinstance(tech_overlap_val, str):
                        tech_overlap_val = [t.strip() for t in tech_overlap_val.split(",") if t.strip()]
                    elif not isinstance(tech_overlap_val, list):
                        tech_overlap_val = []
                        
                    reason_val = item.get("reason") or item.get("why_fit") or item.get("explanation") or item.get("description") or ""

                    rec = CompanyRecommendation(
                        name=str(name_val),
                        role=str(role_val),
                        match=int(match_val),
                        location=location_str,
                        tech_overlap=[str(t) for t in tech_overlap_val],
                        reason=str(reason_val)
                    )
                    
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
                except Exception as item_err:
                    print(f"[Placement]: Error parsing single company recommendation ({item_err}): {item}")
            
            # Fallback if strict filtering leaves us with 0 recommendations
            if len(companies) == 0 and len(data) > 0:
                print("[Placement]: Strict remote filter yielded 0 recommendations. Relaxing filter to keep all results.")
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    try:
                        name_val = item.get("name") or item.get("company_name") or "Unknown Company"
                        role_val = item.get("role") or item.get("title") or "Backend Engineer"
                        match_val = 80
                        location_val = "Remote (Hybrid)" if prefers_remote else (item.get("location") or "Remote")
                        
                        rec = CompanyRecommendation(
                            name=str(name_val),
                            role=str(role_val),
                            match=match_val,
                            location=str(location_val),
                            tech_overlap=[],
                            reason="Aligned recommendation based on backend software engineering profile."
                        )
                        companies.append(rec)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Placement]: JSON parse error ({e}). Raw: {raw[:200]}")
            errors = state.get("errors") or []
            errors.append({"module": "job_matching", "message": f"Unable to generate job recommendations due to parsing error: {str(e)}"})
            status = "partial_success"
            companies = []

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
            "errors": state.get("errors", []) + (errors if 'errors' in locals() else []),
            "status": state.get("status", "success") if 'status' not in locals() else status,
        }
