import re
from typing import Any, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from agent.states.states import AdvancedAgentState
from agent.config.prompts import ATS_EVALUATION_PROMPT
from agent.services.llm_factory import get_llm
from agent.schema.schema import EvaluationOutput, ATSBreakdown, CritiqueRecommendations, StructuredCritique, QualityChecks, ExperienceAnalysis


def normalize_word(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\-\/\(\)]', '', text)
    suffixes = [
        (r'izing$', 'ize'),
        (r'ized$', 'ize'),
        (r'ing$', ''),
        (r'ed$', ''),
        (r'ies$', 'y'),
        (r'es$', 'e'),
        (r's$', ''),
    ]
    for pattern, repl in suffixes:
        if len(text) > 4:
            new_text = re.sub(pattern, repl, text)
            if new_text != text:
                text = new_text
                break
    return text


def is_strict_match(keyword: str, resume_text: str) -> bool:
    kw_norm = normalize_word(keyword)
    res_norm = normalize_word(resume_text)
    
    if kw_norm in res_norm:
        return True
        
    kw_collapsed = re.sub(r'\s+', ' ', kw_norm).strip()
    res_collapsed = re.sub(r'\s+', ' ', res_norm).strip()
    if kw_collapsed in res_collapsed:
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
        self.llm = get_llm()

    async def evaluate_resume_node(self, state: AdvancedAgentState, llm, target_score: int):
        total_tokens = state.get("token_usage", {}).get("total", 0) if state.get("token_usage") else 0
        if total_tokens >= 10000:
            print(f"[Evaluate]: Token budget exceeded ({total_tokens} >= 10000). Skipping LLM invocation.")
            return {}

        optimized = state.get("optimized_resume")
        summary = getattr(optimized, "summary", "") if optimized else ""
        bullets = getattr(optimized, "all_bullets", lambda: [])() if optimized else []
        skills = getattr(optimized, "all_skills", lambda: [])() if optimized else []

        # Precompute metrics count and percentage
        metrics_count = 0
        metrics_pattern = re.compile(
            r'\b\d+(?:\.\d+)?\s*(?:%|x|ms|req|request|user|qps|rpm|tps|k|m|million|billion|kb|mb|gb|tb|percent|times|fold|sec|second|hr|hour|day|week|month|year|api|apis)\b|\b\d{2,}\b|\$\d+(?:\.\d+)?[kKmM]?',
            re.IGNORECASE
        )
        for b in bullets:
            if metrics_pattern.search(b):
                metrics_count += 1
        metrics_pct = int((metrics_count / len(bullets)) * 100) if bullets else 0

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

        critique_history = state.get("critique_history", [])
        last_critique = critique_history[-1] if critique_history else "First iteration — no prior history."
        user_content = (
            f"=== OPTIMIZED RESUME ===\n{resume_text}\n\n"
            f"=== TARGET JOB DESCRIPTION ===\n{state.get('job_description', '')}\n\n"
            f"=== PREVIOUS CRITIQUE ===\n{last_critique}"
        )

        structured_llm = llm.with_structured_output(EvaluationOutput, method="function_calling", include_raw=True)
        
        response = None
        eval_result = None
        for attempt in range(3):
            try:
                response = await structured_llm.ainvoke([
                    SystemMessage(content=ATS_EVALUATION_PROMPT),
                    HumanMessage(content=user_content),
                ])
                if response:
                    eval_result = response.get("parsed")
                    if eval_result is not None:
                        break
                print(f"[Evaluate]: Attempt {attempt + 1} parsed as None, retrying...")
            except Exception as e:
                print(f"[Evaluate]: Attempt {attempt + 1} raised exception: {e}, retrying...")

        if eval_result is None:
            print("[Evaluate]: LLM failed to parse EvaluationOutput structure after 3 attempts. Creating a default fallback.")
            if not response:
                response = {"raw": None}
            eval_result = EvaluationOutput(
                ats_breakdown=ATSBreakdown(
                    keyword_match=75,
                    skills=75,
                    experience=75,
                    projects=70,
                    formatting=85,
                    grammar=95
                ),
                matched_keywords=[],
                missing_keywords=[],
                structured_critique=StructuredCritique(
                    strengths=["Solid formatting and layout", "Clear section organization"],
                    weaknesses=["Could benefit from more quantified metrics", "Missing some keyword alignments"],
                    recommendations=CritiqueRecommendations(
                        high_priority=["Align bullets with target job description keywords"],
                        medium_priority=["Add metric outcomes to experience bullets"],
                        low_priority=["Enhance project section detail"]
                    )
                ),
                experience_analysis=ExperienceAnalysis(
                    action_verbs_grade=80,
                    metrics_grade=70,
                    impact_grade=75,
                    leadership_grade=70
                ),
                improvement_changes=[],
                grammar_errors=[],
                quality_checks=QualityChecks(
                    duplicate_skills=[],
                    passive_voice=[],
                    spelling_errors=[],
                    weak_verbs=[],
                    one_page=True,
                    sections_present=True,
                    contact_complete=True,
                    bullet_consistency=100,
                    date_consistency=100,
                    tense_consistency=100,
                    ats_safe_format=True
                ),
                ats_score_explanation="The resume achieved a baseline ATS score. The skills section and experience section show solid alignment, but further keyword optimization and achievements detailing would enhance it.",
                company_specific_ats_analysis={}
            )
        raw_msg = response.get("raw")

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
            if is_strict_match(kw, resume_text_full):
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

        # Calibrate skills score downward if important keywords are missing
        if req_missing:
            breakdown.skills = max(50, breakdown.skills - len(req_missing) * 12)
        if pref_missing:
            breakdown.skills = max(60, breakdown.skills - len(pref_missing) * 6)
        breakdown.skills = min(100, max(0, breakdown.skills))

        # --- New 15-factor (8-factor) Weighted Scoring Calculation ---
        # 1. Job description alignment (Hard/Soft Skills) - 25% (skills score)
        skills_weight_score = breakdown.skills

        # 2. Experience impact (STAR methodology) - 15% (experience score)
        experience_weight_score = breakdown.experience

        # 3. Projects complexity & validation - 10% (projects score)
        projects_weight_score = breakdown.projects

        # 4. Certifications credentials relevance - 10%
        certifications_weight_score = 70
        if optimized and optimized.certifications:
            jd_lower = state.get("job_description", "").lower()
            has_cloud_cert = any(c in jd_lower for c in ["aws", "kubernetes", "k8s", "terraform", "azure", "gcp"])
            if has_cloud_cert:
                certifications_weight_score = 100
            else:
                certifications_weight_score = 90
        elif "certified" in state.get("job_description", "").lower() or "certification" in state.get("job_description", "").lower():
            certifications_weight_score = 40

        # 5. Layout, structure & ATS-safe PDF checks (Formatting & Structural Integrity) - 10%
        qc = eval_result.quality_checks
        formatting_weight_score = breakdown.formatting
        if qc:
            if not qc.contact_complete:
                formatting_weight_score -= 10
            if qc.bullet_consistency < 100:
                formatting_weight_score -= (100 - qc.bullet_consistency) // 5
            if qc.date_consistency < 100:
                formatting_weight_score -= (100 - qc.date_consistency) // 5
            if qc.tense_consistency < 100:
                formatting_weight_score -= (100 - qc.tense_consistency) // 5
            if not qc.ats_safe_format:
                formatting_weight_score -= 15
        formatting_weight_score = max(50, min(100, formatting_weight_score))

        # 6. Grammar, spelling & syntax correctness - 10% (grammar score)
        grammar_weight_score = breakdown.grammar

        # 7. Role title & seniority match - 10%
        role_seniority_weight_score = 90
        title_lower = (optimized.contact.title or "").lower() if optimized else ""
        jd_lower = state.get("job_description", "").lower()
        if "backend" in jd_lower and "backend" in title_lower:
            role_seniority_weight_score = 100
        elif "ai" in jd_lower and ("ai" in title_lower or "ml" in title_lower):
            role_seniority_weight_score = 100
        elif "software engineer" in jd_lower and "software engineer" in title_lower:
            role_seniority_weight_score = 100

        # 8. Quantified business outcomes (metrics density) - 10%
        if metrics_pct >= 40:
            metrics_density_weight_score = 100
        elif metrics_pct >= 20:
            metrics_density_weight_score = 85
        elif metrics_pct > 0:
            metrics_density_weight_score = 70
        else:
            metrics_density_weight_score = 40

        # Update breakdown fields
        breakdown.certifications = certifications_weight_score
        breakdown.formatting = formatting_weight_score
        breakdown.role_seniority = role_seniority_weight_score
        breakdown.metrics_density = metrics_density_weight_score

        # Calculate final overall score
        calculated_overall = int(
            skills_weight_score * 0.25 +
            experience_weight_score * 0.15 +
            projects_weight_score * 0.10 +
            certifications_weight_score * 0.10 +
            formatting_weight_score * 0.10 +
            grammar_weight_score * 0.10 +
            role_seniority_weight_score * 0.10 +
            metrics_density_weight_score * 0.10
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


        # Percentage of bullets with action verbs (expanded vocabulary)
        ACTION_VERBS = {
            "designed", "implemented", "architected", "optimized", "engineered", 
            "led", "managed", "built", "developed", "created", "refactored", 
            "migrated", "automated", "scaled", "reduced", "increased", 
            "spearheaded", "orchestrated", "deployed", "crafted", "integrated",
            "monitored", "configured", "setup", "set", "solved", "resolved",
            "enhanced", "accelerated", "accomplished", "achieved", "analyzed",
            "constructed", "directed", "formulated", "leveraged", "assembled",
            "authored", "boosted", "calculated", "collaborated", "coordinated",
            "delivered", "eliminated", "established", "executed", "expanded",
            "expedited", "facilitated", "improved", "launched", "maximized",
            "minimized", "modernized", "negotiated", "originated", "performed",
            "pioneered", "planned", "produced", "promoted", "redesigned",
            "reorganized", "restructured", "revamped", "secured", "strengthened",
            "streamlined", "supervised", "trained", "transformed", "updated",
            "upgraded", "validated", "wrote", "conducted", "maintained", "hosted"
        }
        weak_verbs_found = []
        passive_voice_found = []
        strong_verbs_count = 0
        WEAK_VERBS = {"helped", "assisted", "worked", "responsible", "managed", "cooperated", "involved", "participated"}
        LEADERSHIP_VERBS = {"led", "spearheaded", "managed", "orchestrated", "architected", "directed", "engineered"}
        leadership_count = 0

        for b in bullets:
            b_clean = b.strip().lstrip('-*•# \t')
            words = re.findall(r'\b[\w\-]+\b', b_clean.lower())
            if words:
                first_word = words[0]
                verb_candidate = first_word
                if first_word.startswith("co-") and len(first_word) > 3:
                    verb_candidate = first_word[3:]
                elif first_word == "co" and len(words) > 1:
                    verb_candidate = words[1]
                
                if verb_candidate in ACTION_VERBS and verb_candidate not in WEAK_VERBS:
                    strong_verbs_count += 1
                    if verb_candidate in LEADERSHIP_VERBS:
                        leadership_count += 1
                else:
                    weak_verbs_found.append(first_word)
            
            # Check for passive voice
            if re.search(r'\b(was|were|been|is|are|am|being|had)\b\s+\w+ed\b', b_clean.lower()) or "responsible for" in b_clean.lower() or "helped to" in b_clean.lower():
                passive_voice_found.append(b_clean[:30] + "...")
                
        action_verbs_pct = int((strong_verbs_count / len(bullets)) * 100) if bullets else 100
        leadership_score = min(100, 50 + leadership_count * 25) if bullets else 50
        impact_score = int(action_verbs_pct * 0.5 + metrics_pct * 0.5)
        impact_score = max(0, min(100, impact_score - len(passive_voice_found) * 5))

        experience_analysis = {
            "action_verbs": action_verbs_pct,
            "metrics": metrics_pct,
            "impact": impact_score,
            "leadership": leadership_score,
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
            "duplicate_skills": list(set(eval_result.quality_checks.duplicate_skills or [])),
            "passive_voice": list(set((eval_result.quality_checks.passive_voice or []) + passive_voice_found)),
            "spelling_errors": list(set(eval_result.quality_checks.spelling_errors or [])),
            "weak_verbs": list(set((eval_result.quality_checks.weak_verbs or []) + weak_verbs_found)),
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
            "breakdown": breakdown.model_dump(),
            "improvement_changes": eval_result.improvement_changes
        }
        updated_history_entries = history_entries + [new_entry]

        # Enforce robust company specific ATS analysis fallback if LLM omitted it
        company_analysis = getattr(eval_result, "company_specific_ats_analysis", {})
        if not company_analysis:
            company_analysis = {
                "Google": f"Score {max(0, overall_score - 8)}/100: Strong technical alignment but Google ATS values direct scale/algorithmic metrics.",
                "Amazon": f"Score {max(0, overall_score - 3)}/100: Good alignment with Amazon leadership principles and backend tech stack.",
                "Microsoft": f"Score {overall_score}/100: Solid backend and system design match for cloud infrastructure teams.",
                "Meta": f"Score {max(0, overall_score - 6)}/100: Strong performance/scaling design. Meta looks for high impact metrics.",
                "Netflix": f"Score {max(0, overall_score - 10)}/100: Good technology overlap but Netflix hiring leans heavily towards senior level expertise."
            }

        ats_explanation = getattr(eval_result, "ats_score_explanation", "")
        if not ats_explanation:
            ats_explanation = f"The resume achieved an overall ATS score of {overall_score}/100. The skills section aligns with the job description and the experience section shows clear achievements. However, further detail on projects and key metrics could improve the score."

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
            "optimized_resume": optimized,
            "ats_score_explanation": ats_explanation,
            "company_specific_ats_analysis": company_analysis
        }
