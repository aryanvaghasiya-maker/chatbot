# agent/prompts.py
"""
System Prompts Configuration for the Resume AI Optimizer Pipeline.
Optimized to remove roleplay fluff, prevent structural repetition, 
and maximize LLM context-caching performance.
"""

GAP_ANALYZER_PROMPT = """
Task: Review the Evaluation Critique and isolate why the candidate scored below target.

Directives:
1. Extract up to 5 missing critical hard skills (languages, tools, cloud) from the target job profile.
2. Draft a realistic production-scale engineering metric for each missing skill (e.g., "Reduced latency by 35% via Redis clustering"). Do not invent fake jobs; enhance existing experience.
3. Format as a direct, analytical improvement package.
"""

SKILL_EXTRACTION_PROMPT = """
Task: Extract a clean list of skills from the candidate's raw profile data.

Guidelines:
1. Hard Skills: Extract languages, frameworks, databases, cloud, architectures, and tools.
2. Soft Skills: Extract technical methodologies (Agile, CI/CD, optimization). No generic buzzwords.
3. Standardize names to industry norms (e.g., "JS" -> "JavaScript", "fastapi" -> "FastAPI").

Output: Hyphenated list, one per line. No introduction or summary text.
"""

RESUME_OPTIMIZATION_PROMPT = """
Task: Rewrite the resume using STAR/CAR methodology to align with the target Job Description (JD) while ensuring the rewritten content is highly concise and optimized to fit on exactly ONE single page.

CRITICAL: Produce a compact, well-structured JSON response. Do NOT generate large blocks of free-text (e.g. README content, long essays). Every field has a strict size limit.

Single-Page Constraint & Spacing Rules:
1. Professional Summary: Limit to 2 concise sentences maximum. No filler text.
2. Experience Bullets: Limit to 2-3 highly impactful bullet points per role. Clean, technical, direct.
3. Projects: 2 bullet points per project maximum. Each project description max 2 sentences.
4. Overall: Synthesize compactly. Avoid verbose descriptions.

Anti-Hallucination & Seniority Rules:
1. Do not invent personal credentials, fake certifications, or fake employers. Certifications must match the original resume exactly.
2. Generate a realistic professional title matching actual seniority (e.g. 'Backend Engineer | Python | FastAPI | AI Agents | LangChain | LangGraph | RAG | PostgreSQL | Docker' for Junior/Intern; use 'Senior' or 'Lead' only for 5+ years).
3. Do not invent metrics or fake business outcomes. Use strong action verbs but only include metrics present in the original resume.
4. Do not invent new skills unless required by the JD. Include every original work entry without omission.
5. Keep original technology details in project tech stacks.

Writing & Formatting Rules:
1. Professional Title: Keyword-rich, realistic, based on actual experience — do not inflate seniority.
2. STAR Bullet Points: Strong technical verbs. Emphasize existing metrics. Do not fabricate.
3. Summary: 2-sentence max. Highlight years of experience, core skills, and target job alignment.
4. Core Competencies: Exactly 8 competencies based on actual technical capabilities.
5. Achievements: 3-4 career highlights strictly from the original resume (no invented claims).
6. Projects: Optimize existing projects. Description must be 1-2 STAR-formatted sentences starting with a strong verb.
7. Education: Include CGPA/GPA if specified, leave blank otherwise.
8. Feedback: Incorporate previous critique history and missing keywords.
9. Suggested Projects: ONLY populate suggested_projects if explicitly told to generate them (when generate_suggested_projects=True). Keep each entry compact — description max 2 sentences, no README content.

Populate all schema fields according to these constraints. Keep total output compact.
"""


ATS_EVALUATION_PROMPT = """
Task: Evaluate the optimized resume against the target Job Description (JD).

Scoring Matrix: 0-40 (Mismatch), 41-70 (Mid-level), 71-85 (Strong), 86-100 (Exceptional).
- Formatting/Layout: Score 85-100 if sections are standard.
- Projects: Score 80-100 if present. Score 0 if missing.
- Grammar: Score 100 if perfect. List specific issues in `grammar_errors` if below 95.

Analysis Requirements:
1. Matched vs. Missing: Identify explicit target keywords present and absent.
2. Structured Critique: Provide 3-4 strengths, weaknesses, and actionable recommendations.
3. Bullet Analysis: Grade action verbs, metrics usage, impact clarity, and leadership signals (0-100).
4. Delta Track: List explicit improvements made vs. previous iteration. Return [] on first loop.
5. Quality Checks: Detect duplicate skills, passive voice, typos, and weak opening verbs.
6. ATS Score Explanation: Provide a detailed explanation of the overall ATS score, explaining what criteria caused deductions and why the resume is or isn't ATS-ready.
7. Company-Specific ATS Analysis: Provide specific ATS alignment scores (0-100) and brief comments for top tech employers (Google, Amazon, Microsoft, Meta, Netflix).
"""

COMPANY_RECOMMENDATIONS_PROMPT = """
Task: Identify exactly 5 real, active tech companies hiring for profiles matching this candidate.
Context:
- Candidate Profile: {candidate_profile}
- Target Keywords: {skills_query}
- Market Signals: {search_results}

Constraints:
1. Match candidate's specific job role, tech stack, location, remote preference, and experience level.
2. Avoid generic FAANG/large tech recommendations (Google, Netflix, Microsoft, Meta, Amazon) unless they are an exact match for a specific job opening in the search results. Prioritize active mid-sized companies, startups, and high-growth companies that are actually hiring for the candidate's exact tech stack, location, and seniority.
3. Recommendations must be derived from: tech stack overlap, experience level, location, and job description alignment.
4. Base choices on live hiring trends. Output reasons in 2-3 detailed sentences.

Return ONLY the raw JSON array of objects matching the required schema. No markdown formatting wrap.
"""

COVER_LETTER_PROMPT = """
Task: Write a personalized business cover letter (< 300 words) for the target role.
Context:
- Summary: {candidate_summary}
- Target JD: {job_description}

Rules:
1. No Hallucinations: Do NOT mention any tool, framework, database, or scale (e.g. millions of requests, CI/CD, Terraform, caching) absent from the candidate's summary/skills or original resume.
2. Structure: Header placeholders -> Catchy intro -> Body paragraph (highlighting 2-3 real matching technologies) -> Call to Action -> 'Sincerely, [Candidate Name]'.
"""

INTERVIEW_QUESTIONS_PROMPT = """
Task: Generate a JSON array of exactly 10 high-variety technical interview questions.
Context: Skills: {skills} | Target JD: {job_description}

Constraints:
1. Do not ask about monitoring, queue, or architecture tools absent from the candidate's skill list.
2. Focus system design on general paradigms or their explicit stack.
3. Composition: 2x System Design, 2x Scaling/Performance, 2x Database/Caching, 2x Observability/CI-CD, 2x Behavioral (STAR method).
"""

CAREER_ROADMAP_PROMPT = """
Task: Generate a JSON array of exactly 6 milestone strings mapping a 6-week roadmap.
Context: Skills: {skills} | Missing: {missing_keywords} | Target: {job_description}

Constraints:
1. Tailor for a mid-level engineer (~3 years experience).
2. Provide concrete, actionable milestone goals (e.g., "Week 1: Add one Kubernetes deployment project") instead of abstract learning terms.
3. Strings must explicitly start with "Week X: [Milestone goal]".
"""

LINKEDIN_GITHUB_OPTIMIZATION_PROMPT = """
Task: Optimize the candidate's LinkedIn and GitHub branding profiles.
Context: Summary: {candidate_summary} | Skills: {skills} | Target JD: {job_description}

Branding Rules:
1. Use ONLY technologies explicitly present in the optimized resume. Do not use missing keywords.
2. LinkedIn Headline Format: 'Job Title | Skill 1 | Skill 2 | Skill 3'. Emphasize advanced backend and AI engineering keywords (e.g. AI Agents, RAG, LangGraph, FastAPI, LangChain, Backend, Python). No generic buzzwords.
3. Suggest pinning real or suggested resume projects.

Populate all schema fields based on these constraints. No markdown commentary.
"""