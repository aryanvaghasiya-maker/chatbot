from typing import Literal, List, Any, Type
import re
import asyncio
from agent.services.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from agent.config.prompts import RESUME_OPTIMIZATION_PROMPT
from agent.states.states import AdvancedAgentState
from agent.schema.schema import OptimizedResumeOutput, SlimOptimizedResumeOutput, SuggestedProject, ExperienceEntry, ProjectEntry, Certification


def post_process_optimized_resume(
    optimized_data: OptimizedResumeOutput,
    raw_skills: List[str],
    missing_keywords: List[str],
    original_experience: List[Any],
    original_projects: List[Any],
    original_certifications: List[Any],
    job_description: str,
    original_contact: dict,
    generate_suggested_projects: bool = True,
    raw_resume: str = ""
):
    # Preserve original contact details verbatim (especially location, phone, email, name, github, linkedin, portfolio)
    if original_contact:
        for key in ["name", "email", "phone", "location", "linkedin", "github", "portfolio", "title"]:
            orig_val = original_contact.get(key, "").strip()
            if orig_val:
                setattr(optimized_data.contact, key, orig_val)

    # Categorization sets for skill preservation
    LANGS = {"python", "javascript", "typescript", "c++", "c#", "java", "golang", "ruby", "rust", "php", "sql", "bash", "html", "css"}
    BACKEND = {"fastapi", "django", "flask", "node", "express", "next.js", "spring", "laravel", "gpc", "uwsgi", "gunicorn", "graphql", "grpc"}
    AI_LLM = {"openai", "langchain", "langgraph", "llm", "llama", "pytorch", "tensorflow", "scikit-learn", "mcp", "rag", "pinecone", "qdrant", "vector db", "prompt engineering", "agentic"}
    DATABASES = {"postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle", "cassandra", "mariadb", "dynamodb", "caching"}
    CLOUD = {"aws", "gcp", "azure", "ecs", "eks", "s3", "ec2", "rds", "lambda", "cloudfront"}
    DEVOPS = {"docker", "kubernetes", "helm", "git", "github actions", "gitlab ci", "terraform", "jenkins", "ansible", "docker compose", "gitops", "argocd"}
    MESSAGING = {"rabbitmq", "kafka", "celery", "activemq", "sqs", "sns"}
    MONITORING = {"prometheus", "grafana", "elk", "logstash", "kibana", "datadog", "sentry", "loki", "splunk", "monitoring", "logging", "observability"}
    TESTING = {"pytest", "unittest", "selenium", "cypress", "playwright", "mock"}
    ARCHITECTURE = {"distributed systems", "microservices", "system design", "rest api", "rest apis", "soas", "event-driven"}

    # 1. Enforce Skill Preservation, Categorization, and Anti-Hallucination
    # Strictly allow ONLY skills present in the candidate's original resume raw_skills list
    allowed_skills_set = {s.lower().strip() for s in raw_skills}
    
    # Helper to check if a skill is allowed (case-insensitive fuzzy match against original raw skills only)
    def is_skill_allowed(skill: str) -> bool:
        skill_clean = skill.lower().strip()
        if not skill_clean:
            return False
        # Exact match
        if skill_clean in allowed_skills_set:
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
    for cat in ["languages", "backend", "ai_llm", "databases", "cloud", "devops", "messaging", "monitoring", "testing", "architecture", "other"]:
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
            elif raw_clean in BACKEND or any(b in raw_clean for b in BACKEND):
                if not optimized_data.skills.backend:
                    optimized_data.skills.backend = []
                if raw_s not in optimized_data.skills.backend:
                    optimized_data.skills.backend.append(raw_s)
            elif raw_clean in AI_LLM or any(ai in raw_clean for ai in AI_LLM):
                if not optimized_data.skills.ai_llm:
                    optimized_data.skills.ai_llm = []
                if raw_s not in optimized_data.skills.ai_llm:
                    optimized_data.skills.ai_llm.append(raw_s)
            elif raw_clean in DATABASES:
                if not optimized_data.skills.databases:
                    optimized_data.skills.databases = []
                if raw_s not in optimized_data.skills.databases:
                    optimized_data.skills.databases.append(raw_s)
            elif raw_clean in CLOUD or any(c in raw_clean for c in CLOUD):
                if not optimized_data.skills.cloud:
                    optimized_data.skills.cloud = []
                if raw_s not in optimized_data.skills.cloud:
                    optimized_data.skills.cloud.append(raw_s)
            elif raw_clean in DEVOPS or any(d in raw_clean for d in DEVOPS):
                if not optimized_data.skills.devops:
                    optimized_data.skills.devops = []
                if raw_s not in optimized_data.skills.devops:
                    optimized_data.skills.devops.append(raw_s)
            elif raw_clean in MESSAGING or any(m in raw_clean for m in MESSAGING):
                if not optimized_data.skills.messaging:
                    optimized_data.skills.messaging = []
                if raw_s not in optimized_data.skills.messaging:
                    optimized_data.skills.messaging.append(raw_s)
            elif raw_clean in MONITORING or any(mo in raw_clean for mo in MONITORING):
                if not optimized_data.skills.monitoring:
                    optimized_data.skills.monitoring = []
                if raw_s not in optimized_data.skills.monitoring:
                    optimized_data.skills.monitoring.append(raw_s)
            elif raw_clean in TESTING or any(t in raw_clean for t in TESTING):
                if not optimized_data.skills.testing:
                    optimized_data.skills.testing = []
                if raw_s not in optimized_data.skills.testing:
                    optimized_data.skills.testing.append(raw_s)
            elif raw_clean in ARCHITECTURE or any(ar in raw_clean for ar in ARCHITECTURE):
                if not optimized_data.skills.architecture:
                    optimized_data.skills.architecture = []
                if raw_s not in optimized_data.skills.architecture:
                    optimized_data.skills.architecture.append(raw_s)
            else:
                if not optimized_data.skills.other:
                    optimized_data.skills.other = []
                if raw_s not in optimized_data.skills.other:
                    optimized_data.skills.other.append(raw_s)

    # Helper for fuzzy company and project matching
    def is_fuzzy_match(s1: str, s2: str) -> bool:
        clean1 = s1.lower().strip()
        clean2 = s2.lower().strip()
        if clean1 in clean2 or clean2 in clean1:
            return True
        set1 = set(clean1)
        set2 = set(clean2)
        intersection = set1 & set2
        union = set1 | set2
        if union:
            jaccard = len(intersection) / len(union)
            if jaccard > 0.75:
                return True
        return False

    # 2. Experience Preservation
    opt_companies = [exp.company.lower().strip() for exp in (optimized_data.experience or [])]
    for orig_exp in (original_experience or []):
        orig_company = orig_exp.get("company", "").lower().strip()
        if not orig_company:
            continue
        if not any(is_fuzzy_match(orig_company, opt_c) for opt_c in opt_companies):
            restored_entry = ExperienceEntry(**orig_exp)
            if not optimized_data.experience:
                optimized_data.experience = []
            optimized_data.experience.append(restored_entry)
            opt_companies.append(orig_company)

    # 3. Projects Preservation & ATS Keyword Retainment
    for opt_proj in (optimized_data.projects or []):
        opt_name = opt_proj.name.lower().strip()
        match_orig = None
        for orig_proj in (original_projects or []):
            orig_name = orig_proj.get("name", "").lower().strip()
            if is_fuzzy_match(opt_name, orig_name):
                match_orig = orig_proj
                break
        
        if match_orig:
            orig_desc = match_orig.get("description", "")
            # Ensure critical ATS keywords from original project are preserved
            critical_keywords = [
                "sql safety", "error correction", "groq integration", "groq",
                "multi-agent workflow", "multi-agent", "semantic chunking", "chunking",
                "rag", "retrieval-augmented generation", "latency", "vector search",
                "tool-calling", "agentic", "inference", "api", "apis"
            ]
            opt_desc_lower = opt_proj.description.lower()
            orig_desc_lower = orig_desc.lower()
            
            missing_criticals = []
            for kw in critical_keywords:
                if kw in orig_desc_lower and kw not in opt_desc_lower:
                    missing_criticals.append(kw)
            
            if missing_criticals:
                bullets = [b.strip() for b in orig_desc.split('\n') if b.strip()]
                opt_bullets = [b.strip() for b in opt_proj.description.split('\n') if b.strip()]
                
                for b in bullets:
                    b_lower = b.lower()
                    if any(kw in b_lower for kw in missing_criticals):
                        clean_b = b.lstrip("-*• ").strip()
                        if clean_b not in opt_bullets:
                            opt_bullets.append(clean_b)
                
                opt_proj.description = "\n".join(opt_bullets[:3])

    opt_projects = [proj.name.lower().strip() for proj in (optimized_data.projects or [])]
    for orig_proj in (original_projects or []):
        orig_name = orig_proj.get("name", "").lower().strip()
        if not orig_name:
            continue
        if not any(is_fuzzy_match(orig_name, opt_p) for opt_p in opt_projects):
            restored_entry = ProjectEntry(**orig_proj)
            if not optimized_data.projects:
                optimized_data.projects = []
            optimized_data.projects.append(restored_entry)
            opt_projects.append(orig_name)

    # Anti-Hallucination summary check (clean untruthful buzzwords)
    if optimized_data.summary:
        summary_clean = optimized_data.summary
        hallucinations = [
            ("cloud platforms", ""),
            ("cloud platform", ""),
            ("ci/cd", ""),
            ("production-ready", ""),
            ("production ready", ""),
            ("devops", ""),
        ]
        allowed_techs = {s.lower().strip() for s in (raw_skills + missing_keywords)}
        for term, repl in hallucinations:
            if term not in allowed_techs:
                summary_clean = re.sub(rf'\b{term}\b', repl, summary_clean, flags=re.IGNORECASE)
        
        summary_clean = re.sub(r'\s*,\s*,', ',', summary_clean)
        summary_clean = re.sub(r',\s*and\s*,', ' and', summary_clean)
        summary_clean = re.sub(r'\s+', ' ', summary_clean).strip()
        summary_clean = summary_clean.replace(" .", ".").replace(" ,", ",")
        optimized_data.summary = summary_clean

    # 4. Certifications Preservation & Strict Anti-Hallucination
    orig_cert_names = {c.get("name", "").lower().strip() for c in (original_certifications or [])}
    filtered_certs = []
    
    # Strictly allow ONLY certifications that are in the original certifications list
    for cert in (optimized_data.certifications or []):
        cert_name = cert.name.lower().strip()
        if any(cert_name in o_cert or o_cert in cert_name for o_cert in orig_cert_names):
            if not any(cert_name in c.name.lower() or c.name.lower() in cert_name for c in filtered_certs):
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
    if generate_suggested_projects:
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
    else:
        optimized_data.suggested_projects = []

    # === Strict Anti-Hallucination & Anti-Fabrication Post-Processing ===
    
    # Construct original text blob to match metrics/numbers against
    original_text_blob = raw_resume.lower()
    for exp in original_experience or []:
        for bullet in exp.get("bullets", []):
            original_text_blob += " " + bullet.lower()
    for proj in original_projects or []:
        original_text_blob += " " + proj.get("description", "").lower()

    def clean_fabricated_metrics(text: str) -> str:
        # Find all percentages (e.g. 40%, 30%)
        percentages = re.findall(r'\b\d+(?:\.\d+)?\s*%', text)
        cleaned = text
        for pct in percentages:
            pct_clean = pct.lower().replace(" ", "")
            orig_clean = original_text_blob.replace(" ", "")
            if pct_clean not in orig_clean:
                cleaned = re.sub(r'\b(?:by|of|to)\s+' + re.escape(pct) + r'\b', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\b' + re.escape(pct) + r'\s+(?:increase|decrease|improvement|reduction|boost|scaling|efficiency|gain|gains|growth)\b', 'substantial improvement', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace(pct, '')
                
        # Regex to find numbers/scales >= 10: e.g. 10k, 1M, 500+
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:k|m|b)?\s*\+?\b', text, flags=re.IGNORECASE)
        for num in numbers:
            num_clean = num.lower().strip().replace(" ", "")
            if num_clean.isdigit() and int(num_clean) < 10:
                continue
            if num_clean not in original_text_blob.replace(" ", ""):
                cleaned = re.sub(r'\b' + re.escape(num) + r'\s*(?:requests|users|transactions|jobs|messages|queries|events|customers|visits|sessions|records|files|endpoints|orders)\b(?:\s*/\s*(?:sec|second|min|minute|hr|hour|day))?', 'concurrent requests', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace(num, '')

        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:?!])', r'\1', cleaned)
        return cleaned.strip()

    def clean_fabricated_technologies(text: str) -> str:
        tech_map = {
            "kubernetes": ["kubernetes", "k8s", "helm"],
            "aws": ["aws", "amazon web services", "s3", "ec2", "rds", "lambda", "ecs", "eks", "sqs", "sns", "cloudfront"],
            "rabbitmq": ["rabbitmq"],
            "kafka": ["kafka"],
            "terraform": ["terraform"],
            "prometheus": ["prometheus"],
            "grafana": ["grafana"],
            "pytest": ["pytest"],
            "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment", "github actions", "jenkins", "gitlab ci"],
            "distributed systems": ["distributed systems", "distributed system"],
            "cloud architecture": ["cloud architecture", "cloud-native", "cloud infrastructure"],
            "microservices": ["microservices", "microservice", "micro-services"],
        }
        
        cleaned = text
        for tech, aliases in tech_map.items():
            has_tech = False
            for s in allowed_skills_set:
                s_lower = s.lower().strip()
                if tech in s_lower or any(a in s_lower for a in aliases):
                    has_tech = True
                    break
            
            if not has_tech:
                for alias in aliases:
                    pattern1 = r'\b(?:and|or|,)?\s*' + re.escape(alias) + r'\b'
                    pattern2 = r'\b' + re.escape(alias) + r'\s*(?:and|or|,)?\s*'
                    cleaned = re.sub(pattern1, '', cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(pattern2, '', cleaned, flags=re.IGNORECASE)
                    
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:?!])', r'\1', cleaned)
        cleaned = cleaned.strip().strip(",").strip()
        return cleaned

    # Helper to calculate years of experience from experience entries
    def calculate_years_experience(entries) -> float:
        import re
        from datetime import datetime
        total_months = 0
        for entry in entries:
            start_str = entry.get("start_date", "") or ""
            end_str = entry.get("end_date", "") or ""
            if not start_str:
                continue
            
            start_year_match = re.search(r'\b(19\d\d|20\d\d)\b', start_str)
            if not start_year_match:
                continue
            start_year = int(start_year_match.group(1))
            
            start_month = 1
            months_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
            for m_name, m_val in months_map.items():
                if m_name in start_str.lower():
                    start_month = m_val
                    break
                    
            end_year = datetime.now().year
            end_month = datetime.now().month
            if end_str and "present" not in end_str.lower():
                end_year_match = re.search(r'\b(19\d\d|20\d\d)\b', end_str)
                if end_year_match:
                    end_year = int(end_year_match.group(1))
                    for m_name, m_val in months_map.items():
                        if m_name in end_str.lower():
                            end_month = m_val
                            break
            
            diff_months = (end_year - start_year) * 12 + (end_month - start_month)
            if diff_months > 0:
                total_months += diff_months
                
        return max(0.5, total_months / 12.0)

    # 1. Clean professional title & enforce seniority
    if optimized_data.contact and optimized_data.contact.title:
        years_exp = calculate_years_experience(original_experience)
        final_title = (optimized_data.contact.title or "").strip()
        title_lower = final_title.lower()
        
        # Enforce seniority titles
        if years_exp < 2.0:
            if any(term in title_lower for term in ["senior", "lead", "principal", "ii"]):
                if "ai" in title_lower or "llm" in title_lower or "ml" in title_lower:
                    final_title = "AI/ML Engineer"
                else:
                    final_title = "Backend Engineer"
        elif years_exp < 5.0:
            if any(term in title_lower for term in ["senior", "lead", "principal"]):
                if "ai" in title_lower or "llm" in title_lower or "ml" in title_lower:
                    final_title = "ML Engineer"
                else:
                    final_title = "Software Engineer II"
        else:
            if not any(term in title_lower for term in ["senior", "lead", "principal"]):
                if "ai" in title_lower or "llm" in title_lower or "ml" in title_lower:
                    final_title = "Senior AI/ML Engineer"
                else:
                    final_title = "Senior Backend Engineer"
                    
        title_parts = [p.strip() for p in re.split(r'[|/,]', final_title)]
        cleaned_parts = []
        for part in title_parts:
            part_words = re.findall(r'\b\w+\b', part.lower())
            is_valid = True
            tech_keywords = {"aws", "kubernetes", "k8s", "rabbitmq", "kafka", "terraform", "prometheus", "grafana", "pytest", "ci/cd", "cicd", "microservices", "distributed systems", "cloud architecture"}
            for word in part_words:
                if word in tech_keywords and not is_skill_allowed(word):
                    is_valid = False
                    break
            if is_valid:
                cleaned_parts.append(part)
                
        if cleaned_parts:
            optimized_data.contact.title = " | ".join(cleaned_parts)
        else:
            optimized_data.contact.title = "Backend Engineer"

    # Enforce Summary is strictly 3-4 sentences
    if optimized_data.summary:
        sentences = re.split(r'(?<=[.!?])\s+', optimized_data.summary.strip())
        if len(sentences) > 4:
            optimized_data.summary = " ".join(sentences[:4])
        elif len(sentences) < 3:
            # Pad to 3 sentences if too short
            optimized_data.summary = optimized_data.summary.strip() + " Strong background in scalable backend design and API architecture."

    # 2. Clean core competencies & enforce exactly 8 items
    soft_skills_patterns = {"team player", "communication", "problem solving", "detail-oriented", "hard worker", "fast learner", "flexible"}
    filtered_competencies = []
    if optimized_data.core_competencies:
        for comp in optimized_data.core_competencies:
            comp_words = re.findall(r'\b\w+\b', comp.lower())
            is_valid = True
            tech_keywords = {"aws", "kubernetes", "k8s", "rabbitmq", "kafka", "terraform", "prometheus", "grafana", "pytest", "ci/cd", "cicd", "microservices", "distributed systems", "cloud architecture"}
            for word in comp_words:
                if word in tech_keywords and not is_skill_allowed(word):
                    is_valid = False
                    break
            if is_valid and comp.lower().strip() not in soft_skills_patterns:
                filtered_competencies.append(comp)

    # Slice or pad to exactly 8
    if len(filtered_competencies) > 8:
        optimized_data.core_competencies = filtered_competencies[:8]
    else:
        # Pad with allowed tech skills
        available_skills = [s for s in optimized_data.all_skills() if s not in filtered_competencies][:8 - len(filtered_competencies)]
        combined = filtered_competencies + available_skills
        if len(combined) < 8:
            defaults = ["Backend Development", "System Design", "API Design", "Distributed Systems", "Database Management", "Software Engineering", "Microservices", "Cloud Integration"]
            for d in defaults:
                if len(combined) >= 8:
                    break
                if d not in combined:
                    combined.append(d)
        optimized_data.core_competencies = combined

    # Enforce achievements section is omitted if empty/not grounded in original resume
    has_orig_achievements = False
    for keyword in ["achievement", "award", "highlight", "accomplishment"]:
        if keyword in raw_resume.lower():
            has_orig_achievements = True
            break
    if not has_orig_achievements:
        optimized_data.achievements = []

    # 3. Clean experience entry bullets
    if optimized_data.experience:
        for exp in optimized_data.experience:
            cleaned_bullets = []
            for bullet in exp.bullets:
                cb = clean_fabricated_technologies(bullet)
                cb = clean_fabricated_metrics(cb)
                if cb:
                    cb = cb[0].upper() + cb[1:]
                    cb = cb.strip().strip(",").strip(".") + "."
                    cb_parts = [p.strip() for p in cb.split(" ") if p.strip()]
                    if len(cb_parts) >= 3:
                        cleaned_bullets.append(cb)
            exp.bullets = cleaned_bullets

    # 4. Clean project descriptions and tech stacks
    if optimized_data.projects:
        for proj in optimized_data.projects:
            if "\n" in proj.description:
                desc_bullets = proj.description.split("\n")
                cleaned_desc_bullets = []
                for b in desc_bullets:
                    cb = clean_fabricated_technologies(b)
                    cb = clean_fabricated_metrics(cb)
                    if cb:
                        cleaned_desc_bullets.append(cb)
                proj.description = "\n".join(cleaned_desc_bullets)
            else:
                proj.description = clean_fabricated_metrics(clean_fabricated_technologies(proj.description))
            
            if proj.tech_stack:
                cleaned_tech = []
                for t in proj.tech_stack:
                    if is_skill_allowed(t):
                        cleaned_tech.append(t)
                proj.tech_stack = cleaned_tech

    # 5. Clean achievements
    if optimized_data.achievements:
        cleaned_achievements = []
        for ach in optimized_data.achievements:
            cleaned_ach = clean_fabricated_technologies(ach)
            cleaned_ach = clean_fabricated_metrics(cleaned_ach)
            if cleaned_ach:
                cleaned_ach = cleaned_ach[0].upper() + cleaned_ach[1:]
                cleaned_ach = cleaned_ach.strip().strip(",").strip(".") + "."
                cleaned_ach_parts = [p.strip() for p in cleaned_ach.split(" ") if p.strip()]
                if len(cleaned_ach_parts) >= 3:
                    cleaned_achievements.append(cleaned_ach)
        optimized_data.achievements = cleaned_achievements



class check_optimization_quality:
    def __init__(self, max_loops: int = 3, target_score: int = 90):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = get_llm()

    def _get_resume_text(self, optimized) -> str:
        if not optimized:
            return ""
        summary = getattr(optimized, "summary", "") or ""
        if isinstance(optimized, dict):
            summary = optimized.get("summary", "") or ""
            
        bullets = []
        if hasattr(optimized, "all_bullets"):
            bullets = optimized.all_bullets()
        elif isinstance(optimized, dict):
            for exp in optimized.get("experience", []):
                bullets.extend(exp.get("bullets", []) or [])
        else:
            for exp in getattr(optimized, "experience", []):
                bullets.extend(getattr(exp, "bullets", []) or [])
                
        skills = []
        if hasattr(optimized, "all_skills"):
            skills = optimized.all_skills()
        elif isinstance(optimized, dict):
            sk = optimized.get("skills", {}) or {}
            skills = (
                (sk.get("backend", []) or []) +
                (sk.get("ai_llm", []) or []) +
                (sk.get("databases", []) or []) +
                (sk.get("cloud", []) or []) +
                (sk.get("devops", []) or [])
            )
        else:
            sk_obj = getattr(optimized, "skills", None)
            if sk_obj:
                skills = (
                    (getattr(sk_obj, "backend", []) or []) +
                    (getattr(sk_obj, "ai_llm", []) or []) +
                    (getattr(sk_obj, "databases", []) or []) +
                    (getattr(sk_obj, "cloud", []) or []) +
                    (getattr(sk_obj, "devops", []) or [])
                )
        return f"{summary}\n{'/'.join(bullets)}\n{'/'.join(skills)}"

    def check_optimization_quality(self, state: AdvancedAgentState) -> Literal["gap_analysis", "inject_suggestions"]:
        from difflib import SequenceMatcher
        
        ats_score = state.get("ats_score", 0)
        iterations = state.get("iterations", 0)
        history = state.get("history", []) or []
        token_usage = state.get("token_usage", {}) or {}
        total_tokens = token_usage.get("total", 0)
        
        # 1. Stop if target loops reached
        if iterations >= self.max_loops:
            print(f"[Loop Controller]: Stop - Max loops reached ({iterations} >= {self.max_loops}). Proceeding to Final Asset Generation.")
            return "inject_suggestions"

        # 2. Stop if ATS score is already high
        if ats_score >= self.target_score:
            print(f"[Loop Controller]: Stop - Target ATS score reached ({ats_score} >= {self.target_score}). Proceeding to Final Asset Generation.")
            return "inject_suggestions"

        # 3. Stop if Token Budget exceeded
        if total_tokens >= 10000:
            print(f"[Loop Controller]: Stop - Token budget exceeded ({total_tokens} >= 10000). Proceeding to Final Asset Generation.")
            return "inject_suggestions"

        # 4. Stop if cumulative keyword additions exceed prevent-stuffing limit (MAX_NEW_KEYWORDS = 10)
        cumulative_kws = sum(h.get("changes", {}).get("keywords_added", 0) for h in history)
        if cumulative_kws >= 10:
            print(f"[Loop Controller]: Stop - Keyword stuffing limit reached ({cumulative_kws} >= 10). Proceeding to Final Asset Generation.")
            return "inject_suggestions"

        # 5. Stop if no meaningful improvements in the latest iteration
        if len(history) >= 1:
            latest_change = history[-1].get("changes", {}) or {}
            keywords_added = latest_change.get("keywords_added", 0)
            metrics_added = latest_change.get("metrics_added", 0)
            sections_updated = latest_change.get("sections_updated", []) or []
            if keywords_added == 0 and metrics_added == 0 and len(sections_updated) == 0:
                print(f"[Loop Controller]: Stop - No meaningful improvements in iteration (keywords: 0, metrics: 0, sections: 0). Proceeding to Final Asset Generation.")
                return "inject_suggestions"

        # 6. Stop if minimal score gain between consecutive iterations (MIN_SCORE_GAIN = 2)
        if len(history) >= 2:
            current_score = history[-1].get("score", 0)
            previous_score = history[-2].get("score", 0)
            if current_score - previous_score < 2:
                print(f"[Loop Controller]: Stop - Score improvement gain too small ({current_score} - {previous_score} < 2). Proceeding to Final Asset Generation.")
                return "inject_suggestions"

        # 7. Stop if resume changes are minimal (MIN_RESUME_DIFF = 5%)
        if len(history) >= 2:
            curr_text = self._get_resume_text(state.get("optimized_resume"))
            prev_text = self._get_resume_text(history[-2].get("resume"))
            if curr_text and prev_text:
                ratio = SequenceMatcher(None, curr_text, prev_text).ratio()
                change_pct = (1.0 - ratio) * 100
                if change_pct < 5.0:
                    print(f"[Loop Controller]: Stop - Resume change percentage too small ({change_pct:.2f}% < 5%). Proceeding to Final Asset Generation.")
                    return "inject_suggestions"

        # 8. Stop if duplicate suggestions are repeated
        if len(history) >= 2:
            curr_improvements = set(state.get("improvement_changes", []) or [])
            prev_improvements = set(history[-2].get("improvement_changes", []) or [])
            if curr_improvements and curr_improvements == prev_improvements:
                print(f"[Loop Controller]: Stop - Repeated suggestions detected. Proceeding to Final Asset Generation.")
                return "inject_suggestions"

        # 9. Stop if Final Quality Gate is met: ATS >= 90, Grammar >= 95, Formatting >= 90, Keyword Match >= 95
        breakdown = state.get("ats_breakdown", {}) or {}
        grammar = breakdown.get("grammar", 0)
        formatting = breakdown.get("formatting", 0)
        keyword_match = breakdown.get("keyword_match", 0)
        if ats_score >= 90 and grammar >= 95 and formatting >= 90 and keyword_match >= 95:
            print(f"[Loop Controller]: Stop - Final quality gate criteria satisfied (ATS >= 90, Grammar >= 95, Formatting >= 90, Keyword Match >= 95). Proceeding to Final Asset Generation.")
            return "inject_suggestions"

        print(f"[Loop Controller]: Quality Rejected ({ats_score}/100, Loops: {iterations}). Re-routing to Gap Analysis.")
        return "gap_analysis"


class optimize_resume_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score
        self.llm = get_llm()

    async def _invoke_with_retry(
        self,
        schema: Type,
        messages: list,
        max_retries: int = 3,
    ) -> dict:
        """
        Invoke structured output with exponential backoff retry.
        Handles tool_use_failed (400) and rate-limit (429) errors.
        Returns the raw response dict from include_raw=True.
        """
        structured_llm = self.llm.with_structured_output(
            schema, method="function_calling", include_raw=True
        )
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await structured_llm.ainvoke(messages)
                # Check if parsing succeeded
                if result.get("parsed") is not None:
                    return result
                # parsed is None means the model returned invalid JSON
                err_str = str(result.get("parsing_error", "unknown parsing error"))
                print(f"[Optimizer] Attempt {attempt}/{max_retries}: parsed=None, error={err_str[:120]}")
                last_err = err_str
            except Exception as exc:
                err_str = str(exc)
                print(f"[Optimizer] Attempt {attempt}/{max_retries}: exception={err_str[:200]}")
                last_err = exc
                # tool_use_failed (400) — schema may be too large; no point retrying identically
                if "tool_use_failed" in err_str or "400" in err_str:
                    print("[Optimizer] tool_use_failed detected — aborting retries for this schema.")
                    break

            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s
                print(f"[Optimizer] Waiting {wait}s before retry...")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Structured output failed after {max_retries} attempts. Last error: {last_err}")

    async def optimize_resume_node(self, state: AdvancedAgentState):
        total_tokens = state.get("token_usage", {}).get("total", 0) if state.get("token_usage") else 0
        if total_tokens >= 10000:
            print(f"[Optimizer]: Token budget exceeded ({total_tokens} >= 10000). Skipping LLM invocation.")
            return {}

        generate_suggested_projects = state.get("generate_suggested_projects", True)

        # Choose the slimmest schema that satisfies the request to avoid token overflow
        output_schema: Type = OptimizedResumeOutput if generate_suggested_projects else SlimOptimizedResumeOutput
        print(f"[Optimizer]: Using schema={'OptimizedResumeOutput' if generate_suggested_projects else 'SlimOptimizedResumeOutput'} (suggested_projects={generate_suggested_projects})")

        critique_history = state.get("critique_history", [])
        history_context = critique_history[-1] if critique_history else ""

        user_content_parts = [
            f"=== RAW RESUME ===\n{state.get('raw_resume', '')}",
            f"=== TARGET JOB DESCRIPTION ===\n{state.get('job_description', '')}",
            f"=== EXTRACTED SKILLS ===\n{', '.join(state.get('extracted_skills', []))}",
        ]
        if history_context:
            user_content_parts.append(f"=== PREVIOUS CRITIQUE (address these gaps) ===\n{history_context}")
        if state.get('missing_keywords'):
            user_content_parts.append(f"=== MISSING KEYWORDS TO INJECT ===\n{', '.join(state.get('missing_keywords', []))}") 
        if not generate_suggested_projects:
            user_content_parts.append("NOTE: Do NOT generate suggested_projects. Leave that field empty.")

        user_content = "\n\n".join(user_content_parts)
        messages = [
            SystemMessage(content=RESUME_OPTIMIZATION_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = await self._invoke_with_retry(output_schema, messages)
        except RuntimeError as err:
            # If slim schema also fails, propagate error up gracefully
            errors = state.get("errors", []) or []
            errors.append({"module": "optimizer", "message": str(err)[:300]})
            print(f"[Optimizer] FATAL: {err}")
            return {"errors": errors, "status": "partial_success"}

        raw_result = response.get("parsed")
        raw_msg = response.get("raw")

        # If we used the slim schema, convert it to OptimizedResumeOutput
        if isinstance(raw_result, SlimOptimizedResumeOutput):
            optimized_data: OptimizedResumeOutput = raw_result.to_full()
        else:
            optimized_data: OptimizedResumeOutput = raw_result

        if optimized_data is None:
            errors = state.get("errors", []) or []
            errors.append({"module": "optimizer", "message": "LLM returned None for optimized resume — skipping."})
            return {"errors": errors, "status": "partial_success"}

        # Post-process optimized resume for safety, completeness, and skill preservation
        post_process_optimized_resume(
            optimized_data,
            state.get("extracted_skills", []),
            state.get("missing_keywords", []),
            state.get("original_experience", []),
            state.get("original_projects", []),
            state.get("original_certifications", []),
            state.get("job_description", ""),
            state.get("original_contact", {}),
            generate_suggested_projects=generate_suggested_projects,
            raw_resume=state.get("raw_resume", "")
        )

        token_usage = {"input": 0, "output": 0, "total": 0}
        if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
            token_usage = {
                "input": raw_msg.usage_metadata.get("input_tokens", 0),
                "output": raw_msg.usage_metadata.get("output_tokens", 0),
                "total": raw_msg.usage_metadata.get("total_tokens", 0),
            }

        return {
            "optimized_resume": optimized_data,
            "iterations": state.get("iterations", 0) + 1,
            "token_usage": token_usage,
        }
