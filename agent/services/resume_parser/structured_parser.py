import re
from .cleaner import ResumeCleaner

class StructuredResumeParser:

    @staticmethod
    def extract(sections):
        contact_text = sections.get("CONTACT", "") or sections.get("HEADER", "")
        experience_text = sections.get("EXPERIENCE", "")
        projects_text = sections.get("PROJECTS", "")
        education_text = sections.get("EDUCATION", "")
        skills_text = sections.get("SKILLS", "")
        languages_text = sections.get("LANGUAGES", "")
        summary_text = sections.get("SUMMARY", "")

        return {
            "contact": StructuredResumeParser.contact(contact_text),
            "summary": summary_text,
            "experience": StructuredResumeParser.parse_experience(experience_text),
            "projects": StructuredResumeParser.parse_projects(projects_text),
            "skills": StructuredResumeParser.parse_skills(skills_text),
            "education": StructuredResumeParser.parse_education(education_text),
            "languages": languages_text,
        }

    @staticmethod
    def contact(text):
        if not text:
            return {
                "name": "",
                "phone": "",
                "email": "",
                "github": "",
                "linkedin": "",
                "address": ""
            }

        # Clean spacing of input block first to make matching robust
        text_clean = ResumeCleaner.clean(text)

        # Extract Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.[\w\.-]+', text_clean)
        email = email_match.group() if email_match else ""

        # Extract Phone (10 digits)
        phone_match = re.search(r'\b\d{10}\b', text_clean)
        phone = phone_match.group() if phone_match else ""

        # Extract Github and Linkedin URLs
        github_match = re.search(r'github\.com/[a-zA-Z0-9_\-]+', text_clean, flags=re.IGNORECASE)
        github = ""
        if github_match:
            github = github_match.group()
            if not github.lower().startswith("https://"):
                github = "https://" + github

        linkedin_match = re.search(r'linkedin\.com/in/[a-zA-Z0-9_\-]+', text_clean, flags=re.IGNORECASE)
        linkedin = ""
        if linkedin_match:
            linkedin = linkedin_match.group()
            if not linkedin.lower().startswith("https://"):
                linkedin = "https://" + linkedin

        # Extract Name (usually first line of contact block)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        name = lines[0] if lines else ""

        # Extract Address (look for patterns with Zip/Pin code e.g. Surat, 395004)
        address = ""
        address_match = re.search(r'\b[a-zA-Z\t ]+,\s*\d{6}\b', text_clean)
        if address_match:
            address = address_match.group().strip()

        return {
            "name": name,
            "phone": phone,
            "email": email,
            "github": github,
            "linkedin": linkedin,
            "address": address
        }

    @staticmethod
    def detect_bullet_points(lines: list[str]) -> list[str]:
        action_verbs = {
            "Built", "Developed", "Integrated", "Implemented", "Added", "Engineered",
            "Designed", "Created", "Led", "Managed", "Configured", "Optimized",
            "Architected", "Maintained", "Collaborated", "Spearheaded", "Formulated",
            "Enhanced", "Improved", "Redesigned", "Authored", "Established", "Automated",
            "Deployed", "Monitored", "Orchestrated", "Provisioned", "Instrumented"
        }
        processed = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Already starts with a bullet point indicator
            if line_str.startswith(('-', '*', '•')) or (len(line_str) > 2 and line_str[0].isdigit() and line_str[1:3] in ('. ', ') ')):
                processed.append(line_str)
                continue

            # Treat links/GitHub labels as bullets
            if line_str.lower().startswith(('github:', 'linkedin:', 'link:', 'http', 'git hub:', 'linked in:')):
                processed.append(f"• {line_str}")
                continue

            words = line_str.split()
            if words and words[0] in action_verbs:
                processed.append(f"• {line_str}")
            else:
                processed.append(line_str)
        return processed

    @staticmethod
    def split_runon_lines(lines: list[str]) -> list[str]:
        action_verbs = {
            "Built", "Developed", "Integrated", "Implemented", "Added", "Engineered",
            "Designed", "Created", "Led", "Managed", "Configured", "Optimized",
            "Architected", "Maintained", "Collaborated", "Spearheaded", "Formulated",
            "Enhanced", "Improved", "Redesigned", "Authored", "Established", "Automated",
            "Deployed", "Monitored", "Orchestrated", "Provisioned", "Instrumented"
        }
        new_lines = []
        for line in lines:
            # Split before any action verb preceded by spaces
            pattern = r'\s+(?=' + '|'.join(action_verbs) + r')\b'
            parts = re.split(pattern, line)
            for part in parts:
                if part.strip():
                    new_lines.append(part.strip())
        return new_lines

    @staticmethod
    def parse_experience(text: str) -> list[dict]:
        if not text:
            return []
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = StructuredResumeParser.split_runon_lines(raw_lines)
        lines = StructuredResumeParser.detect_bullet_points(lines)

        experience = []
        current_entry = None

        for line in lines:
            is_bullet = line.startswith(('-', '*', '•')) or (len(line) > 2 and line[0].isdigit() and line[1:3] in ('. ', ') '))
            if is_bullet:
                # Strip leading bullet indicators
                clean_bullet = re.sub(r'^[-\*•\s\d\.\)]+', '', line).strip()
                if current_entry is None:
                    current_entry = {"experience": "Experience", "bullets": []}
                    experience.append(current_entry)
                if len(current_entry["bullets"]) < 5:  # Trim to max 5 bullets
                    current_entry["bullets"].append(clean_bullet)
            else:
                current_entry = {"experience": line, "bullets": []}
                experience.append(current_entry)

        return experience

    @staticmethod
    def parse_projects(text: str) -> list[dict]:
        if not text:
            return []
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = StructuredResumeParser.split_runon_lines(raw_lines)
        lines = StructuredResumeParser.detect_bullet_points(lines)

        projects = []
        current_project = None

        for line in lines:
            is_bullet = line.startswith(('-', '*', '•')) or (len(line) > 2 and line[0].isdigit() and line[1:3] in ('. ', ') '))
            if is_bullet:
                # Strip leading bullet indicators
                clean_bullet = re.sub(r'^[-\*•\s\d\.\)]+', '', line).strip()
                if current_project is None:
                    current_project = {"project": "Project", "bullets": []}
                    projects.append(current_project)
                if len(current_project["bullets"]) < 5:  # Trim to max 5 bullets
                    current_project["bullets"].append(clean_bullet)
            else:
                current_project = {"project": line, "bullets": []}
                projects.append(current_project)

        return projects

    @staticmethod
    def parse_education(text: str) -> list[str]:
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def parse_skills(text: str) -> dict:
        categories = {
            "languages": [],
            "frameworks": [],
            "databases": [],
            "vector_db": [],
            "tools": [],
            "ai": []
        }
        if not text:
            return categories

        skill_map = {
            "languages": {"python", "javascript", "typescript", "c++", "c#", "java", "golang", "ruby", "rust", "php", "sql", "bash", "html", "css", "c"},
            "frameworks": {"fastapi", "django", "flask", "next.js", "nextjs", "react", "vue", "angular", "node", "express", "spring", "laravel"},
            "databases": {"postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle", "cassandra", "mariadb", "dynamodb"},
            "vector_db": {"pinecone", "qdrant", "chroma", "milvus", "weaviate"},
            "tools": {"docker", "kubernetes", "git", "github actions", "gitlab ci", "terraform", "jenkins", "ansible", "aws", "gcp", "azure"},
            "ai": {"openai", "langchain", "langgraph", "llm", "llama", "pytorch", "tensorflow", "scikit-learn", "rag", "mcp"}
        }

        # Split skills by commas, semicolons, newlines, or bullets
        raw_skills = [s.strip() for s in re.split(r'[,;\n•\-\*]', text) if s.strip()]

        for s in raw_skills:
            if ":" in s:
                parts = s.split(":", 1)
                header = parts[0].lower().strip()
                skills_sub = [sk.strip() for sk in re.split(r'[,;]', parts[1]) if sk.strip()]

                matched_cat = None
                for cat in categories.keys():
                    if cat in header or header in cat:
                        matched_cat = cat
                        break
                if matched_cat:
                    categories[matched_cat].extend(skills_sub)
                    continue
                else:
                    for sk in skills_sub:
                        StructuredResumeParser._categorize_single_skill(sk, skill_map, categories)
            else:
                StructuredResumeParser._categorize_single_skill(s, skill_map, categories)

        # Deduplicate and sort list values
        for cat in categories:
            categories[cat] = sorted(list(set(categories[cat])), key=lambda x: x.lower())

        return categories

    @staticmethod
    def _categorize_single_skill(skill: str, skill_map: dict, categories: dict):
        skill_clean = skill.lower().strip()
        if not skill_clean:
            return

        # 1. Exact match across all categories first
        for cat, skills_set in skill_map.items():
            if skill_clean in skills_set:
                categories[cat].append(skill)
                return

        # 2. Word boundary match
        for cat, skills_set in skill_map.items():
            for s in skills_set:
                if re.search(r'\b' + re.escape(s) + r'\b', skill_clean):
                    categories[cat].append(skill)
                    return

        # 3. Fallback
        categories["languages"].append(skill)
