import re

class ResumeCleaner:

    @staticmethod
    def clean(text: str) -> str:
        # Strip ASCII control characters (except \n, \r, \t) to avoid LaTeX compilation issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 1. OCR Confidence check / Abnormal spacing check
        abnormal = ResumeCleaner.detect_abnormal_spacing(text)
        if abnormal:
            pass

        # 2. Fix broken/merged words
        text = ResumeCleaner.fix_broken_words(text)

        # 3. Fix missing spaces (transitions, labels, camelCase)
        text = ResumeCleaner.fix_missing_spaces(text)

        # 4. Fix dates (e.g. Aug2022 -> Aug 2022)
        text = ResumeCleaner.fix_dates(text)

        # 5. Fix character spacing (letters and alphanumeric spacings)
        # Run this BEFORE fix_digits so alphanumeric spacing like "7 0 B" is completely joined to "70B"
        text = ResumeCleaner.fix_character_spacing(text)

        # 6. Fix digits / phone numbers / ZIP codes (e.g. 395 0 0 4 -> 395004)
        text = ResumeCleaner.fix_digits(text)

        # 7. Fix emails
        text = ResumeCleaner.fix_emails(text)

        # 8. Fix URLs
        text = ResumeCleaner.fix_urls(text)

        # 9. Normalize Hyphens
        text = ResumeCleaner.fix_hyphens(text)

        # 10. Normalize section headers newlines
        text = ResumeCleaner.fix_section_headers(text)

        # 11. Normalize other punctuation
        text = ResumeCleaner.fix_punctuation(text)

        # 12. Clean extra whitespace
        # Strip trailing/leading spaces on lines first, then collapse multiple blank lines
        text = ResumeCleaner.remove_blank_lines(text)
        text = ResumeCleaner.remove_extra_spaces(text)
        return text.strip()

    @staticmethod
    def detect_abnormal_spacing(text: str) -> bool:
        words = text.split()
        if not words:
            return False
        spaced_count = sum(1 for w in words if len(w) == 1 and w.isalnum())
        pct = (spaced_count / len(words)) * 100
        return pct > 20.0

    @staticmethod
    def fix_character_spacing(text: str) -> str:
        lines = []
        for line in text.splitlines():
            # Match sequences of single alphanumeric characters separated by single spaces
            # e.g., "D A R S H A N", "8 3 2 0 4 7 7 6 0 5", "2 0 2 6", "7 0 B"
            pattern = r'\b[A-Za-z0-9](?:\s[A-Za-z0-9])+\b'
            line = re.sub(
                pattern,
                lambda m: m.group().replace(" ", ""),
                line
            )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def fix_emails(text: str) -> str:
        # Match email address patterns with internal spaces
        # e.g., "darshanjain 2 2 0 2 @ gmail. com" -> "darshanjain2202@gmail.com"
        pattern = r'\b[a-zA-Z0-9._%+-]+(?:\s+[a-zA-Z0-9._%+-]+)*\s*@\s*[a-zA-Z0-9.-]+(?:\s+[a-zA-Z0-9.-]+)*\s*\.\s*[a-zA-Z]{2,4}\b'
        return re.sub(pattern, lambda m: m.group().replace(" ", ""), text)

    @staticmethod
    def fix_urls(text: str) -> str:
        # Pre-replace protocol splits
        replacements = {
            "h t t p s : / /": "https://",
            "h t t p : / /": "http://",
            "g i t h u b": "github",
            "l i n k e d i n": "linkedin",
            ". c o m": ".com",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove spaces in URL paths (e.g. "github.com / Darshan273 / SQL - RAG-text2sql" -> "github.com/Darshan273/SQL-RAG-text2sql")
        # Use [ \t] instead of \s so it does not match across newlines!
        refined_url_pattern = r'\b(https?://|github\.com|linkedin\.com)[ \t]*(?:[\w\.\-/@?&=]+[ \t]*)*'
        def clean_url_match(m):
            url = m.group()
            url = url.replace(" ", "")
            url = re.sub(r'/+', '/', url)
            url = url.replace("https:/", "https://").replace("http:/", "http://")
            return url

        text = re.sub(refined_url_pattern, clean_url_match, text)

        # Normalize spacing after labels like LinkedIn: https://... while keeping URL intact
        # Also standardize label casing
        def norm_label(m):
            lbl = m.group(1).lower().replace(" ", "")
            if lbl == "linkedin":
                lbl_clean = "LinkedIn"
            elif lbl == "github":
                lbl_clean = "GitHub"
            else:
                lbl_clean = m.group(1).capitalize()
            return f"{lbl_clean}: {m.group(2)}"

        text = re.sub(r'\b(LinkedIn|GitHub|Github|Git\s*Hub|Email|Phone|Address)[ \t]*:[ \t]*(https?://|\S+)', norm_label, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def fix_broken_words(text: str) -> str:
        fixes = {
            "PERSONALPROFILE": "PERSONAL PROFILE",
            "LANGUAGESPROFILE": "LANGUAGES\n\nPROFILE",
            "BachelorofTechnologyinInformationTechnology": "Bachelor of Technology in Information Technology",
            "SarvajanikCollegeofEngineeringandTechnology": "Sarvajanik College of Engineering and Technology",
            "WORKEXPERIENCE": "WORK EXPERIENCE",
            "TECHNICALSKILLS": "TECHNICAL SKILLS",
        }
        for old, new in fixes.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def fix_missing_spaces(text: str) -> str:
        # Pre-replace specific URL case transition targets so they are not split incorrectly by camelCase rules
        text = text.replace("RAGtext", "RAG-text")
        text = text.replace("RAGText", "RAG-Text")

        # 1. Insert space on lowercase-to-uppercase transitions (e.g., "BuiltanAI" -> "Builtan AI")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # 2. Insert space before an uppercase letter followed by lowercase (e.g., "MLMulti" -> "ML Multi")
        text = re.sub(r'([A-Za-z0-9])(?=[A-Z][a-z])', r'\1 ', text)

        # Prefix inline labels with newlines (e.g., namePhone: -> name\n\nPhone:)
        # Supports space-split words like "Git Hub:" or "Linked In:"
        text = re.sub(r'(?<!\n)\s*(Phone|Email|Git\s*Hub|Linked\s*In|Address):', r'\n\n\1:', text, flags=re.IGNORECASE)

        # 3. Fix common verb/article/API joins and split restorations
        fixes = {
            "Linked In": "LinkedIn",
            "Git Hub": "GitHub",
            "Fast API": "FastAPI",
            "APIAP Is": "API APIs",
            "Builtan": "Built an",
            "Builta": "Built a",
            "DevelopedFastAPI": "Developed FastAPI",
            "chat-basedplatform": "chat-based platform",
            "basedplatform": "based platform",
            "poweredassistant": "powered assistant",
            "AI-poweredassistant": "AI-powered assistant",
            "Prompt Engineerning": "Prompt Engineering",
            "Engineerning": "Engineering",
            "HuggingFace": "Hugging Face",
            "Gujrati": "Gujarati",
            "gujrati": "Gujarati",
        }
        for old, new in fixes.items():
            text = text.replace(old, new)

        # Regex replacements for others
        text = re.sub(r'\b([a-zA-Z]+)platform\b', r'\1 platform', text)
        text = re.sub(r'\b([a-zA-Z]+)assistant\b', r'\1 assistant', text)

        # Fix technology naming conventions: e.g. React. js -> React.js
        text = re.sub(r'\b(React|Node|Vue|Next|Nuxt|Nest)[ \t]*\.[ \t]*js\b', r'\1.js', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def fix_dates(text: str) -> str:
        # Separate month and year if they are merged, e.g. "Aug2022" -> "Aug 2022"
        months = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        text = re.sub(rf'\b({months})\s*(\d{{4}})\b', r'\1 \2', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def fix_digits(text: str) -> str:
        # Match digit sequences separated by spaces (e.g. "8 3 2 0", "395 0 0 4")
        def merge_spaced_digits(m):
            val = m.group()
            val_clean = val.replace(" ", "")
            if len(val_clean) >= 4 or "  " not in val:
                return val_clean
            return val
        return re.sub(r'\b\d+(?:\s+\d+)+\b', merge_spaced_digits, text)

    @staticmethod
    def fix_hyphens(text: str) -> str:
        # Normalize spacing around hyphens between letters/numbers (e.g., "AI - Powered" -> "AI-Powered")
        return re.sub(r'\b\s*-\s*\b', '-', text)

    @staticmethod
    def fix_section_headers(text: str) -> str:
        headers = {
            "CONTACT", "SUMMARY", "EXPERIENCE", "PROJECTS", "SKILLS", "EDUCATION", "LANGUAGES",
            "PERSONAL PROFILE", "WORK EXPERIENCE", "TECHNICAL SKILLS", "PROJECT", "PROFILE"
        }
        lines = []
        for line in text.splitlines():
            line_strip = line.strip()
            if line_strip.upper() in headers:
                lines.append(f"\n\n{line_strip.upper()}\n\n")
            else:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def fix_punctuation(text: str):
        text = re.sub(r"\s+:", ":", text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s+\.", ".", text)
        return text

    @staticmethod
    def remove_extra_spaces(text: str):
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def remove_blank_lines(text: str):
        # Keep empty lines intact, but strip surrounding spaces from each line
        return "\n".join(line.strip() for line in text.splitlines())
