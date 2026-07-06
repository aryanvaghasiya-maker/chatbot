import re

SECTION_HEADERS = [
    "SUMMARY",
    "PROFILE",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "EXPERIENCES",
    "PROJECTS",
    "PROJECT",
    "SKILLS",
    "TECHNICAL SKILLS",
    "EDUCATION",
    "CERTIFICATIONS",
    "LANGUAGES",
    "CONTACT",
]

class ResumeSectionParser:

    @staticmethod
    def parse(text: str):
        sections = {}
        current = "HEADER"
        sections[current] = []

        headers_map = {
            "SUMMARY": "SUMMARY",
            "PROFILE": "SUMMARY",
            "EXPERIENCE": "EXPERIENCE",
            "WORK EXPERIENCE": "EXPERIENCE",
            "EXPERIENCES": "EXPERIENCE",
            "PROJECTS": "PROJECTS",
            "PROJECT": "PROJECTS",
            "SKILLS": "SKILLS",
            "TECHNICAL SKILLS": "SKILLS",
            "EDUCATION": "EDUCATION",
            "CERTIFICATIONS": "CERTIFICATIONS",
            "LANGUAGES": "LANGUAGES",
            "CONTACT": "CONTACT",
        }

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Strip common bullet and formatting symbols for header matching
            upper = re.sub(r'^[-\*•#\s]+|[:\s]+$', '', stripped.upper()).strip()

            if upper in headers_map:
                current = headers_map[upper]
                if current not in sections:
                    sections[current] = []
                continue

            sections[current].append(stripped)

        return {
            key: "\n".join(value).strip()
            for key, value in sections.items()
        }
