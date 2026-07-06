from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class StructuredResume:
    contact: dict
    summary: str
    experience: list
    projects: list
    skills: dict
    education: list
    languages: str = ""

@dataclass
class ParsedResume:
    filename: str
    text: str
    ocr_used: bool
    mime_type: str
    contact: dict = field(default_factory=dict)
    summary: str = ""
    experience: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    skills: dict = field(default_factory=dict)
    education: list = field(default_factory=list)
    languages: str = ""
    raw_text_length: int = 0
