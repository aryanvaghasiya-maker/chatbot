from .parser import ResumeParser
from .models import ParsedResume, StructuredResume
from .cleaner import ResumeCleaner
from .section_parser import ResumeSectionParser
from .structured_parser import StructuredResumeParser
from .cache import ResumeCache
from .utils import reconstruct_text_from_json

def parse_resume_file(filename: str, file_bytes: bytes) -> ParsedResume:
    return ResumeParser.parse(filename, file_bytes)
