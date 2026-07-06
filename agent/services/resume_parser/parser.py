from pathlib import Path
from io import BytesIO

from docx import Document
from pypdf import PdfReader

from .cleaner import ResumeCleaner
from .section_parser import ResumeSectionParser
from .structured_parser import StructuredResumeParser
from .cache import ResumeCache
from .models import ParsedResume

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:
    pytesseract = None
    convert_from_bytes = None

class ResumeParser:

    @classmethod
    def parse(cls, filename: str, file_bytes: bytes) -> ParsedResume:
        cache_key = ResumeCache.key(file_bytes)

        if ResumeCache.exists(cache_key):
            cached_data = ResumeCache.load(cache_key)
            return ParsedResume(**cached_data)

        suffix = Path(filename).suffix.lower()
        ocr_used = False
        mime_type = "text/plain"

        if suffix == ".pdf":
            text, ocr_used = cls._pdf(file_bytes)
            mime_type = "application/pdf"
        elif suffix == ".docx":
            text = cls._docx(file_bytes)
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            raise ValueError(f"Unsupported resume format: {suffix}")

        cleaned = ResumeCleaner.clean(text)
        sections = ResumeSectionParser.parse(cleaned)
        structured = StructuredResumeParser.extract(sections)

        parsed = ParsedResume(
            filename=filename,
            text=cleaned,
            ocr_used=ocr_used,
            mime_type=mime_type,
            contact=structured["contact"],
            summary=structured["summary"],
            experience=structured["experience"],
            projects=structured["projects"],
            skills=structured["skills"],
            education=structured["education"],
            languages=structured["languages"],
            raw_text_length=len(cleaned)
        )

        # Save to cache
        cache_data = {
            "filename": parsed.filename,
            "text": parsed.text,
            "ocr_used": parsed.ocr_used,
            "mime_type": parsed.mime_type,
            "contact": parsed.contact,
            "summary": parsed.summary,
            "experience": parsed.experience,
            "projects": parsed.projects,
            "skills": parsed.skills,
            "education": parsed.education,
            "languages": parsed.languages,
            "raw_text_length": parsed.raw_text_length
        }
        ResumeCache.save(cache_key, cache_data)

        return parsed

    @classmethod
    def _pdf(cls, file_bytes: bytes) -> tuple[str, bool]:
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        ocr_used = False
        if not text.strip():
            text = cls._ocr_pdf(file_bytes)
            ocr_used = bool(text)
        return text, ocr_used

    @staticmethod
    def _ocr_pdf(file_bytes: bytes) -> str:
        if convert_from_bytes is None or pytesseract is None:
            return ""

        try:
            images = convert_from_bytes(file_bytes)
        except Exception:
            return ""

        text_parts = []
        for image in images:
            try:
                extracted_text = pytesseract.image_to_string(image)
            except Exception:
                continue
            if extracted_text:
                text_parts.append(extracted_text)

        return "\n".join(text_parts).strip()

    @staticmethod
    def _docx(file_bytes: bytes) -> str:
        doc = Document(BytesIO(file_bytes))
        paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs)
