import json
from typing import Any
from agent.states.states import ResumeState
from agent.services.resume_parser import ResumeCleaner, ResumeSectionParser, StructuredResumeParser, reconstruct_text_from_json

class resume_parser_node:
    def __init__(self, max_loops: int = 3, target_score: int = 85):
        self.max_loops = max_loops
        self.target_score = target_score

    async def parse_resume_node(self, state: ResumeState) -> dict[str, Any]:
        raw_resume = state.get("raw_resume", "").strip()
        parsed_dict = None

        if raw_resume.startswith("{") and raw_resume.endswith("}"):
            try:
                parsed_dict = json.loads(raw_resume)
            except Exception:
                parsed_dict = None

        if parsed_dict:
            reconstructed_text = reconstruct_text_from_json(parsed_dict)
        else:
            cleaned_text = ResumeCleaner.clean(raw_resume)
            sections = ResumeSectionParser.parse(cleaned_text)
            parsed_dict = StructuredResumeParser.extract(sections)
            reconstructed_text = cleaned_text

        parsed = {
            "parsed_resume": parsed_dict,
            "extracted_text": reconstructed_text,
            "raw_resume": reconstructed_text,
        }
        return parsed

