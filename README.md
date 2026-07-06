# Resume Optimization Service

A FastAPI-backed service that orchestrates a resume analysis and optimization pipeline using a LangGraph state graph.

## Features
- Resume upload parsing for `.pdf` and `.docx`
- Skill extraction and resume optimization
- ATS score evaluation and iterative gap remediation
- Final asset generation including LaTeX resume output and company suggestions

## Running Locally
1. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. Set the `OPENAI_API_KEY` environment variable.
3. Start the app:
   ```bash
   uvicorn app:app --reload
   ```

## API Endpoints
- `POST /upload-resume` — upload a resume file and extract text
- `POST /optimize` — optimize a resume with a job description

## Notes
- The current implementation depends on `langgraph`, `langchain-openai`, and Redis for checkpointing.
- The service builds a directed state graph in `agent/schema/llm.py` and uses prompt templates from `agent/config/prompts.py`.
