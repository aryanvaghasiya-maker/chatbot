import pytest
from pathlib import Path
from agent.services.resume_parser import (
    ResumeCleaner,
    ResumeSectionParser,
    StructuredResumeParser,
    ResumeCache,
    reconstruct_text_from_json,
)

def test_character_spacing_normalizations():
    # Spacing and number normalizations
    assert ResumeCleaner.clean("8 3 2 0 4 7 7 6 0 5") == "8320477605"
    assert ResumeCleaner.clean("3 9 5 0 0 4") == "395004"
    assert ResumeCleaner.clean("2 0 2 6") == "2026"
    assert ResumeCleaner.clean("7 0 B") == "70B"

def test_email_cleanup():
    assert ResumeCleaner.clean("darshanjain 2 2 0 2 @ gmail. com") == "darshanjain2202@gmail.com"

def test_url_cleanup():
    assert ResumeCleaner.clean("github.com / Darshan 2 7 3 / SQL - RAGtext 2 sql") == "github.com/Darshan273/SQL-RAG-text2sql"
    assert "LinkedIn: https://www.linkedin.com/in/darshan-jain-401791273/" in ResumeCleaner.clean("LinkedIn:https://www.linkedin.com/in/darshan-jain-401791273/")
    assert ResumeCleaner.clean("github.com / Darshan 2 7 3 / medibot - RAG") == "github.com/Darshan273/medibot-RAG"

def test_remove_broken_words():
    assert ResumeCleaner.clean("PERSONALPROFILE") == "PERSONAL PROFILE"
    assert "LANGUAGES\n\nPROFILE" in ResumeCleaner.clean("LANGUAGESPROFILE")
    assert ResumeCleaner.clean("BachelorofTechnologyinInformationTechnology") == "Bachelor of Technology in Information Technology"
    assert ResumeCleaner.clean("SarvajanikCollegeofEngineeringandTechnology") == "Sarvajanik College of Engineering and Technology"

def test_fix_missing_spaces():
    assert ResumeCleaner.clean("BuiltanAI-poweredassistant") == "Built an AI-powered assistant"
    assert ResumeCleaner.clean("DevelopedFastAPIAPIs") == "Developed FastAPI APIs"
    assert ResumeCleaner.clean("chat-basedplatform") == "chat-based platform"
    assert ResumeCleaner.clean("HuggingFace") == "Hugging Face"
    assert ResumeCleaner.clean("React. js") == "React.js"
    assert "ML Multi-Model AI Assistant\n\nGitHub:" in ResumeCleaner.clean("MLMulti-ModelAIAssistantGitHub:")

def test_normalize_hyphens():
    assert ResumeCleaner.clean("AI - Powered") == "AI-Powered"
    assert ResumeCleaner.clean("Text - to - SQL") == "Text-to-SQL"
    assert ResumeCleaner.clean("Retrieval - Augmented Generation") == "Retrieval-Augmented Generation"

def test_detect_bullet_points():
    lines = ["Built a chatbot", "Developed APIs", "• Already a bullet"]
    processed = StructuredResumeParser.detect_bullet_points(lines)
    assert processed[0] == "• Built a chatbot"
    assert processed[1] == "• Developed APIs"
    assert processed[2] == "• Already a bullet"

def test_project_and_experience_boundaries():
    proj_text = """
    SQL RAG Text-to-SQL
    Built an AI-powered assistant
    Developed FastAPI APIs
    AI Powered Medical Chatbot
    Created chat-based platform
    """
    projects = StructuredResumeParser.parse_projects(proj_text)
    assert len(projects) == 2
    assert projects[0]["project"] == "SQL RAG Text-to-SQL"
    assert len(projects[0]["bullets"]) == 2
    assert projects[0]["bullets"][0] == "Built an AI-powered assistant"
    assert projects[1]["project"] == "AI Powered Medical Chatbot"

def test_categorize_skills():
    skills_text = "Python, Go, FastAPI, PostgreSQL, Pinecone, Docker, OpenAI"
    skills = StructuredResumeParser.parse_skills(skills_text)
    assert "Python" in skills["languages"]
    assert "Go" in skills["languages"]
    assert "FastAPI" in skills["frameworks"]
    assert "PostgreSQL" in skills["databases"]
    assert "Pinecone" in skills["vector_db"]
    assert "Docker" in skills["tools"]
    assert "OpenAI" in skills["ai"]

def test_extract_contact():
    contact_block = """
    Darshan Jain
    Phone: 8 3 2 0 4 7 7 6 0 5
    Email: darshanjain 2 2 0 2 @ gmail. com
    github.com / Darshan 2 7 3
    linkedin.com/in/darshan-jain-401791273
    Surat, 395004
    """
    contact = StructuredResumeParser.contact(contact_block)
    assert contact["name"] == "Darshan Jain"
    assert contact["phone"] == "8320477605"
    assert contact["email"] == "darshanjain2202@gmail.com"
    assert contact["github"] == "https://github.com/Darshan273"
    assert contact["linkedin"] == "https://linkedin.com/in/darshan-jain-401791273"
    assert contact["address"] == "Surat, 395004"

def test_abnormal_spacing_check():
    normal = "This is a normal text block."
    abnormal = "D A R S H A N  J A I N  8 3 2 0 4"
    assert not ResumeCleaner.detect_abnormal_spacing(normal)
    assert ResumeCleaner.detect_abnormal_spacing(abnormal)

def test_restore_section_order():
    data = {
        "contact": {"name": "Darshan Jain", "email": "darshanjain2202@gmail.com"},
        "summary": "Short summary",
        "experience": [{"experience": "Software Engineer at TechHive", "bullets": ["Developed APIs"]}],
        "projects": [{"project": "SQL RAG", "bullets": ["Built SQL generator"]}],
        "skills": {"languages": ["Python"], "frameworks": ["FastAPI"]},
        "education": ["B.Tech IT"],
        "languages": "English",
    }
    reconstructed = reconstruct_text_from_json(data)
    
    # Verify section headings presence
    assert "CONTACT" in reconstructed
    assert "SUMMARY" in reconstructed
    assert "EXPERIENCE" in reconstructed
    assert "PROJECTS" in reconstructed
    assert "SKILLS" in reconstructed
    assert "EDUCATION" in reconstructed
    assert "LANGUAGES" in reconstructed
    
    # Check section order
    contact_idx = reconstructed.index("CONTACT")
    summary_idx = reconstructed.index("SUMMARY")
    skills_idx = reconstructed.index("SKILLS")
    exp_idx = reconstructed.index("EXPERIENCE")
    proj_idx = reconstructed.index("PROJECTS")
    edu_idx = reconstructed.index("EDUCATION")
    lang_idx = reconstructed.index("LANGUAGES")
    
    assert contact_idx < summary_idx < skills_idx < exp_idx < proj_idx < edu_idx < lang_idx

def test_dates_normalization():
    assert ResumeCleaner.clean("March 2 0 2 6") == "March 2026"
    assert ResumeCleaner.clean("Aug2022") == "Aug 2022"

def test_runon_lines_splitting():
    text = "Built an AI-powered assistant leveraging RAG. Integrated Spam Detection. Developed FastAPI APIs."
    lines = StructuredResumeParser.split_runon_lines([text])
    assert len(lines) == 3
    assert lines[0] == "Built an AI-powered assistant leveraging RAG."
    assert lines[1] == "Integrated Spam Detection."
    assert lines[2] == "Developed FastAPI APIs."

def test_typos_and_casing():
    assert ResumeCleaner.clean("Gujrati") == "Gujarati"
    assert "GitHub: https://github.com" in ResumeCleaner.clean("GitHub:https://github.com")

def test_full_user_resume_clean():
    raw_text = """ARSHAN JAINPhone: 8 3 2 0 4 7 7 6 0 5
Email: darshanjain 2 2 0 2 @ gmail. com
Address: Surat, 3 9 5 0 0 4
CONTACT
PERSONALPROFILE
AI / ML ENGINEER
PROJECT
B. Tech graduate in Information Technology with hands - on experience in AI / ML, Generative AI, RAG
systems, and Backend Development. Currently working as an AI / ML Intern at Enthusia Softtech Pvt. Ltd.,
gaining practical experience in LangChain, LangGraph, FastAPI, and AI application development.
Passionate about building intelligent systems and solving real - world problem.
Experience
Enthusia Softtech Pvt. Ltd. March 2 0 2 6 – Present
AI / ML Intern Surat, India
Working on Generative AI, LangChain, LangGraph, and Retrieval - Augmented Generation ( RAG ) systems.
Developing backend APIs using FastAPI for AI - powered applications.
Building and testing AI agents and document - based question - answering systems.
Continuously learning and implementing industry best practices in AI application development.
SQL RAG Text - to - SQL System GitHub: github.com / Darshan 2 7 3 / SQL - RAGtext 2 sql
Built an intelligent Text - to - SQL application using RAG and Large Language Models ( LLMs ) to translate
natural language into SQL queries.
Developed a multi - agent workflow using LangGraph for query generation, validation, execution, and answer
synthesis.
Integrated Groq Llama - 3 - 7 0 B to achieve low - latency SQL generation and response delivery.
Implemented automated error detection and query correction to enhance system reliability.
Added SQL safety checks and read - only controls to ensure secure database interactions.
AI - Powered Medical Chatbot GitHub: github.com / Darshan 2 7 3 / medibot - RAG
Built a Retrieval - Augmented Generation ( RAG ) chatbot for answering medical queries from PDF documents.
Implemented PDF ingestion, semantic chunking, and embedding generation using LangChain and
HuggingFace models.
Utilized FAISS vector search with top - k retrieval for efficient context extraction.
Integrated Groq LLM with prompt engineering to generate accurate, context - aware responses.
Developed a Streamlit - based conversational interface and optimized retrieval performance for improved
answer quality.
SKILLS
Languages: Python, HTML, CSS, JavaScript, SQL
AI / ML / LLM: Machine Learning, Deep Learning, Generative AI, Large Language Models ( LLMs ),
Embeddings, Retrieval - Augmented Generation ( RAG ), Prompt Engineerning
Frameworks: FastAPI, LangChain, LangGraph, React. js, Streamlit
Libraries: NumPy, Pandas, Scikit - learn, Matplotlib, Seaborn
Vector Databases: FAISS, ChromaDB, Qdrant
Databases: PostgreSQL, MongoDB,
Tools & Platforms: Git, GitHub, VS Code, Antigravity Postman, Hugging Face, Docker, Groq API,
LangSmith
Gujrati
Hindi
English
Github: https://github.com/Darshan273
LinkedIn:https://www.linkedin.com/in/darshan-jain-401791273/
LANGUAGESPROFILE
BachelorofTechnologyinInformationTechnology
SarvajanikCollegeofEngineeringandTechnology,SarvajanikUniversity,Surat
Aug2022–June2026|CGPA:8.51
·12thGrade–83%
·10thGrade-77%
EDUCATION
MLMulti-ModelAIAssistantGitHub:github.com/Darshan273/ML-Multi-model-tool-calling
BuiltanAI-poweredassistantleveragingGroqLLMtool-callingtorouteuserqueriestospecializedML
models.
IntegratedSpamDetection,CustomerChurnPrediction,andBrainTumorMRIClassificationintoaunified
chat-basedplatform.
DevelopedFastAPIAPIsandautomatedpredictionworkflowsforseamlessmodelinferenceanduser
interaction.
Designedanend-to-endAIarchitecturecombiningLLMorchestration,MLpipelines,andreal-time
conversationalresponses."""

    cleaned = ResumeCleaner.clean(raw_text)
    
    # Assertions for spacing normalizations
    assert "8320477605" in cleaned
    assert "darshanjain2202@gmail.com" in cleaned
    assert "395004" in cleaned
    assert "March 2026" in cleaned
    assert "Aug 2022" in cleaned
    assert "PERSONAL PROFILE" in cleaned
    assert "Bachelor of Technology in Information Technology" in cleaned
    assert "Sarvajanik College of Engineering and Technology" in cleaned
    
    # Assertions for hyphenation
    assert "Text-to-SQL" in cleaned
    assert "AI-Powered" in cleaned
    assert "real-world" in cleaned
    assert "question-answering" in cleaned
    assert "top-k" in cleaned
    assert "context-aware" in cleaned
    
    # Assertions for technology names
    assert "Hugging Face" in cleaned
    assert "React.js" in cleaned
    assert "Prompt Engineering" in cleaned
    
    # Assertions for language spelling
    assert "Gujarati" in cleaned
    
    # Assertions for split restorations
    assert "GitHub: https://github.com" in cleaned
    assert "LinkedIn: https://www.linkedin.com" in cleaned
