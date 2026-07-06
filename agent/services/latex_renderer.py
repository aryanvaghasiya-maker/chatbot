"""
latex_renderer.py
Compiles structured OptimizedResumeOutput schemas into high-quality LaTeX code
using a fixed Jinja2 template (Jake's Resume style).
"""

import re
from jinja2 import Environment
from agent.schema.schema import OptimizedResumeOutput


# ---------------------------------------------------------------------------
# LaTeX special-character escaping
# ---------------------------------------------------------------------------
_LATEX_ESCAPE_MAP = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]

def _escape(text: str) -> str:
    """Escape all LaTeX-special characters in a plain-text string."""
    if not text:
        return ""
    result = text.replace("\\", r"\textbackslash{}")
    for char, replacement in _LATEX_ESCAPE_MAP[1:]:
        result = result.replace(char, replacement)
    return result


def _format_latex_link(link: str, default_domain: str) -> tuple[str, str]:
    """Returns a tuple of (href_url, display_text) for hyperref, ensuring full URLs are preserved."""
    if not link:
        return "", ""
    
    link_raw = link.strip()
    link_clean = link_raw.lower()
    
    # 1. Determine href_url
    if link_clean.startswith("http://") or link_clean.startswith("https://"):
        href_url = link_raw
    else:
        href_url = f"https://{link_raw}"
        
    # 2. Determine display_text
    display_text = link_raw
    # Remove http:// or https:// prefix for cleaner display text
    display_text = re.sub(r'^https?://', '', display_text)
    
    # If the domain is not in the text, format it using the default domain
    if default_domain and default_domain not in display_text.lower():
        if default_domain == "linkedin.com":
            display_text = f"linkedin.com/in/{display_text}"
            href_url = f"https://linkedin.com/in/{link_raw}"
        elif default_domain == "github.com":
            display_text = f"github.com/{display_text}"
            href_url = f"https://github.com/{link_raw}"
            
    return href_url, display_text


# ---------------------------------------------------------------------------
# Jinja2 environment — uses (( )) delimiters so they don't clash with LaTeX
# ---------------------------------------------------------------------------
_jinja_env = Environment(
    variable_start_string="((",
    variable_end_string="))",
    block_start_string="(%",
    block_end_string="%)",
    comment_start_string="(#",
    comment_end_string="#)",
    autoescape=False,
    keep_trailing_newline=True,
)
_jinja_env.filters["e"] = _escape


# ---------------------------------------------------------------------------
# Jake's Resume–style LaTeX template
# ---------------------------------------------------------------------------
_RESUME_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.6in]{geometry}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{tabularx}

% Clear formatting for ATS readability
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=blue,
    pdfauthor={(( contact.name ))},
    pdftitle={Resume},
    pdfcreator={Resume AI Agent}
}

% Style headings using standard, recognizable text
\titleformat{\section}{\large\bfseries\uppercase}{}{0em}{}[\titlerule]
\titlespacing*{\section}{0pt}{10pt}{4pt}

% Configure bullet lists for clean parsing
\setlist[itemize]{leftmargin=0.15in, label={$\bullet$}, itemsep=1.5pt, parsep=0pt, topsep=0pt}

\pagestyle{empty}

\begin{document}

% CONTACT INFORMATION
\begin{center}
    {\LARGE \textbf{(( contact.name ))}} \\ \vspace{4pt}
    (% if contact.title or core_competencies %)\textbf{\large (% if contact.title %)(( contact.title ))(% endif %)(% if contact.title and core_competencies %) --- (% endif %)(% if core_competencies %)(( core_competencies | map('e') | join(' --- ') ))(% endif %)} \\ \vspace{6pt}(% endif %)
    \small
    (( contact.location ))
    (% if contact.location and contact.email %) \ $\bullet$ \ (% endif %)\href{mailto:(( contact.email ))}{(( contact.email ))}
    (% if contact.phone %) \ $\bullet$ \ (( contact.phone ))(% endif %)
    (% if contact.linkedin_url %) \ $\bullet$ \ \href{(( contact.linkedin_url ))}{(( contact.linkedin_display ))}(% endif %)
    (% if contact.github_url %) \ $\bullet$ \ \href{(( contact.github_url ))}{(( contact.github_display ))}(% endif %)
    (% if contact.portfolio_url %) \ $\bullet$ \ \href{(( contact.portfolio_url ))}{(( contact.portfolio_display ))}(% endif %)
\end{center}

\vspace{6pt}

% PROFESSIONAL SUMMARY
\section{Professional Summary}
(( summary ))

% SKILLS
(% if skills.languages or skills.backend or skills.ai_llm or skills.databases or skills.cloud or skills.devops or skills.messaging or skills.monitoring or skills.testing or skills.architecture or skills.other %)
\section{Core Skills}
\begin{itemize}
    (% if skills.languages %)
      \item \textbf{Languages:} (( skills.languages | map('e') | join(', ') ))
    (% endif %)
    (% if skills.backend %)
      \item \textbf{Backend Frameworks \& Libraries:} (( skills.backend | map('e') | join(', ') ))
    (% endif %)
    (% if skills.ai_llm %)
      \item \textbf{AI \& LLM Technologies:} (( skills.ai_llm | map('e') | join(', ') ))
    (% endif %)
    (% if skills.databases %)
      \item \textbf{Databases \& Caching:} (( skills.databases | map('e') | join(', ') ))
    (% endif %)
    (% if skills.cloud %)
      \item \textbf{Cloud Platforms \& Services:} (( skills.cloud | map('e') | join(', ') ))
    (% endif %)
    (% if skills.devops %)
      \item \textbf{DevOps, Containers \& CI/CD:} (( skills.devops | map('e') | join(', ') ))
    (% endif %)
    (% if skills.messaging %)
      \item \textbf{Messaging \& Event Streaming:} (( skills.messaging | map('e') | join(', ') ))
    (% endif %)
    (% if skills.monitoring %)
      \item \textbf{Monitoring, Logging \& Observability:} (( skills.monitoring | map('e') | join(', ') ))
    (% endif %)
    (% if skills.testing %)
      \item \textbf{Testing \& QA Frameworks:} (( skills.testing | map('e') | join(', ') ))
    (% endif %)
    (% if skills.architecture %)
      \item \textbf{System Architecture \& Design:} (( skills.architecture | map('e') | join(', ') ))
    (% endif %)
    (% if skills.other %)
      \item \textbf{Other Technologies:} (( skills.other | map('e') | join(', ') ))
    (% endif %)
\end{itemize}
(% endif %)

% ACHIEVEMENTS
(% if achievements %)
\section{Achievements}
\begin{itemize}
    (% for ach in achievements %)
      \item (( ach | e ))
    (% endfor %)
\end{itemize}
\vspace{6pt}
(% endif %)

% EXPERIENCE
(% if experience %)
\section{Professional Experience}
(% for job in experience %)
\noindent \textbf{(( job.title | e ))} \hfill (( job.start_date | e )) -- (( job.end_date | e )) \\
(( job.company | e )) \hfill (( job.location | e ))
(% if job.bullets %)
\begin{itemize}
    (% for bullet in job.bullets %)
      \item (( bullet | e ))
    (% endfor %)
\end{itemize}
(% endif %)
\vspace{6pt}
(% endfor %)
(% endif %)

% PROJECTS
(% if projects %)
\section{Projects}
(% for proj in projects %)
\noindent \textbf{(( proj.name | e ))} 
(% if proj.github_url or proj.live_demo_url %)
  \hfill (% if proj.github_url %)\href{(( proj.github_url ))}{GitHub}(% endif %)(% if proj.github_url and proj.live_demo_url %) $|$ (% endif %)(% if proj.live_demo_url %)\href{(( proj.live_demo_url ))}{Live Demo}(% endif %)
(% endif %) \\
\textit{\small Tech Stack: (( proj.tech_stack | map('e') | join(', ') ))}
(% set ns = namespace(has_bullets=false) %)
(% for bullet in proj.description.split('\n') %)
  (% if bullet.strip() %)
    (% set ns.has_bullets = true %)
  (% endif %)
(% endfor %)
(% if ns.has_bullets %)
\begin{itemize}
    (% for bullet in proj.description.split('\n') %)
      (% if bullet.strip() %)
        \item (( bullet.strip().lstrip('-*• ').strip() | e ))
      (% endif %)
    (% endfor %)
\end{itemize}
(% endif %)
\vspace{6pt}
(% endfor %)
(% endif %)

% EDUCATION
(% if education %)
\section{Education}
(% for edu in education %)
\noindent \textbf{(( edu.degree | e ))} \hfill (( edu.graduation_date | e )) \\
(( edu.institution | e )) (% if edu.gpa %)\hfill GPA: (( edu.gpa | e ))(% endif %)
\vspace{6pt}
(% endfor %)
(% endif %)

% CERTIFICATIONS
(% if certifications %)
\section{Certifications}
(% for cert in certifications %)
\noindent \textbf{(( cert.name | e ))} (% if cert.issuer %) -- (( cert.issuer | e ))(% endif %) \hfill (( cert.date | e )) \par
\vspace{3pt}
(% endfor %)
(% endif %)

\end{document}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_latex(optimized: OptimizedResumeOutput) -> str:
    """
    Render an OptimizedResumeOutput into a complete, compilable LaTeX string.
    All user-supplied text is escaped; the document structure comes from the
    fixed template.
    """
    # Pre-process links for contact info
    linkedin_url, linkedin_display = _format_latex_link(optimized.contact.linkedin, "linkedin.com")
    github_url, github_display = _format_latex_link(optimized.contact.github, "github.com")
    portfolio_url, portfolio_display = _format_latex_link(optimized.contact.portfolio, "")
    
    # Pre-escape all contact fields
    contact_data = {
        "name": _escape(optimized.contact.name),
        "email": _escape(optimized.contact.email),
        "phone": _escape(optimized.contact.phone),
        "location": _escape(optimized.contact.location),
        "title": _escape(optimized.contact.title) if getattr(optimized.contact, "title", None) else "",
        "linkedin_url": linkedin_url,
        "linkedin_display": _escape(linkedin_display),
        "github_url": github_url,
        "github_display": _escape(github_display),
        "portfolio_url": portfolio_url,
        "portfolio_display": _escape(portfolio_display),
    }

    # Pre-process projects to include URL formatted links
    projects_data = []
    for proj in (optimized.projects or []):
        gh_url = ""
        if getattr(proj, "github_url", None):
            gh_url, _ = _format_latex_link(proj.github_url, "github.com")
        demo_url = ""
        if getattr(proj, "live_demo_url", None):
            demo_url, _ = _format_latex_link(proj.live_demo_url, "")
        
        projects_data.append({
            "name": proj.name,
            "tech_stack": proj.tech_stack,
            "description": proj.description,
            "github_url": gh_url,
            "live_demo_url": demo_url,
        })

    achievements_data = getattr(optimized, "achievements", []) or []
    core_competencies_data = getattr(optimized, "core_competencies", []) or []
    certifications_data = getattr(optimized, "certifications", []) or []

    template = _jinja_env.from_string(_RESUME_TEMPLATE)
    latex = template.render(
        contact=contact_data,
        summary=_escape(optimized.summary),
        skills=optimized.skills,
        experience=optimized.experience,
        education=optimized.education,
        projects=projects_data,
        achievements=achievements_data,
        core_competencies=core_competencies_data,
        certifications=certifications_data,
    )
    return latex
