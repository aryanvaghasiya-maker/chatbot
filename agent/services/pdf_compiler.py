"""
pdf_compiler.py
Compiles a LaTeX string into a PDF using pdflatex.
Runs in a temporary directory and returns the raw PDF bytes.
"""
import asyncio
import shutil
import tempfile
from pathlib import Path


async def compile_latex_to_pdf(latex: str) -> bytes:
    """
    Compile a LaTeX source string to PDF bytes using pdflatex.

    Runs pdflatex twice (standard practice for resolving internal references).
    Raises RuntimeError on compilation failure with the pdflatex log excerpt.
    """
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found. Install TeX Live: "
            "sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-latex-extra"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "resume.tex"
        pdf_path = Path(tmpdir) / "resume.pdf"
        log_path = Path(tmpdir) / "resume.log"

        tex_path.write_text(latex, encoding="utf-8")

        cmd = [
            pdflatex,
            "-interaction=nonstopmode",   # don't pause on errors
            "-halt-on-error",              # exit non-zero on first error
            "-output-directory", tmpdir,
            str(tex_path),
        ]

        # Run twice — first pass builds the document, second resolves references
        for pass_num in (1, 2):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Extract the useful part of the log (last 60 lines)
                log_excerpt = ""
                if log_path.exists():
                    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    log_excerpt = "\n".join(log_lines[-60:])
                raise RuntimeError(
                    f"pdflatex failed on pass {pass_num} (exit {proc.returncode}).\n"
                    f"Log tail:\n{log_excerpt or stderr.decode(errors='replace')}"
                )

        if not pdf_path.exists():
            raise RuntimeError("pdflatex ran successfully but no PDF was produced.")

        return pdf_path.read_bytes()
