import re

def reconstruct_text_from_json(data: dict) -> str:
    lines = []
    
    # 1. Contact Information
    contact = data.get("contact", {})
    if contact:
        contact_line = []
        if contact.get("name"):
            contact_line.append(f"Name: {contact['name']}")
        if contact.get("email"):
            contact_line.append(f"Email: {contact['email']}")
        if contact.get("phone"):
            contact_line.append(f"Phone: {contact['phone']}")
        if contact.get("github"):
            contact_line.append(f"GitHub: {contact['github']}")
        if contact.get("linkedin"):
            contact_line.append(f"LinkedIn: {contact['linkedin']}")
        if contact.get("address"):
            contact_line.append(f"Address: {contact['address']}")

        if contact_line:
            lines.append("CONTACT")
            lines.append(" | ".join(contact_line))
            lines.append("")

    # 2. Summary
    if data.get("summary"):
        lines.append("SUMMARY")
        lines.append(data["summary"])
        lines.append("")

    # 3. Skills
    skills = data.get("skills", {})
    if skills:
        lines.append("SKILLS")
        if isinstance(skills, dict):
            for k, v in skills.items():
                if isinstance(v, list) and v:
                    lines.append(f"{k.capitalize()}: {', '.join(v)}")
                elif isinstance(v, str) and v:
                    lines.append(f"{k.capitalize()}: {v}")
        elif isinstance(skills, list):
            lines.append(", ".join(skills))
        else:
            lines.append(str(skills))
        lines.append("")

    # 4. Experience
    experience = data.get("experience", [])
    if experience:
        lines.append("EXPERIENCE")
        if isinstance(experience, list):
            for item in experience:
                if isinstance(item, dict):
                    title = item.get("experience", item.get("company", ""))
                    lines.append(title)
                    for bullet in item.get("bullets", []):
                        lines.append(f"• {bullet}")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(experience))
        lines.append("")

    # 5. Projects
    projects = data.get("projects", [])
    if projects:
        lines.append("PROJECTS")
        if isinstance(projects, list):
            for item in projects:
                if isinstance(item, dict):
                    title = item.get("project", item.get("name", ""))
                    lines.append(title)
                    for bullet in item.get("bullets", []):
                        lines.append(f"• {bullet}")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(projects))
        lines.append("")

    # 6. Education
    education = data.get("education", [])
    if education:
        lines.append("EDUCATION")
        if isinstance(education, list):
            for item in education:
                lines.append(str(item))
        else:
            lines.append(str(education))
        lines.append("")

    # 7. Languages
    languages = data.get("languages", "")
    if languages:
        lines.append("LANGUAGES")
        lines.append(languages)
        lines.append("")

    return "\n".join(lines).strip()
