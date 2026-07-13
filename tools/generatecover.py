from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post."""
    try:
        from google import genai
        client = genai.Client()
        response = client.models.generate_content(model="gemini-2.5-flash-image", contents=[prompt])
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = PROJECT_ROOT / "blogs" / slug / "hero.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"
        return "No image generated"
    except Exception as e:
        return f"Image Generation Skipped: {e}"
