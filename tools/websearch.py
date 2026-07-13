from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
import os
from tavily import TavilyClient

@tool
def web_search(query: str, max_results: int = 5, topic: str = "general") -> str:
    """Search the web for information using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."
    try:
        client = TavilyClient(api_key=api_key)
        # Limit max_results between 1 and 5 to prevent excessive usage
        max_results = max(1, min(max_results, 5))
        
        response = client.search(query=query, max_results=max_results, topic=topic)
        results = response.get("results", [])
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            url = res.get("url", "No URL")
            content = res.get("content", "")
            formatted_results.append(f"{i}. [{title}]({url})\n   {content}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Web Search Failed: {e}"
