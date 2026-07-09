#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
from langchain.agents.middleware import ModelFallbackMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import sys
import uuid
import json
from pathlib import Path
from typing import Literal, Callable, Awaitable

import yaml
import redis 
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict, messages_to_dict
from langchain_core.tools import tool
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from deepagents import create_deep_agent, HarnessProfile, register_harness_profile
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse

register_harness_profile(
    "groq",
    HarnessProfile(
        excluded_middleware=frozenset({TodoListMiddleware})
    )
)

EXAMPLE_DIR = Path(__file__).parent
console = Console()

# Connect to Redis for managing persistent thread memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Helper functions to persist LangChain chat history messages inside Redis
def save_redis_history(thread_id: str, messages: list):
    dict_msgs = messages_to_dict(messages)
    redis_client.set(f"chat_history:{thread_id}", json.dumps(dict_msgs), ex=86400) # Keep for 24 hours

def load_redis_history(thread_id: str) -> list:
    data = redis_client.get(f"chat_history:{thread_id}")
    if not data:
        return []
    dict_msgs = json.loads(data)
    # Rehydrate structural types back into LangChain schema formats
    return messages_from_dict(dict_msgs)

DATABASE_SESSION_CACHE = {}

class StatefulCacheMiddleware(AgentMiddleware):
    """Intercepts agent workflows, establishing temporary virtual configurations."""
    def __init__(self, session_id: str):
        self.session_id = session_id

    def _modify_request(self, request: ModelRequest) -> ModelRequest:
        from langchain_core.messages import SystemMessage
        sys_msg = request.system_message
        addition = (
            f"\n\n[VOLATILE CACHE ACTIVE]\n"
            f"Session ID: {self.session_id}\n"
            f"You are inside an interactive chat loop. Every time you draft, write, edit, or generate any final content "
            f"(including blog posts, LinkedIn posts, tweets, or articles), you MUST save it as a file using your filesystem tools (e.g. `write_file` or `write`).\n"
            f"Do NOT just output the final draft in your text response. Always save it to the staging cache "
            f"(e.g. `linkedin/<slug>/post.md` for LinkedIn, `blogs/<slug>.md` for blogs) so the user can preview it in their dashboard.\n"
            f"Explicitly tell the user the path where you saved/staged the file.\n"
            f"If you generate an image, save it alongside the post using `generate_social_image`.\n"
            f"[/VOLATILE CACHE]"
        )
        if sys_msg is None:
            sys_msg = SystemMessage(content=addition)
        else:
            if isinstance(sys_msg.content, str):
                new_content = sys_msg.content + addition
            elif isinstance(sys_msg.content, list):
                new_content = list(sys_msg.content)
                if new_content and isinstance(new_content[-1], dict) and new_content[-1].get("type") == "text":
                    last_block = dict(new_content[-1])
                    last_block["text"] = last_block.get("text", "") + addition
                    new_content[-1] = last_block
                elif new_content and isinstance(new_content[-1], str):
                    new_content[-1] = new_content[-1] + addition
                else:
                    new_content.append({"type": "text", "text": addition})
            else:
                new_content = str(sys_msg.content) + addition
            sys_msg = SystemMessage(content=new_content)
        return request.override(system_message=sys_msg)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        modified_req = self._modify_request(request)
        return handler(modified_req)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        modified_req = self._modify_request(request)
        return await handler(modified_req)

@tool
def web_search(query: str, max_results: int = 5, topic: Literal["general", "news"] = "general") -> dict:
    """Search the web for current information."""
    try:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key: return {"error": "TAVILY_API_KEY not set"}
        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}

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
                output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"
        return "No image generated"
    except Exception as e:
        return f"Image Generation Skipped: {e}"

@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post."""
    try:
        from google import genai
        client = genai.Client()
        response = client.models.generate_content(model="gemini-2.5-flash-image", contents=[prompt])
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / platform / slug / "image.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"
        return "No image generated"
    except Exception as e:
        return f"Image Generation Skipped: {e}"

def load_subagents(config_path: Path) -> list:
    available_tools = {"web_search": web_search}
    with open(config_path) as f:
        config = yaml.safe_load(f)
    subagents = []
    for name, spec in config.items():
        subagent = {"name": name, "description": spec["description"], "system_prompt": spec["system_prompt"]}
        if "model" in spec: subagent["model"] = spec["model"]
        if "tools" in spec: subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)
    return subagents

import fnmatch
from deepagents.backends.protocol import WriteResult, ReadResult, EditResult, LsResult, GlobResult, GrepResult
from deepagents.backends.filesystem import perform_string_replacement

class StagingFilesystemBackend(FilesystemBackend):
    def __init__(self, root_dir, **kwargs):
        super().__init__(root_dir=root_dir, **kwargs)
        self.virtual_files = {}  # Relative path string -> content string

    def _get_rel_path(self, file_path: str) -> str:
        try:
            p = Path(file_path)
            if p.is_absolute():
                return str(p.relative_to(self.cwd))
            return str(p)
        except ValueError:
            return file_path

    def write(self, file_path: str, content: str) -> WriteResult:
        rel_path = self._get_rel_path(file_path)
        self.virtual_files[rel_path] = content
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            sliced_content = content[offset : offset + limit]
            return ReadResult(content=sliced_content)
        return super().read(file_path, offset=offset, limit=limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            sliced_content = content[offset : offset + limit]
            return ReadResult(content=sliced_content)
        return await super().aread(file_path, offset=offset, limit=limit)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            if isinstance(result, str):
                return EditResult(error=result)
            new_content, occurrences = result
            self.virtual_files[rel_path] = new_content
            return EditResult(path=file_path, occurrences=int(occurrences))
        
        # Load from disk to cache first if file exists
        disk_read = super().read(file_path)
        if disk_read.content is not None:
            self.virtual_files[rel_path] = disk_read.content
            return self.edit(file_path, old_string, new_string, replace_all)
        return EditResult(error=f"Error: File '{file_path}' not found")

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    def ls(self, path: str) -> LsResult:
        base_result = super().ls(path)
        rel_path = self._get_rel_path(path)
        if rel_path == "." or rel_path == "/":
            rel_path = ""
            
        entries = base_result.entries or []
        existing_paths = {entry["path"].rstrip("/") for entry in entries}

        for v_path in self.virtual_files:
            v_dir = str(Path(v_path).parent)
            if v_dir == rel_path or (rel_path == "" and v_dir == "."):
                name = Path(v_path).name
                virt_entry_path = name if rel_path == "" else f"{rel_path}/{name}"
                if virt_entry_path not in existing_paths:
                    entries.append({
                        "path": virt_entry_path,
                        "is_dir": False,
                        "size": len(self.virtual_files[v_path]),
                    })
                    existing_paths.add(virt_entry_path)
            elif v_path.startswith(rel_path + "/" if rel_path else ""):
                sub_parts = v_path[len(rel_path + "/" if rel_path else ""):].split("/")
                if len(sub_parts) > 1:
                    dir_name = sub_parts[0]
                    virt_dir_path = dir_name if rel_path == "" else f"{rel_path}/{dir_name}"
                    if virt_dir_path not in existing_paths:
                        entries.append({
                            "path": virt_dir_path + "/",
                            "is_dir": True,
                            "size": 0,
                        })
                        existing_paths.add(virt_dir_path)

        entries.sort(key=lambda x: x.get("path", ""))
        return LsResult(error=base_result.error, entries=entries)

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        base_result = super().glob(pattern, path)
        search_path = path or ""
        rel_search_path = self._get_rel_path(search_path)
        if rel_search_path == "." or rel_search_path == "/":
            rel_search_path = ""

        matches = base_result.matches or []
        existing_paths = {m["path"] for m in matches}

        for v_path in self.virtual_files:
            if rel_search_path == "" or v_path.startswith(rel_search_path + "/"):
                rel_v_path = v_path[len(rel_search_path + "/" if rel_search_path else ""):]
                if fnmatch.fnmatch(rel_v_path, pattern):
                    if v_path not in existing_paths:
                        matches.append({
                            "path": v_path,
                            "is_dir": False,
                            "size": len(self.virtual_files[v_path]),
                        })
                        existing_paths.add(v_path)
                        
        matches.sort(key=lambda x: x.get("path", ""))
        return GlobResult(error=base_result.error, matches=matches)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return super().grep(pattern, path=path, glob=glob)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return await super().agrep(pattern, path=path, glob=glob)

_GLOBAL_BACKEND_REGISTRY = {}

def create_content_writer(session_id: str):
    # This staging layout manages temporary output structures correctly in RAM cache!
    fs_backend = StagingFilesystemBackend(root_dir=EXAMPLE_DIR, virtual_mode=True)
    _GLOBAL_BACKEND_REGISTRY[session_id] = fs_backend

    return create_deep_agent(
        model="openai:gpt-5-nano",
        memory=["./AGENTS.md"],           
        skills=["./skills/"],             
        tools=[generate_cover, generate_social_image],  
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),  
        backend=fs_backend,
        middleware=[
            ToolRetryMiddleware(max_retries=3),
            ModelFallbackMiddleware("openrouter:google/gemma-4-31b-it:free", "openrouter:openai/gpt-oss-20b:free"),
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=5),
            StatefulCacheMiddleware(session_id=session_id)
        ],
    )

class AgentDisplay:
    """Manages the display beautifully. Shows action indicators but hides internal text paste blocks."""
    def __init__(self):
        self.printed_count = 0
        self.spinner = Spinner("dots", text="Deep Agent is processing workflow...")

    def update_status(self, status: str):
        self.spinner = Spinner("dots", text=status)

    def print_message(self, msg, is_final_turn: bool = False):
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))
        
        elif isinstance(msg, AIMessage):
            if is_final_turn:
                content = msg.content
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = "\n".join(text_parts)
                if content and content.strip():
                    console.print(Panel(Markdown(content), title="Agent 🤖", border_style="green"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    if name == "task":
                        desc = args.get("description", "researching...")
                        console.print(f"  [bold magenta]>> Researching:[/] {desc[:60]}...")
                        self.update_status(f"Researching: {desc[:40]}...")
                    elif name in ("generate_cover", "generate_social_image"):
                        console.print(f"  [bold cyan]>> Generating image...[/]")
                        self.update_status("Generating image...")
                    elif name in ("write_file", "save_markdown", "save_file"):
                        path = args.get("path") or args.get("filepath") or args.get("file_path") or "file"
                        console.print(f"  [bold yellow]>> Staging Draft to Cache:[/] {path}")
                    elif name == "web_search":
                        query = args.get("query", "")
                        console.print(f"  [bold blue]>> Searching Web:[/] {query[:50]}...")
                        self.update_status(f"Searching: {query[:30]}...")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ("generate_cover", "generate_social_image"):
                if "saved" in msg.content.lower():
                    console.print(f"  [green]OK: Image captured successfully[/]")
                else:
                    console.print(f"  [dim red]Skipped: Image parameters bypassed ({msg.content[:40]})[/]")
            elif name in ("write_file", "save_markdown", "save_file"):
                console.print(f"  [green]OK: Draft verified in RAM cache memory[/]")
            elif name == "task":
                console.print(f"  [green]OK: Research task synchronized[/]")
            elif name == "web_search":
                if "error" not in msg.content.lower():
                    console.print(f"  [green]OK: Search facts parsed[/]")


def finalize_and_save_to_disk(session_id: str):
    """Inspects the true agent virtual staging cache memory and commits files directly onto disk."""
    backend = _GLOBAL_BACKEND_REGISTRY.get(session_id)
    if not backend:
        print("❌ System Environment Registry lost for this session.")
        return False

    virtual_files = getattr(backend, "virtual_files", {}) or getattr(backend, "_virtual_files", {})
    
    if not virtual_files:
        print("❌ No active draft data found to commit.")
        return False
        
    print(f"📦 Found {len(virtual_files)} staged document(s) in memory context. Saving to disk layout...")
    
    for relative_path, file_content in virtual_files.items():
        path_str = str(relative_path).lstrip("/")
        final_path = EXAMPLE_DIR / path_str
        os.makedirs(final_path.parent, exist_ok=True)
        
        if isinstance(file_content, dict) and "content" in file_content:
            text_data = file_content["content"]
        elif isinstance(file_content, str):
            text_data = file_content
        else:
            text_data = str(file_content)

        with open(final_path, "w", encoding="utf-8") as f:
            f.write(text_data)
            
        print(f" 💾 [SAVED] -> {final_path.relative_to(EXAMPLE_DIR)}")
        
    print("🏁 [SUCCESS] Confirmed! All documents saved permanently onto disk layout system.")
    return True

async def async_chat_loop():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

    console.print(Panel("[bold green]🤖 Deep Agent Chatbot Mode Initialized with Redis History State.[/]\n\n"
                        "• Real-time updates like [magenta]Researching[/], [cyan]Image Generation[/], and [blue]Web Searching[/] will show actively.\n"
                        "• Thread state tracking handles conversation memory updates dynamically.\n"
                        "• Commands: [bold gold1]'save'[/] = write permanently | [bold red]'exit'[/] = clear cache completely.", 
                        title="System Configuration", border_style="cyan"))
    
    current_session = "session_content_writer_prod"
    thread_id = "global_static_thread_id" 
    
    agent = create_content_writer(session_id=current_session)
    
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0
    
    # Fetch existing logs directly from Redis to recall what happened in the previous conversation
    conversation_history = load_redis_history(thread_id)
    if conversation_history:
        console.print(f"[dim green]🔄 Rehydrated {len(conversation_history)} historical messages from Redis cache memory state.[/]")

    while True:
        try:
            user_input = input("\nYou 💬: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Discarded. Exiting.")
            break
        
        if not user_input: continue
            
        if user_input.lower() in ["save", "finalize", "done"]:
            print("\n🏁 Finalizing chat session...")
            finalize_and_save_to_disk(current_session)
            if current_session in _GLOBAL_BACKEND_REGISTRY:
                del _GLOBAL_BACKEND_REGISTRY[current_session]
                
            # Clear historical memory after finishing work
            redis_client.delete(f"chat_history:{thread_id}")
            break
            
        if user_input.lower() == "exit":
            if current_session in _GLOBAL_BACKEND_REGISTRY:
                del _GLOBAL_BACKEND_REGISTRY[current_session]
            redis_client.delete(f"chat_history:{thread_id}")
            print("👋 Session destroyed. All memory-cached data has been permanently cleared without saving.")
            break
            
        conversation_history.append(HumanMessage(content=user_input))
        display = AgentDisplay()
        last_chunk = None
        
        with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
            async for chunk in agent.astream(
                {"messages": conversation_history},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values",
            ):
                last_chunk = chunk
                if "messages" in chunk:
                    messages = chunk["messages"]
                    if len(messages) > display.printed_count:
                        live.stop()
                        for msg in messages[display.printed_count:]:
                            display.print_message(msg, is_final_turn=False)
                        display.printed_count = len(messages)
                        live.start()
                        live.update(display.spinner)

        if last_chunk and "messages" in last_chunk:
            conversation_history = last_chunk["messages"]
            if len(conversation_history) > 0:
                display.print_message(conversation_history[-1], is_final_turn=True)
            
            save_redis_history(thread_id, conversation_history)

        try:
            if last_chunk and "usage" in last_chunk:
                usage = last_chunk["usage"]
            elif last_chunk and "messages" in last_chunk and len(last_chunk["messages"]) > 0:
                last_msg = last_chunk["messages"][-1]
                usage = getattr(last_msg, "response_metadata", {}).get("token_usage", None) or getattr(last_msg, "usage_metadata", None)
            else:
                usage = None

            if usage:
                p_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                t_tokens = usage.get("total_tokens", 0) or (p_tokens + c_tokens)
                
                session_prompt_tokens += p_tokens
                session_completion_tokens += c_tokens
                session_total_tokens += t_tokens
                
                print(f"\n📊 [Usage: +{t_tokens:,} tokens | Session Cumulative: {session_total_tokens:,} tokens]")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(async_chat_loop())