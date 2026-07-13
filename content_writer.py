import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
from langchain.agents.middleware import ModelFallbackMiddleware, ToolRetryMiddleware, ToolCallLimitMiddleware
from dotenv import load_dotenv
load_dotenv()
from tools.generatecover import generate_cover
from tools.socialmediaimage import generate_social_image
from tools.websearch import web_search
from components.filesystem import StagingFilesystemBackend
from components.agentdisplay import AgentDisplay
from components.coustommiddleware import StatefulCacheMiddleware
import asyncio
import os
import sys
from database.redis import load_redis_history,save_redis_history,redis_client
import yaml
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from pathlib import Path
from deepagents import create_deep_agent, HarnessProfile, register_harness_profile
from langchain.agents.middleware import TodoListMiddleware

register_harness_profile(
    "groq",
    HarnessProfile(
        excluded_middleware=frozenset({TodoListMiddleware})
    )
)

EXAMPLE_DIR = Path(__file__).parent
console = Console()

DATABASE_SESSION_CACHE = {}

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
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=2,thread_limit=3,exit_behavior="end"),
            StatefulCacheMiddleware(session_id=session_id)
        ])

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