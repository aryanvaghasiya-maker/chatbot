import streamlit as st
import asyncio
import os
import uuid
import json
import redis
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import helper functions and registries from content_writer
from content_writer import (
    create_content_writer,
    _GLOBAL_BACKEND_REGISTRY,
    finalize_and_save_to_disk,
    load_redis_history,
    save_redis_history,
    redis_client,
    EXAMPLE_DIR
)

# Page configuration
st.set_page_config(
    page_title="Deep Agent Console",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern Custom CSS with dark mode, glowing borders, and clean typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Style input elements, chat input and textareas to ensure readable black text on white background */
    input, textarea, [data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
    }
    
    input:disabled, textarea:disabled {
        color: #666666 !important;
        background-color: #f5f5f5 !important;
        opacity: 0.8;
    }
    
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    
    .section-header {
        color: #a78bfa;
        font-size: 1.3rem;
        font-weight: 600;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    .file-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        transition: all 0.2s ease-in-out;
    }
    
    .file-card:hover {
        border-color: #8b5cf6;
        background: rgba(30, 41, 59, 0.6);
        cursor: pointer;
    }
    
    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    
    .badge-staged {
        background-color: #f59e0b;
        color: #1e1b4b;
    }
    
    .badge-saved {
        background-color: #10b981;
        color: #064e3b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "session_id" not in st.session_state:
    # Use a static prod session or generate one
    st.session_state.session_id = "session_content_writer_prod"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "global_static_thread_id"
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "session_prompt_tokens" not in st.session_state:
    st.session_state.session_prompt_tokens = 0
if "session_completion_tokens" not in st.session_state:
    st.session_state.session_completion_tokens = 0
if "session_total_tokens" not in st.session_state:
    st.session_state.session_total_tokens = 0

# Initialize agent or get existing, ensuring the backend is registered
if "agent" not in st.session_state or st.session_state.session_id not in _GLOBAL_BACKEND_REGISTRY:
    st.session_state.agent = create_content_writer(st.session_state.session_id)

# Fetch conversation history from Redis
if "messages" not in st.session_state:
    st.session_state.messages = load_redis_history(st.session_state.thread_id)

# Fetch staged virtual files
backend = _GLOBAL_BACKEND_REGISTRY.get(st.session_state.session_id)
virtual_files = getattr(backend, "virtual_files", {}) if backend else {}

# Main Title Layout
st.markdown("<div class='main-title'>🤖 Deep Agent Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Staging-based Content Writing Chatbot powered by Redis & DeepAgents</div>", unsafe_allow_html=True)

# Define Sidebar Controls
with st.sidebar:
    st.markdown("<div class='section-header'>Controls & Actions</div>", unsafe_allow_html=True)
    
    # Session Details
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    st.text_input("Thread ID (Redis Key)", value=st.session_state.thread_id, disabled=True)
    
    st.markdown("---")
    
    # Save/Finalize Action
    if st.button("🏁 Finalize & Commit to Disk", use_container_width=True, type="primary"):
        if virtual_files:
            success = finalize_and_save_to_disk(st.session_state.session_id)
            if success:
                st.success("Successfully saved all files to disk!")
                # Clear virtual cache from memory
                if st.session_state.session_id in _GLOBAL_BACKEND_REGISTRY:
                    del _GLOBAL_BACKEND_REGISTRY[st.session_state.session_id]
                # Clear Redis history
                redis_client.delete(f"chat_history:{st.session_state.thread_id}")
                st.session_state.messages = []
                st.session_state.selected_file = None
                st.session_state.session_prompt_tokens = 0
                st.session_state.session_completion_tokens = 0
                st.session_state.session_total_tokens = 0
                st.rerun()
            else:
                st.error("Failed to save files.")
        else:
            st.warning("No virtual drafts staged to save.")
            
    # Discard Action
    if st.button("❌ Discard Session & Exit", use_container_width=True):
        if st.session_state.session_id in _GLOBAL_BACKEND_REGISTRY:
            del _GLOBAL_BACKEND_REGISTRY[st.session_state.session_id]
        redis_client.delete(f"chat_history:{st.session_state.thread_id}")
        st.session_state.messages = []
        st.session_state.selected_file = None
        st.session_state.session_prompt_tokens = 0
        st.session_state.session_completion_tokens = 0
        st.session_state.session_total_tokens = 0
        st.success("Session discarded. Cache cleared.")
        st.rerun()

    st.markdown("<div class='section-header'>Token Usage</div>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Prompt", f"{st.session_state.session_prompt_tokens:,}")
    with col_t2:
        st.metric("Completion", f"{st.session_state.session_completion_tokens:,}")
    st.metric("Total Tokens Used", f"{st.session_state.session_total_tokens:,}")

# Layout split: Column 1 = Chat | Column 2 = Staged Document Preview
col_chat, col_preview = st.columns([1.1, 0.9])

with col_chat:
    st.markdown("<div class='section-header'>Interactive Chat Console</div>", unsafe_allow_html=True)
    
    # Render historical messages
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            # Try to print standard text or tool calls
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text_parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        text_parts.append(p)
                content = "\n".join(text_parts)
            
            if content.strip():
                with st.chat_message("assistant"):
                    st.markdown(content)
            
            # Show tool executions if present
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    with st.expander(f"⚙️ Running Tool: `{name}`", expanded=False):
                        st.json(args)

    # Chat Input
    if user_prompt := st.chat_input("Ask Deep Agent to write or modify content..."):
        # Display user query
        with st.chat_message("user"):
            st.write(user_prompt)
        
        # Append to message history
        st.session_state.messages.append(HumanMessage(content=user_prompt))
        
        # Execute agent asynchronously and capture output
        async def run_agent():
            status_placeholder = st.empty()
            with status_placeholder.status("Deep Agent is processing workflow...", expanded=True) as status:
                last_chunk = None
                async for chunk in st.session_state.agent.astream(
                    {"messages": st.session_state.messages},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                    stream_mode="values",
                ):
                    last_chunk = chunk
                    # Render tool execution details in real-time
                    if "messages" in chunk and len(chunk["messages"]) > 0:
                        last_msg = chunk["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                name = tc.get("name", "unknown")
                                args = tc.get("args", {})
                                if name == "task":
                                    desc = args.get("description", "researching...")
                                    st.write(f"🔍 **Researching**: {desc}")
                                elif name in ("generate_cover", "generate_social_image"):
                                    st.write(f"🎨 **Generating cover image** using Gemini...")
                                elif name in ("write_file", "save_markdown", "save_file"):
                                    path = args.get("path") or args.get("filepath") or args.get("file_path") or "file"
                                    st.write(f"📝 **Staging draft to cache**: `{path}`")
                                elif name == "web_search":
                                    query = args.get("query", "")
                                    st.write(f"🌐 **Searching web for**: *{query}*")
                
                status.update(label="Workflow process completed!", state="complete")
            
            # Retrieve final messages
            if last_chunk and "messages" in last_chunk:
                st.session_state.messages = last_chunk["messages"]
                # Save to Redis
                save_redis_history(st.session_state.thread_id, st.session_state.messages)
                
                # Extract and update token usage
                try:
                    if last_chunk and "usage" in last_chunk:
                        usage = last_chunk["usage"]
                    elif last_chunk and "messages" in last_chunk and len(last_chunk["messages"]) > 0:
                        last_msg = last_chunk["messages"][-1]
                        usage = getattr(last_msg, "response_metadata", {}).get("token_usage", None) or getattr(last_msg, "usage_metadata", None)
                    else:
                        usage = None

                    if usage:
                        p_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
                        c_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
                        t_tokens = usage.get("total_tokens", 0) or (p_tokens + c_tokens)
                        
                        st.session_state.session_prompt_tokens += p_tokens
                        st.session_state.session_completion_tokens += c_tokens
                        st.session_state.session_total_tokens += t_tokens
                except Exception:
                    pass
                st.rerun()

        # Run async agent thread
        asyncio.run(run_agent())

with col_preview:
    st.markdown("<div class='section-header'>Staged File Explorer (RAM Cache)</div>", unsafe_allow_html=True)
    
    if not virtual_files:
        st.info("No documents are currently staged in the volatile memory cache. Ask the agent to write a post to see them here!")
    else:
        # Display list of virtual files
        st.write("Click on a staged file to preview its code and content:")
        
        selected = None
        for rel_path in sorted(virtual_files.keys()):
            path_str = str(rel_path)
            card_col1, card_col2 = st.columns([3, 1])
            with card_col1:
                if st.button(f"📄 {path_str}", key=f"btn_{path_str}", use_container_width=True):
                    st.session_state.selected_file = path_str
            with card_col2:
                st.markdown("<span class='status-badge badge-staged'>Staged</span>", unsafe_allow_html=True)
        
        # Display selected file contents
        if st.session_state.selected_file and st.session_state.selected_file in virtual_files:
            file_data = virtual_files[st.session_state.selected_file]
            
            # Parse text content from the virtual backend structure
            if isinstance(file_data, dict) and "content" in file_data:
                text_content = file_data["content"]
            elif isinstance(file_data, str):
                text_content = file_data
            else:
                text_content = str(file_data)
                
            st.markdown(f"### Previewing: `{st.session_state.selected_file}`")
            
            # Tabs for styled preview vs raw source
            tab_raw, tab_render = st.tabs(["Raw Code", "Rendered Markdown"])
            with tab_raw:
                st.code(text_content, language="markdown")
            with tab_render:
                st.markdown(text_content)
