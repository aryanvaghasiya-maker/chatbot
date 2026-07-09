# 📚 Deep Agent — Complete Project Guide

> **Who is this for?** If you are new to this project or returning after a break, this document explains the full current architecture, every component, and how to run the system — step by step, in plain English.

---

## 🗂️ Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Key Concepts You Need to Know](#key-concepts-you-need-to-know)
3. [Project Structure — Every File Explained](#project-structure--every-file-explained)
4. [Architecture Overview — How It All Fits Together](#architecture-overview--how-it-all-fits-together)
5. [The Streamlit Dashboard (app.py)](#the-streamlit-dashboard-apppy)
6. [The Volatile Staging System](#the-volatile-staging-system)
7. [Redis — Persistent Chat History](#redis--persistent-chat-history)
8. [The Middleware System](#the-middleware-system)
9. [The Skill System](#the-skill-system)
10. [The Subagent System](#the-subagent-system)
11. [The Fallback System (Reliability Layer)](#the-fallback-system-reliability-layer)
12. [API Keys — What They Do](#api-keys--what-they-do)
13. [How a Request Flows — Step by Step](#how-a-request-flows--step-by-step)
14. [How to Run the Project](#how-to-run-the-project)
15. [Output Files — What Gets Created](#output-files--what-gets-created)
16. [Common Errors & Fixes](#common-errors--fixes)
17. [Extending the Project](#extending-the-project)

---

## What Is This Project?

This is an **AI Content Writer Agent** with a full **web-based Streamlit dashboard**, built on the `deepagents` framework.

**In simple words:** You open a browser, chat with the AI agent, and it:

1. 🔍 **Searches the web** for current information automatically
2. 🧠 **Researches the topic** using a helper subagent
3. ✍️  **Writes the content** following your brand's style guide
4. 💾 **Stages the draft in RAM** (not saved to disk yet — you preview it first!)
5. 🏁 **Commits to disk only when you click "Finalize"** — no accidental writes

The agent is smart enough to **recover on its own** if one AI model fails (rate limit, etc.) by automatically trying other fallback models.

---

## Key Concepts You Need to Know

### What Is an AI Agent?

A regular AI chatbot just answers your question. An **agent** can:
- **Use tools** (search the web, write files, call APIs)
- **Plan multi-step tasks** (research → write → stage → commit)
- **Delegate to other agents** (ask a subagent to do research)
- **Retry on failure** automatically

### What Is `deepagents`?

`deepagents` is the Python library that powers this agent. It handles:
- Building the agent loop (think → act → observe → repeat)
- Loading memory (`AGENTS.md`), skills, and subagents from files
- Managing the conversation history and tool calls
- Orchestrating all the middleware layers

### What Is "Volatile Staging"?

Instead of writing files directly to disk, all agent file writes go into an **in-memory virtual cache** (`StagingFilesystemBackend`). This means:
- You can review and iterate on drafts before they are saved
- Nothing touches the filesystem until you explicitly click **"Finalize & Commit to Disk"**
- Discarding a session clears everything — no leftover files

### What Is Redis?

Redis is a fast in-memory database used here to **persist the chat conversation history**. Even if you refresh the browser or restart the server, your previous conversation is reloaded from Redis automatically.

---

## Project Structure — Every File Explained

```
deep agent/
│
├── app.py                ← STREAMLIT WEB DASHBOARD — main UI entry point
├── content_writer.py     ← AGENT CORE — agent definition, middleware, backend
├── AGENTS.md             ← Brand voice & style guide (loaded into every prompt)
├── subagents.yaml        ← Defines the "researcher" helper agent
├── pyproject.toml        ← Python package config & dependencies (uses uv)
├── .env                  ← SECRET API keys (never share this!)
│
├── skills/               ← Skill definitions (teach the agent how to do tasks)
│   ├── blog-post/
│   │   └── SKILL.md      ← Instructions for writing blog posts
│   └── social-media/
│       └── SKILL.md      ← Instructions for LinkedIn/Twitter posts
│
├── blogs/                ← OUTPUT: Blog posts saved here after "Finalize"
│   └── <slug>/
│       ├── post.md       ← The written blog post
│       └── hero.png      ← Auto-generated cover image
│
├── research/             ← OUTPUT: Research notes from the researcher subagent
│   └── <topic>.md
│
└── linkedin/             ← OUTPUT: Social media posts saved after "Finalize"
    └── <topic>/
        └── post.md
```

> ⚠️ Files in `blogs/`, `research/`, and `linkedin/` are **only created after you click "Finalize & Commit to Disk"**. During the session they exist only in RAM.

---

## Architecture Overview — How It All Fits Together

```
Browser (http://localhost:8501)
         │
         ▼
   ┌─────────────┐
   │   app.py    │  ← Streamlit dashboard (UI, chat, file explorer)
   └──────┬──────┘
          │  Creates & calls
          ▼
   ┌──────────────────────┐
   │  content_writer.py   │  ← Agent core
   │  create_content_writer()
   │  StagingFilesystemBackend  ← virtual RAM file system
   │  StatefulCacheMiddleware   ← injects staging instructions
   │  ModelFallbackMiddleware   ← auto model switching
   │  ToolRetryMiddleware       ← auto retry on tool errors
   │  ToolCallLimitMiddleware   ← web_search capped at 5 calls
   └──────┬───────────────┘
          │  Reads history from / writes history to
          ▼
   ┌─────────────┐
   │    Redis    │  ← Stores chat history (24h TTL)
   └─────────────┘
          │
          ▼
   ┌──────────────────────────────┐
   │  _GLOBAL_BACKEND_REGISTRY    │  ← Maps session_id → StagingFilesystemBackend
   │  virtual_files = {}          │     (in-process Python dict, not persisted)
   └──────────────────────────────┘
```

---

## The Streamlit Dashboard (`app.py`)

The dashboard at `http://localhost:8501` has three main areas:

### Sidebar — Controls & Actions

| Control | What It Does |
|---------|-------------|
| **Session ID** | Identifies the current agent session (read-only) |
| **Thread ID** | Redis key used for chat history (read-only) |
| **🏁 Finalize & Commit to Disk** | Writes all staged virtual files to the real filesystem |
| **❌ Discard Session & Exit** | Clears RAM cache and Redis history without saving |

### Left Column — Interactive Chat Console

- Shows all conversation messages (user + agent) with proper formatting
- Displays **tool use expanders** (e.g., "⚙️ Running Tool: `write_file`") for transparency
- Real-time **status spinner** with live tool-call updates while the agent works
- Standard `st.chat_input` for sending new messages

### Right Column — Staged File Explorer (RAM Cache)

- Lists all files currently in the virtual RAM cache (staged but not saved)
- Click any file button to preview its content
- Two tabs: **Raw Code** (syntax-highlighted) and **Rendered Markdown**
- Shows badge `Staged` for files pending commit

---

## The Volatile Staging System

### How It Works

The `StagingFilesystemBackend` class (in `content_writer.py`) subclasses the `deepagents` `FilesystemBackend`. It **overrides all write operations** to redirect them into an in-memory dictionary instead of the real filesystem:

```python
class StagingFilesystemBackend(FilesystemBackend):
    def __init__(self, root_dir, **kwargs):
        super().__init__(root_dir=root_dir, **kwargs)
        self.virtual_files = {}  # path → content (in RAM only)

    def write(self, file_path, content) -> WriteResult:
        rel_path = self._get_rel_path(file_path)
        self.virtual_files[rel_path] = content   # ← goes to RAM, not disk
        return WriteResult(path=file_path)
```

It also overrides `read`, `edit`, `ls`, `glob`, and their async variants so that virtual files are visible to the agent's own read-back operations.

### The Global Registry

```python
_GLOBAL_BACKEND_REGISTRY = {}  # session_id → StagingFilesystemBackend
```

This global dict lets `app.py` access the same backend object the agent is using, so the UI can display the virtual files in the Staged File Explorer.

### Committing to Disk

When you click **"Finalize & Commit to Disk"**, `finalize_and_save_to_disk(session_id)` is called:

```python
def finalize_and_save_to_disk(session_id: str) -> bool:
    backend = _GLOBAL_BACKEND_REGISTRY.get(session_id)
    for relative_path, content in backend.virtual_files.items():
        path_str = str(relative_path).lstrip("/")
        final_path = EXAMPLE_DIR / path_str
        os.makedirs(final_path.parent, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(content)
```

After this, the registry entry is deleted and Redis history is cleared.

---

## Redis — Persistent Chat History

Redis stores the full conversation history so it survives browser refreshes and server restarts.

### Setup

Redis must be running locally on port 6379 before starting the dashboard:

```bash
# Linux / macOS
redis-server

# Or check if it is running
redis-cli ping  # should return: PONG
```

### How It Works

```python
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Save history (24h TTL)
def save_redis_history(thread_id, messages):
    redis_client.set(f"chat_history:{thread_id}", json.dumps(messages_to_dict(messages)), ex=86400)

# Load history on startup
def load_redis_history(thread_id) -> list:
    data = redis_client.get(f"chat_history:{thread_id}")
    return messages_from_dict(json.loads(data)) if data else []
```

The `thread_id` used is `"global_static_thread_id"` — a fixed key shared across all browser sessions.

---

## The Middleware System

**Middleware** = Code that wraps around every model call, like layers of an onion. Each layer can inspect or modify the request before it reaches the LLM, and the response on the way back.

Current middleware stack (outermost to innermost):

```
Request →
    ToolRetryMiddleware          (auto-retry failed tool calls, max 3)
    ModelFallbackMiddleware      (switch model on rate limit / error)
    ToolCallLimitMiddleware      (cap web_search at 5 calls/turn)
    StatefulCacheMiddleware      (inject staging instructions into system prompt)
    TodoListMiddleware           (tracks multi-step tasks)
    SkillsMiddleware             (injects skill SKILL.md instructions)
    PatchToolCallsMiddleware     (normalizes tool call formats)
    ToolCallLimitMiddleware[ws]  (inner web_search guard)
    MemoryMiddleware             (injects AGENTS.md)
→ LLM Model (openai:gpt-5-nano)
```

### StatefulCacheMiddleware (Custom)

**File:** `content_writer.py` — `class StatefulCacheMiddleware`

**What it does:** On every model call, it appends a `[VOLATILE CACHE ACTIVE]` block to the system message. This block explicitly instructs the agent:
- Always save content using `write_file` tools, not just printing it in chat
- Use the correct staging paths (`linkedin/<slug>/post.md`, `blogs/<slug>.md`)
- Tell the user the path where the file was staged

```python
def _modify_request(self, request: ModelRequest) -> ModelRequest:
    addition = "[VOLATILE CACHE ACTIVE]\n... must save files via tools ..."
    sys_msg = SystemMessage(content=request.system_message.content + addition)
    return request.override(system_message=sys_msg)
```

### ModelFallbackMiddleware

If the primary model fails, it tries fallback models in order:

```
Primary:    openai:gpt-5-nano
Fallback 1: openrouter:google/gemma-4-31b-it:free
Fallback 2: openrouter:openai/gpt-oss-20b:free
```

### ToolRetryMiddleware

Automatically retries any failed tool call up to 3 times before giving up.

### ToolCallLimitMiddleware

Caps the `web_search` tool at **5 calls per agent turn** to prevent runaway searches.

---

## The Skill System

Skills are **instruction files** that teach the agent how to do specific tasks. `SkillsMiddleware` automatically detects which skill applies and injects the `SKILL.md` instructions into the system prompt.

### `skills/blog-post/SKILL.md`

Tells the agent:
- **Always research first** using the `researcher` subagent
- **Structure**: Hook → Context → Main Content → Practical Application → CTA
- **Always generate a cover image** after writing via `generate_cover`
- **Save to** `blogs/<slug>/post.md` and `blogs/<slug>/hero.png`
- SEO rules: Keep title under 60 chars, include keywords naturally

### `skills/social-media/SKILL.md`

Tells the agent:
- Write short, punchy LinkedIn/Twitter posts
- Include 3–5 hashtags
- Save to `linkedin/<topic>/post.md`

---

## The Subagent System

**Subagents** are specialized mini-agents that the main agent can delegate tasks to.

Defined in `subagents.yaml`:

```yaml
researcher:
  description: >
    ALWAYS use this first to research any topic before writing content.
    Searches the web for current information, statistics, and sources.
  system_prompt: |
    You are a research assistant. You have access to web_search and write_file tools.
    1. Use web_search to find information on the topic
    2. Make 2-3 targeted searches with specific queries
    3. Save findings to the file path specified in your task
  tools:
    - web_search
```

**How the main agent calls it:**

```python
task(
    subagent_type="researcher",
    description="Research AI agents in 2025. Save to research/ai-agents.md"
)
```

The researcher runs independently, does its searches, saves the file to the virtual cache, and returns a summary to the main agent.

> ⚠️ **Groq models cannot be used as the primary model** — they reject subagent tool calls because `researcher` is not a standard declared tool. Use OpenAI or OpenRouter models only.

---

## The Fallback System (Reliability Layer)

| Level | Model | Notes |
|-------|-------|-------|
| **Primary** | `openai:gpt-5-nano` | Fast, capable, supports subagents |
| **Fallback 1** | `openrouter:google/gemma-4-31b-it:free` | Free, no daily request limit |
| **Fallback 2** | `openrouter:openai/gpt-oss-20b:free` | Second free option |

The `ModelFallbackMiddleware` catches these error types:

| Error | Meaning | Action |
|-------|---------|--------|
| `rate_limit_exceeded` | Too many calls | Try next model |
| `free-models-per-day` | Daily OpenRouter free quota hit | Try next model |
| `model_not_found` | Wrong model name | Skip (permanent) |
| `tool_use_failed` | Model can't handle subagent tools (Groq) | Skip (permanent) |
| `model_decommissioned` | Model retired | Skip (permanent) |

---

## API Keys — What They Do

All keys live in `.env` and are **never committed to git**.

| Key | Service | Used For |
|-----|---------|----------|
| `OPENAI_API_KEY` | OpenAI | Primary model (`gpt-5-nano`) and subagent model |
| `OPENROUTER_API_KEY` | OpenRouter | Access to 100+ free/paid fallback models |
| `GROQ_API_KEY` | Groq | Fast inference for Llama/Qwen models (not used as primary) |
| `TAVILY_API_KEY` | Tavily | Web search API for real-time information |
| `GOOGLE_API_KEY` | Google | Cover image & social image generation via Gemini Imagen |
| `REDIS_URL` | Redis | Optional override (defaults to `redis://localhost:6379`) |

---

## How a Request Flows — Step by Step

Here is what happens when you type `"make linkedin post for ai agent"` in the dashboard:

**Step 1 — User Submits Message**
```
app.py receives input from st.chat_input
→ appends HumanMessage to session_state.messages
→ calls agent.astream({messages: ...}, stream_mode="values")
```

**Step 2 — StatefulCacheMiddleware Fires**
```
On first model call:
→ Appends [VOLATILE CACHE ACTIVE] block to system prompt
→ Instructs agent: "You MUST save files using write_file tools"
```

**Step 3 — SkillsMiddleware Fires**
```
Detects "linkedin post" intent
→ Injects skills/social-media/SKILL.md instructions into system prompt
```

**Step 4 — Model Thinks**
```
gpt-5-nano reads:
  - AGENTS.md (brand voice + staging guidelines)
  - social-media/SKILL.md (how to write a LinkedIn post)
  - [VOLATILE CACHE ACTIVE] block (must save via write_file)
  - User message

Decides: "I should call write_file to stage the post"
Returns: tool_call { write_file, path: "linkedin/ai-agents/post.md", content: "..." }
```

**Step 5 — StagingFilesystemBackend Intercepts**
```
write_file tool is called
→ StagingFilesystemBackend.write() fires
→ virtual_files["linkedin/ai-agents/post.md"] = "..." (stored in RAM)
→ Nothing written to disk yet
```

**Step 6 — UI Updates**
```
app.py re-renders after agent completes
→ Staged File Explorer reads _GLOBAL_BACKEND_REGISTRY[session_id].virtual_files
→ Shows "📄 linkedin/ai-agents/post.md" button with [Staged] badge
```

**Step 7 — User Clicks "Finalize & Commit to Disk"**
```
finalize_and_save_to_disk(session_id) runs
→ Iterates virtual_files dict
→ Writes each file to real disk under EXAMPLE_DIR
→ Clears backend registry + Redis history
→ UI reloads empty
```

---

## How to Run the Project

### Prerequisites

```bash
# 1. Install uv (Python package manager)
pip install uv

# 2. Install all dependencies
uv sync

# 3. Create .env file with your API keys
cp .env.example .env   # or create manually
# Add: OPENAI_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY

# 4. Start Redis (must be running before the dashboard)
redis-server
# Verify: redis-cli ping  → should return PONG
```

### Running the Dashboard (Recommended)

```bash
uv run streamlit run app.py
# Open http://localhost:8501 in your browser
```

### Running the CLI (Legacy Mode)

```bash
uv run python content_writer.py
# Starts an interactive terminal chat loop
# Type 'save' to commit files to disk
# Type 'exit' to discard and quit
```

### What You'll See in the Dashboard

```
🤖 Deep Agent Dashboard
Staging-based Content Writing Chatbot powered by Redis & DeepAgents

[ Chat Console ]                    [ Staged File Explorer (RAM Cache) ]

You: make linkedin post for ai agent

  ⚙️ Running Tool: write_file       📄 linkedin/ai-agents/post.md  [Staged]
  ⚙️ Running Tool: generate_social_image

Agent: Saved to linkedin/ai-agents/post.md.
       Here's the post: ...
```

---

## Output Files — What Gets Created

All output files are created **only after clicking "Finalize & Commit to Disk"**.

### LinkedIn / Social Post

```
linkedin/
└── ai-agents/
    └── post.md
```

### Blog Post

```
blogs/
└── latest-llm-models/
    ├── post.md       ← Full blog post in Markdown
    └── hero.png      ← Cover image (Gemini Imagen generated)
```

### Research Notes

```
research/
└── latest-llm-models.md    ← Raw research from subagent
```

---

## Common Errors & Fixes

### ❌ `No documents are currently staged in the volatile memory cache`

**Cause:** The agent wrote its response as chat text instead of calling `write_file`.

**Fix:** This is fixed by `StatefulCacheMiddleware` and the `AGENTS.md` Staging Guidelines section. If it still happens, try a more explicit prompt: *"Write a LinkedIn post about AI and save it to the staging cache."*

### ❌ `No virtual drafts staged to save` (on Finalize)

**Cause:** The backend registry entry was cleared (e.g., after a previous Finalize) but the UI still shows staged files.

**Fix:** Fixed in `app.py` — the agent is now re-initialized automatically when `session_id` is missing from the registry.

### ❌ `Rate limit exceeded: free-models-per-day`

**Cause:** OpenRouter daily free model limit hit.

**Fix:** `ModelFallbackMiddleware` handles this automatically. If all free models are exhausted, wait until midnight UTC or add OpenRouter credits.

### ❌ `tool call validation failed: attempted to call tool 'researcher'`

**Cause:** Groq models reject subagent tool calls.

**Fix:** Only use OpenAI or OpenRouter models as the primary model. The current config already does this (`openai:gpt-5-nano`).

### ❌ Redis connection refused

**Cause:** Redis is not running.

**Fix:**
```bash
redis-server
# Or on systemd systems:
sudo systemctl start redis
```

### ❌ `TAVILY_API_KEY not set`

**Cause:** Missing key in `.env`.

**Fix:** The agent still works — it just won't have pre-fetched web search context. Add the key to `.env` for best results.

---

## Extending the Project

### Adding a New Skill

1. Create a folder: `skills/my-new-skill/`
2. Create `skills/my-new-skill/SKILL.md` with:
   ```yaml
   ---
   name: my-new-skill
   description: When to use this skill (used for detection)
   ---
   # Instructions for the agent...
   ```
3. `SkillsMiddleware` will auto-detect and inject it when needed.

### Adding a New Subagent

In `subagents.yaml`:
```yaml
fact-checker:
  description: Verify facts and statistics against credible sources.
  system_prompt: |
    You are a fact-checking assistant. Verify claims using web_search.
  tools:
    - web_search
```

### Adding a New Tool

In `content_writer.py`:
```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email with the content."""
    # your implementation
    ...

# Then add to create_content_writer():
tools=[generate_cover, generate_social_image, send_email]
```

### Changing the Primary Model

In `content_writer.py` inside `create_content_writer()`:
```python
# GPT-4o (more capable)
model="openai:gpt-4o",

# Claude Sonnet via OpenRouter
model="openrouter:anthropic/claude-sonnet-4-5",

# Gemini Flash (very fast & cheap)
model="openrouter:google/gemini-flash-1.5",
```

> ⚠️ **Do not use Groq models as primary** — they reject subagent calls.

### Adding a New Dashboard Tab

In `app.py`, add a new column or tab inside the `col_preview` section and read from `virtual_files` to display data.

---

## Quick Reference Card

| What | Where |
|------|-------|
| Web dashboard | `app.py` |
| Agent core & factory | `content_writer.py` — `create_content_writer()` |
| Volatile staging backend | `content_writer.py` — `StagingFilesystemBackend` |
| Staging instruction middleware | `content_writer.py` — `StatefulCacheMiddleware` |
| Commit to disk | `content_writer.py` — `finalize_and_save_to_disk()` |
| Agent personality | `AGENTS.md` |
| Staging rules for agent | `AGENTS.md` — "Staging & Filesystem Guidelines" |
| Blog post skill | `skills/blog-post/SKILL.md` |
| Social media skill | `skills/social-media/SKILL.md` |
| Subagent definitions | `subagents.yaml` |
| API keys | `.env` |
| Primary AI model | `content_writer.py` — `create_content_writer()` line ~301 |
| Fallback models | `content_writer.py` — `ModelFallbackMiddleware(...)` |
| Redis history functions | `content_writer.py` — `save_redis_history` / `load_redis_history` |
| Blog output | `blogs/<slug>/post.md` |
| Research output | `research/<slug>.md` |
| LinkedIn output | `linkedin/<topic>/post.md` |

---

*Updated to reflect the current Streamlit dashboard, volatile RAM staging, Redis persistence, and StagingFilesystemBackend architecture.*
