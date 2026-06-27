import os
from fastapi import FastAPI, Response
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from contextlib import asynccontextmanager
from agent.schema.llm import ResumeAgent

orchestrator = ResumeAgent()
compiled_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.setup()
        compiled_graph = orchestrator.build_graph(checkpointer=saver)
        yield

app = FastAPI(lifespan=lifespan)

# Endpoint to get the Visual Graph Workflow
@app.get("/workflow")
async def get_workflow_graph():
    # Generates the Mermaid representation of your workflow
    mermaid_graph = compiled_graph.get_graph().draw_mermaid()
    return {"mermaid_code": mermaid_graph}

@app.post("/optimize")
async def optimize_resume(data: dict):
    config = {"configurable": {"thread_id": data["thread_id"]}}
    initial_state = {
        "raw_resume": data["resume"],
        "job_description": data["job_description"],
        "iterations": 0,
        "critique_history": [] # <--- FIX: Add the empty list brackets here
    }
    result = await compiled_graph.ainvoke(initial_state, config)
    return {
        "score": result["ats_score"],
        "latex_code": result["suggested_companies"] # This contains the LaTeX string
    }
