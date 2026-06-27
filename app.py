import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from agent.schema.llm import ResumeAgent
from agent.schema.schema import OptimizedResumeOutput

orchestrator = None
redis_pool_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, redis_pool_url
    redis_pool_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    print("[Lifespan]: Initializing Resume AI Orchestration Engine...")
    orchestrator = ResumeAgent(max_loops=3, target_score=85)
    print("[Lifespan]: System Ready for Asynchronous Operations.")
    yield
    print("[Lifespan]: Shutting Down.")

app = FastAPI(title="Async Production LangGraph Service", lifespan=lifespan)

class OptimizeRequest(BaseModel):
    resume: str
    job_description: str
    thread_id: str = "async_api_session_token"

class OptimizeResponse(BaseModel):
    thread_id: str
    ats_score: int
    optimized_resume: OptimizedResumeOutput
    overleaf_latex_code: str
    suggested_companies: str

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_resume_endpoint(payload: OptimizeRequest):
    global orchestrator, redis_pool_url
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine initialization unverified.")

    async with AsyncRedisSaver.from_conn_string(redis_pool_url) as checkpointer:
        try:
            await checkpointer.setup()
            compiled_graph = orchestrator.build_graph(checkpointer=checkpointer)
            
            config_params = {"configurable": {"thread_id": payload.thread_id}}
            
            initial_inputs = {
                "raw_resume": payload.resume, 
                "job_description": payload.job_description,
                "iterations": 0,
                "ats_score": 0,
                "extracted_skills": [],
                "missing_keywords": [],
                "critique_history": []
            }
            
            print(f"[API Endpoint]: Executing Graph Processing for Thread: {payload.thread_id}")
            execution_output = await compiled_graph.ainvoke(initial_inputs, config_params)
            
            return OptimizeResponse(
                thread_id=payload.thread_id,
                ats_score=execution_output.get("ats_score", 0),
                optimized_resume=execution_output["optimized_resume"],
                overleaf_latex_code=execution_output.get("suggested_companies", ""),
                suggested_companies=execution_output.get("placement_market_report", "")
            )
        except Exception as e:
            print(f"[API Error]: Execution pipeline failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Graph Processing Fault: {str(e)}")
