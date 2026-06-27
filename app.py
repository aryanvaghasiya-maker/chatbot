from fastapi import FastAPI,Request
from langchain_core.messages import HumanMessage,AIMessage
from contextlib import asynccontextmanager
from dependencies.llm import Agent
from agent.state import chatrequest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from decouple import config
from database.chat_repository import (save_conversation,load_conversation)
from database.chat_history import ChatSession
from database.database import SessionLocal
from uuid import UUID
from fastapi import HTTPException

graph = None
checkpointer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = config("REDIS_URL",default="redis://localhost:6379")
    print("Connecting to Redis...")

    global graph, checkpointer

    async with (
        AsyncRedisSaver.from_conn_string(redis_url) as checkpointer,):
        try:
     
            await checkpointer.setup()
            agent = Agent()
            graph = agent.build_graph(checkpointer=checkpointer)

            print("Redis Connected & Agent Ready")

            yield

        except Exception as e:
            print(f"Startup failed: {e}")
            raise

        finally:
            print("Cleaning up resources...")
    print("Redis Connection safely closed. Shutting Down.")

@staticmethod 
def get_history(session_id): 
    db = SessionLocal()
    try: 
        return ( db.query(ChatSession) 
                .filter(ChatSession.session_id == session_id) 
                .order_by(ChatSession.created_at) 
                .all() )
    finally: 
        db.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "hello"
    }

@app.post("/chat")
async def chat(message: chatrequest, request: Request):

    result = await graph.ainvoke({
        "messages":[HumanMessage(content=message.message)]},
        config={"configurable": {"thread_id": message.session_id}
    })
    
    save_conversation(
        session_id=message.session_id,
        messages=result["messages"],
    )
    return {
        "response": result["messages"][-1].content
    }

@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):

    history = load_conversation(session_id)

    if not history:
        raise HTTPException(status_code=404,detail="Conversation not found",)
    conversation = []

    for msg in history:
        if isinstance(msg, HumanMessage):
            conversation.append({"role": "human","content": msg.content,})

        elif isinstance(msg, AIMessage):
            conversation.append({"role": "ai","content": msg.content,})

    return {
        "session_id": session_id,
        "conversation": conversation,
    }