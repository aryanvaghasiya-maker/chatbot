from fastapi import FastAPI,Request
from langchain_core.messages import HumanMessage,AIMessage
from contextlib import asynccontextmanager
from dependencies.llm import Agent
from agent.state import chatrequest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from decouple import config
from database.chat_repository import (save_to_redis,save_redis_to_database)
from database.chat_history import ChatSession
from database.database import SessionLocal
import uvicorn


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
async def chat(message: chatrequest):

    result = await graph.ainvoke(
        {"messages":[HumanMessage(content=message.message)]},config={"configurable":{"thread_id":message.session_id}})

    ai_reply = result["messages"][-1].content
    save_to_redis(message.session_id,"human",message.message,)
    save_to_redis(message.session_id,"ai",ai_reply,)

    return {
        "response": ai_reply
    }
@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):

    db = SessionLocal()

    chats = (
        db.query(ChatSession).filter(ChatSession.session_id == session_id).order_by(ChatSession.id).all())

    db.close()

    history = []

    for chat in chats:
        if chat.human_mess:
            history.append(HumanMessage(content=chat.human_mess))
        if chat.ai_mess:
            history.append(AIMessage(content=chat.ai_mess))

    return history

@app.post("/end-chat/{session_id}")
async def end_chat(session_id: str):

    save_redis_to_database(session_id)

    return {
        "message": "Conversation saved successfully."
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)