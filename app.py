from fastapi import FastAPI,Request
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from dependencies.llm import Agent
from agent.state import chatrequest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from decouple import config

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

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "hello"
    }

@app.post("/chat")
async def chat(message: chatrequest,request: Request):

    result = await graph.ainvoke({"messages": [HumanMessage(content=message.message)]},
                          config={"configurable": {"thread_id": message.session_id }})
    print(result)
    return {
        "response": result["messages"][-1].content
    }

