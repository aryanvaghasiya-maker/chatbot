import os
import redis 
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict, messages_to_dict
import json

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
