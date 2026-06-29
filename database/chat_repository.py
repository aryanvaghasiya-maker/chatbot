import json
from database.database import SessionLocal
from database.chat_history import ChatSession
import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

def save_to_redis(session_id, role, content):
    redis_client.rpush(
        f"chat:{session_id}",
        json.dumps({
            "role": role,
            "content": content
        })
    )
def get_messages(session_id):
    return redis_client.lrange(f"chat:{session_id}", 0, -1)

def delete_messages(session_id):
    redis_client.delete(f"chat:{session_id}")

def save_redis_to_database(session_id):

    db = SessionLocal()

    try:
        messages = get_messages(session_id)

        print(messages)
        for i in range(0, len(messages) - 1, 2):
            human = json.loads(messages[i])
            ai = json.loads(messages[i + 1])
            db.add(
                ChatSession(
                    session_id=session_id,
                    human_mess=human["content"],
                    ai_mess=ai["content"]
                )
            )
        
        db.commit()

    finally:

        db.close()

    delete_messages(session_id)