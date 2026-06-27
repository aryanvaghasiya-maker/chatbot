from langchain_core.messages import HumanMessage, AIMessage
from database.database import SessionLocal
from database.chat_history import ChatSession

def load_conversation(session_id):
    db = SessionLocal()

    chat = db.query(ChatSession).filter(
        ChatSession.session_id == session_id
    ).first()

    db.close()

    if not chat:
        return []

    history = []

    for msg in chat.conversation:
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    return history

def save_conversation(session_id: str, messages):
    db = SessionLocal()

    conversation = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation.append({
                "role": "human",
                "content": msg.content
            })

        elif isinstance(msg, AIMessage):
            conversation.append({
                "role": "ai",
                "content": msg.content
            })

    chat = db.query(ChatSession).filter(
        ChatSession.session_id == session_id
    ).first()

    if chat:
        # Replace old conversation
        chat.conversation = conversation
    else:
        chat = ChatSession(
            session_id=session_id,
            conversation=conversation
        )
        db.add(chat)

    db.commit()
    db.close()