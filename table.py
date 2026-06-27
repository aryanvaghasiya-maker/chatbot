from database.database import engine
from database.chat_history import Base

Base.metadata.create_all(bind=engine)

print("Tables created successfully.")