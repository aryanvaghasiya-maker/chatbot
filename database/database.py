from decouple import config

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = config("DATABASE_URL")
from sqlalchemy import text


engine = create_engine(
    DATABASE_URL,
    echo=True
)

with engine.connect() as conn:
    print("=" * 40)
    print("Database :", conn.execute(text("SELECT current_database()")).scalar())
    print("User     :", conn.execute(text("SELECT current_user")).scalar())
    print("Schema   :", conn.execute(text("SELECT current_schema()")).scalar())
    print("=" * 40)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()
from sqlalchemy import text
