from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from decouple import config

DATABASE_URL = config(
    "DATABASE_URL",
    default="sqlite:///company.db",
)
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

        print("=" * 50)
        print("✅ SQLite Connected Successfully")
        print("Database :", DATABASE_URL)
        print(
            "SQLite Version :",
            conn.execute(text("SELECT sqlite_version()")).scalar(),
        )
        print("=" * 50)

except Exception as e:
    print("=" * 50)
    print("❌ Database Connection Failed")
    print(e)
    print("=" * 50)