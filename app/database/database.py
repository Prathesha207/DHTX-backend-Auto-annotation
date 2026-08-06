from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./db/app.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# SQLite does not enforce foreign keys by default.
# This listener enables it on every new connection.
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()