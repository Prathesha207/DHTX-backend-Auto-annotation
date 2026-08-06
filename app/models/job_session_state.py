from sqlalchemy import Column, Integer, JSON, String
from app.database.database import Base

class JobSessionState(Base):
    __tablename__ = "job_session_states"

    batch_id = Column(Integer, primary_key=True)
    state_json = Column(JSON, nullable=False)
    updated_at = Column(String, nullable=False)
