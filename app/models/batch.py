from app.database.database import Base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship

class Batch(Base):
    __tablename__ = "batches"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_name = Column(String)
    input_type = Column(String, nullable=False)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=False)
    status = Column(String, default="queued")
    
    total_videos = Column(Integer, nullable=False)
    completed_videos = Column(Integer, default=0)
    failed_videos = Column(Integer, default=0)
    progress = Column(Float, default=0)
    interrupted_reason = Column(String, nullable=True)

    created_at = Column(String, nullable=False)
    started_at = Column(String)
    completed_at = Column(String)

    video_runs = relationship(
        "VideoRun",
        back_populates="batch",
        cascade="all, delete"
    )

    logs = relationship(
        "Log",
        back_populates="batch",
        cascade="all, delete"
    )