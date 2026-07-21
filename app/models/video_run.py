from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from app.database.database import Base


class VideoRun(Base):

    __tablename__ = "video_runs"

    id = Column(Integer, primary_key=True, index=True)

    batch_id = Column(
        Integer,
        ForeignKey("batches.id", ondelete="CASCADE"),
        nullable=False
    )

    input_filename = Column(String, nullable=False)

    input_path = Column(String, nullable=False)

    output_video_path = Column(String)

    excel_report_path = Column(String)

    queue_position = Column(Integer, nullable=False)

    status = Column(String, default="queued")

    progress = Column(Float, default=0)

    current_frame = Column(Integer, default=0)

    total_frames = Column(Integer)

    width = Column(Integer)

    height = Column(Integer)

    fps = Column(Float)

    duration_seconds = Column(Float)

    error_message = Column(String)

    created_at = Column(String, nullable=False)

    started_at = Column(String)

    completed_at = Column(String)

    batch = relationship(
        "Batch",
        back_populates="video_runs"
    )

    cycles = relationship(
        "Cycle",
        back_populates="video_run",
        cascade="all, delete"
    )

    logs = relationship(
        "Log",
        back_populates="video_run",
        cascade="all, delete"
    )