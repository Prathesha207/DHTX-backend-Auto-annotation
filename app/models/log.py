from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from app.database.database import Base


class Log(Base):

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)

    batch_id = Column(
        Integer,
        ForeignKey("batches.id", ondelete="CASCADE"),
        nullable=False
    )

    video_run_id = Column(
        Integer,
        ForeignKey("video_runs.id", ondelete="CASCADE"),
        nullable=True
    )

    cycle_id = Column(Integer, nullable=True)

    frame_number = Column(Integer, nullable=True)

    state = Column(String, nullable=True)

    timestamp = Column(String, nullable=False)

    level = Column(String, default="info")

    message = Column(String, nullable=False)

    batch = relationship(
        "Batch",
        back_populates="logs"
    )

    video_run = relationship(
        "VideoRun",
        back_populates="logs"
    )