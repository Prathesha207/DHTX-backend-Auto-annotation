from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from app.database.database import Base


class Cycle(Base):

    __tablename__ = "cycles"

    id = Column(Integer, primary_key=True, index=True)

    video_run_id = Column(
        Integer,
        ForeignKey("video_runs.id", ondelete="CASCADE"),
        nullable=False
    )

    cycle_number = Column(Integer, nullable=False)

    start_frame = Column(Integer)

    end_frame = Column(Integer)

    duration_seconds = Column(Float)

    final_verdict = Column(String, nullable=False)

    output_video_path = Column(String, nullable=False)

    tube_blue = Column(String)

    transition_middle = Column(String)

    transition_end = Column(String)

    detected_sequence = Column(String)

    tube_order_result = Column(String)

    anomaly_ratio = Column(Float)

    ok_votes = Column(Integer)

    anomaly_votes = Column(Integer)

    total_frames = Column(Integer)

    warmup_frames = Column(Integer)

    inference_frames = Column(Integer)

    average_fps = Column(Float)

    created_at = Column(String, nullable=False)

    video_run = relationship(
        "VideoRun",
        back_populates="cycles"
    )