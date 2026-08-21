from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class StorageSetting(Base):
    __tablename__ = "storage_settings"

    id = Column(Integer, primary_key=True, index=True)
    root_path = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    batches = relationship("Batch", back_populates="storage")

class Batch(Base):
    __tablename__ = "batches"

    id = Column(String, primary_key=True, index=True) # UUID
    batch_number = Column(Integer, nullable=False)
    batch_date = Column(String, nullable=False) # YYYY-MM-DD
    storage_root_id = Column(Integer, ForeignKey("storage_settings.id"), nullable=False)
    folder_name = Column(String, nullable=True)  # Display name from folder picker
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    storage = relationship("StorageSetting", back_populates="batches")
    videos = relationship("Video", back_populates="batch", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint(
            "storage_root_id",
            "batch_date",
            "batch_number",
            name="uq_storage_date_batch_number"
        ),
    )

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True) # UUID
    batch_id = Column(String, ForeignKey("batches.id"), nullable=False)
    filename = Column(String, nullable=False)
    relative_path = Column(String, nullable=True)    # e.g. camera1/video001.mp4
    source_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    last_modified = Column(Integer, nullable=True)   # Unix timestamp (ms) from browser

    status = Column(String, default="QUEUED", nullable=False)
    verdict = Column(String, nullable=True)
    output_path = Column(String, nullable=True)

    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    processing_started_at = Column(DateTime, nullable=True)
    
    batch = relationship("Batch", back_populates="videos")

# Composite index: fast per-batch status queries used by worker + status API
Index("idx_video_batch_status", Video.batch_id, Video.status)

class ModelSetting(Base):
    __tablename__ = "model_settings"

    id = Column(Integer, primary_key=True, index=True)

    m1_total = Column(Integer, nullable=False, default=30)
    m1_pass = Column(Integer, nullable=False, default=18)

    m2_total = Column(Integer, nullable=False, default=30)
    m2_pass = Column(Integer, nullable=False, default=18)

    is_active = Column(Boolean, nullable=False, default=True)

    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
