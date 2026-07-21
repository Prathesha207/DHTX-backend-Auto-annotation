from pydantic import BaseModel, ConfigDict
from typing import Optional


class VideoRunCreate(BaseModel):
    batch_id: int
    input_filename: str
    input_path: str
    queue_position: int


class VideoRunUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = None
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    output_video_path: Optional[str] = None
    excel_report_path: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class VideoRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    batch_id: int
    input_filename: str
    input_path: str
    output_video_path: Optional[str] = None
    excel_report_path: Optional[str] = None
    queue_position: int
    status: str
    progress: float
    current_frame: int
    total_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
