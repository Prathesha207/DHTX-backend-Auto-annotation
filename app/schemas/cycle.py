from pydantic import BaseModel, ConfigDict
from typing import Optional


class CycleCreate(BaseModel):
    video_run_id: int
    cycle_number: int
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    duration_seconds: Optional[float] = None
    final_verdict: str  # NORMAL, ANOMALY, UNKNOWN
    output_video_path: str
    tube_blue: Optional[str] = None
    transition_middle: Optional[str] = None
    transition_end: Optional[str] = None
    detected_sequence: Optional[str] = None
    tube_order_result: Optional[str] = None
    anomaly_ratio: Optional[float] = None
    ok_votes: Optional[int] = None
    anomaly_votes: Optional[int] = None
    total_frames: Optional[int] = None
    warmup_frames: Optional[int] = None
    inference_frames: Optional[int] = None
    average_fps: Optional[float] = None


class CycleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    video_run_id: int
    cycle_number: int
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    duration_seconds: Optional[float] = None
    final_verdict: str
    output_video_path: str
    tube_blue: Optional[str] = None
    transition_middle: Optional[str] = None
    transition_end: Optional[str] = None
    detected_sequence: Optional[str] = None
    tube_order_result: Optional[str] = None
    anomaly_ratio: Optional[float] = None
    ok_votes: Optional[int] = None
    anomaly_votes: Optional[int] = None
    total_frames: Optional[int] = None
    warmup_frames: Optional[int] = None
    inference_frames: Optional[int] = None
    average_fps: Optional[float] = None
    created_at: str
