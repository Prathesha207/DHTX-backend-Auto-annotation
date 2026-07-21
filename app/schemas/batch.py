from pydantic import BaseModel, ConfigDict
from typing import Optional


class BatchCreate(BaseModel):
    batch_name: Optional[str] = None
    input_type: str
    input_path: str
    output_path: str
    total_videos: int


class BatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    batch_name: Optional[str] = None
    input_type: str
    input_path: str
    output_path: str
    status: str
    total_videos: int
    completed_videos: int
    failed_videos: int
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
