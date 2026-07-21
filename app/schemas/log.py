from pydantic import BaseModel, ConfigDict
from typing import Optional


class LogCreate(BaseModel):
    batch_id: int
    video_run_id: Optional[int] = None
    level: Optional[str] = "info"  # info, warning, error
    message: str


class LogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    batch_id: int
    video_run_id: Optional[int] = None
    timestamp: str
    level: str
    message: str
