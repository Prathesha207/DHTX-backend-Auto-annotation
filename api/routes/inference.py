from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.queue_service import inference_queue, stop_batch_inference
from services.inference_service import get_active_video
from database import database, models
from crud import crud
from utils.storage_discovery import resolve_file_location

router = APIRouter(
    tags=["Inference"]
)

class InferenceRequest(BaseModel):
    video_id: str

class BasicInferenceResponse(BaseModel):
    success: bool
    video_id: str
    status: str

class StopInferenceResponse(BaseModel):
    success: bool
    status: str
    message: str

@router.post("/inference/stop", response_model=StopInferenceResponse)
def stop_inference():
    """
    Stop the currently active inference batch immediately. No parameters or body required.
    """
    res = stop_batch_inference()
    
    return StopInferenceResponse(
        success=res.get("success", True),
        status="STOPPED",
        message=res.get("message", "Batch inference stopped.")
    )

class ActiveInferenceResponse(BaseModel):
    is_active: bool
    video_id: str | None = None
    batch_id: str | None = None

@router.get("/inference/active", response_model=ActiveInferenceResponse)
def get_active_inference(db: Session = Depends(database.get_db)):
    """
    Returns the currently processing video and its batch, if any.
    Useful for restoring UI state if the user refreshes the page.
    """
    active_vid_id = get_active_video()
    if not active_vid_id:
        return ActiveInferenceResponse(is_active=False)
        
    video = db.query(models.Video).filter(models.Video.id == active_vid_id).first()
    if not video:
        return ActiveInferenceResponse(is_active=False)
        
    return ActiveInferenceResponse(
        is_active=True,
        video_id=video.id,
        batch_id=video.batch_id
    )

@router.post("/inference/video", response_model=BasicInferenceResponse)
def start_inference(request: InferenceRequest, db: Session = Depends(database.get_db)):
    """
    Queue an existing video for inference.
    """
    video = db.query(models.Video).filter(models.Video.id == request.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status in ("QUEUED", "PROCESSING"):
        raise HTTPException(status_code=400, detail="Video is already queued or processing")

    video.status = "QUEUED"
    video.verdict = None
    db.commit()

    inference_queue.put_nowait(video.id)

    return BasicInferenceResponse(
        success=True,
        video_id=request.video_id,
        status="queued"
    )

class InferenceStatusResponse(BaseModel):
    video_id: str
    filename: str
    batch: str
    status: str
    result: dict
    output: dict

@router.get("/inference/{video_id}/status", response_model=InferenceStatusResponse)
def get_inference_status(video_id: str, db: Session = Depends(database.get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    batch = video.batch
    batch_name = f"batch_{batch.batch_number}" if batch else "unknown"
    
    response = {
        "video_id": video.id,
        "filename": video.filename,
        "batch": batch_name,
        "status": video.status,
        "result": {"verdict": video.verdict},
        "output": {"path": video.output_path},
    }
    return InferenceStatusResponse(**response)

@router.get("/inference/{video_id}/video")
def get_inference_video(video_id: str, db: Session = Depends(database.get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    active_storage = crud.get_active_storage(db)
    active_root = active_storage.root_path if active_storage else None
    
    # 1. Return processed output video if available
    out_file = resolve_file_location(video.output_path, active_root, Path(video.output_path).name if video.output_path else None)
    if out_file and Path(out_file).is_file():
        return FileResponse(out_file, media_type="video/mp4")

    # 2. Return original source video if available
    src_file = resolve_file_location(video.source_path, active_root, video.filename)
    if src_file and Path(src_file).is_file():
        return FileResponse(src_file, media_type="video/mp4")
        
    raise HTTPException(status_code=404, detail="Video file not found on disk")

@router.get("/inference/history/list")
def get_recent_inferences(db: Session = Depends(database.get_db)):
    videos = db.query(models.Video).order_by(models.Video.uploaded_at.desc()).limit(20).all()
    results = []
    for v in videos:
        batch_name = f"batch_{v.batch.batch_number}" if v.batch else "unknown"
        results.append({
            "video_id": v.id,
            "filename": v.filename,
            "batch": batch_name,
            "status": v.status,
            "verdict": v.verdict,
            "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None,
            "has_output_video": bool(v.output_path and Path(v.output_path).is_file())
        })
    return results
