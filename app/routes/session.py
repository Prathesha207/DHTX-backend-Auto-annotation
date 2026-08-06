from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.services.job_queue import BackgroundJobQueue
from app.services.job_session_manager import JobSessionManager
from app.models.batch import Batch
from app.models.video_run import VideoRun
import time

router = APIRouter(
    prefix="/session",
    tags=["Session"]
)

@router.get("/")
def get_current_session(db: Session = Depends(get_db)):
    job_queue = BackgroundJobQueue.get_instance()
    running_batch_id = job_queue.running_batch_id

    # If no batch is running in the queue
    if not running_batch_id:
        return {
            "running": False,
            "batch_id": None
        }

    # Fetch the session
    session_mgr = JobSessionManager.get_instance()
    job_session = session_mgr.get_session(running_batch_id)

    # Fetch DB batch for names
    batch_db = db.query(Batch).filter(Batch.id == running_batch_id).first()
    batch_name = batch_db.batch_name if batch_db else None
    
    current_video_name = None
    if batch_db:
        video = db.query(VideoRun).filter(
            VideoRun.batch_id == running_batch_id,
            VideoRun.status == "running"
        ).first()
        if video:
            current_video_name = video.input_filename

    if job_session:
        snapshot = job_session.to_snapshot_dict()
        snapshot["running"] = True
        snapshot["batch_id"] = running_batch_id
        snapshot["batch_name"] = batch_name
        snapshot["current_video_name"] = current_video_name
        
        snapshot["processed"] = job_session.current_video - 1 if job_session.current_video > 0 else 0
        snapshot["remaining"] = job_session.total_videos - snapshot["processed"]
        
        try:
            from datetime import datetime
            start_ts = datetime.fromisoformat(job_session.started_at.replace('Z', '+00:00')).timestamp()
            snapshot["uptime"] = int(time.time() - start_ts)
        except Exception:
            snapshot["uptime"] = 0
            
        snapshot["current_state"] = job_session.latest_statistics.get("state", "UNKNOWN")
        snapshot["current_phase"] = job_session.latest_statistics.get("phase", "UNKNOWN")
        snapshot["websocket_clients"] = job_session.connected_clients
        return snapshot

    return {
        "running": True,
        "batch_id": running_batch_id,
        "batch_name": batch_name,
        "current_video": 1,
        "total_videos": batch_db.total_videos if batch_db else 1,
        "processed": 0,
        "remaining": batch_db.total_videos if batch_db else 1,
        "current_state": "STARTING",
        "current_phase": "Initializing",
        "uptime": 0,
        "progress": 0.0,
        "current_video_name": current_video_name,
        "websocket_clients": 0
    }
