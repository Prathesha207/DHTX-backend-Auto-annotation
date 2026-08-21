import os
import shutil
import uuid
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import database, models
from crud import crud
from services.queue_service import inference_queue

router = APIRouter(tags=["Upload"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v", ".webm", ".ts"}

class RejectedItem(BaseModel):
    filename: str
    reason: str

class VideoResponse(BaseModel):
    id: str
    batch_id: str
    filename: str
    status: str
    message: str
    total_queued: int = 1
    rejected: List[RejectedItem] = []

class LocalFolderRequest(BaseModel):
    folder_path: str

class LocalVideoRequest(BaseModel):
    file_path: str

def _prepare_batch_and_dir(db: Session):
    active_storage = crud.get_active_storage(db)
    if not active_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System error: No active storage location configured in the database."
        )
    date_str = datetime.now().strftime("%Y-%m-%d")
    batch = crud.create_batch(db, active_storage.id, date_str)
    
    batch_dir = Path(active_storage.root_path) / "raw" / date_str / f"batch_{batch.batch_number}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    return batch, batch_dir

def _save_and_record(upload: UploadFile, batch, batch_dir: Path, db: Session) -> VideoResponse:
    video_id = str(uuid.uuid4())
    safe_filename = upload.filename or f"video_{video_id}.mp4"
    file_path = batch_dir / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file {upload.filename}: {str(e)}"
        )

    db_video = crud.create_video(
        db=db,
        video_id=video_id,
        batch_id=batch.id,
        filename=safe_filename,
        source_path=str(file_path.absolute()),
        file_size=os.path.getsize(file_path)
    )

    inference_queue.put_nowait(db_video.id)
    return VideoResponse(
        id=db_video.id,
        batch_id=batch.id,
        filename=db_video.filename,
        status=db_video.status,
        message="Successfully uploaded. Inference running automatically.",
        total_queued=1
    )

@router.post("/upload", response_model=VideoResponse)
def upload_single(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    """
    Upload a single video file from the browser. Starts inference automatically.
    """
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not a recognized video format. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    batch, batch_dir = _prepare_batch_and_dir(db)
    return _save_and_record(file, batch, batch_dir, db)

@router.post("/upload/folder")
def upload_folder():
    """
    Deprecated endpoint.
    """
    raise HTTPException(status_code=410, detail="Deprecated: Please use the /batch endpoints instead.")

@router.post("/upload/local-video", response_model=VideoResponse)
@router.post("/upload/local-folder", response_model=VideoResponse)
def upload_local(request: LocalVideoRequest | LocalFolderRequest, db: Session = Depends(database.get_db)):
    """
    Accepts an absolute server path to either a single video file OR a folder of videos.
    Automatically queues all videos into the batch and runs inference immediately.
    """
    path_str = getattr(request, "file_path", None) or getattr(request, "folder_path", "")
    clean_path_str = str(path_str).strip().strip('"').strip("'").strip()

    if not clean_path_str:
        raise HTTPException(status_code=400, detail="Path cannot be empty.")

    target_path = Path(clean_path_str)
    if not target_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist on server: '{clean_path_str}'"
        )

    batch, batch_dir = _prepare_batch_and_dir(db)
    rejected_files = []

    skipped_subfolder_count = 0

    # Handle Single File or Folder
    if target_path.is_file():
        video_files = [target_path]
    elif target_path.is_dir():
        video_files = [f for f in target_path.iterdir() if f.is_file()]
        for sub_dir in target_path.iterdir():
            if sub_dir.is_dir():
                skipped_subfolder_count += sum(1 for f in sub_dir.rglob("*") if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
    else:
        raise HTTPException(status_code=400, detail="Path is neither a file nor a directory.")

    first_video = None
    count = 0
    enqueued_ids = []
    
    for v_path in sorted(video_files):
        if v_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            rejected_files.append(RejectedItem(filename=v_path.name, reason="Unsupported extension"))
            continue

        # Deduplication check
        file_size = os.path.getsize(v_path)
        existing_video = db.query(models.Video).filter(
            models.Video.batch_id == batch.id,
            models.Video.filename == v_path.name
        ).first()

        if existing_video and existing_video.file_size == file_size:
            rejected_files.append(RejectedItem(filename=v_path.name, reason="Already ingested in this batch"))
            continue
            
        v_id = str(uuid.uuid4())
        dest_path = batch_dir / v_path.name
        
        # Handle duplicate filenames
        if dest_path.exists():
            dest_path = batch_dir / f"{v_path.stem}_{v_id[:8]}{v_path.suffix.lower()}"
            
        try:
            shutil.copy2(v_path, dest_path)
        except Exception as e:
            rejected_files.append(RejectedItem(filename=v_path.name, reason=f"Copy failed: {e}"))
            continue

        db_video = crud.create_video(
            db=db,
            video_id=v_id,
            batch_id=batch.id,
            filename=v_path.name,
            source_path=str(dest_path.absolute()),
            file_size=os.path.getsize(dest_path),
            commit=False
        )
        enqueued_ids.append(v_id)
        count += 1
        if not first_video:
            first_video = db_video

    db.commit()
    for vid in enqueued_ids:
        inference_queue.put_nowait(vid)

    if count == 0 and not rejected_files:
        raise HTTPException(
            status_code=400,
            detail=f"No video files found directly in '{clean_path_str}' (subfolders are not scanned)."
        )

    msg = f"Queued {count} video(s)."
    if rejected_files:
        msg += f" {len(rejected_files)} rejected/skipped."
    if skipped_subfolder_count > 0:
        msg += f" {skipped_subfolder_count} video(s) found in subfolders were ignored."

    return VideoResponse(
        id=first_video.id if first_video else str(uuid.uuid4()),
        batch_id=batch.id,
        filename=f"Batch of {count} videos" if target_path.is_dir() else (first_video.filename if first_video else "video"),
        status="QUEUED",
        message=msg,
        total_queued=count,
        rejected=rejected_files
    )
