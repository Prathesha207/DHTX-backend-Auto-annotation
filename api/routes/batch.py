"""
batch.py — Production batch API

Endpoints:
  POST /api/batch/create
  POST /api/batch/{batch_id}/enqueue-bulk
  GET  /api/batch/{batch_id}/status
  GET  /api/batch/{batch_id}/queue
  GET  /api/batch/{batch_id}/manifest
  POST /api/batch/{batch_id}/stop
  POST /api/batch/{batch_id}/retry
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import database, models
from crud import crud
from services.queue_service import inference_queue, stop_batch_inference, update_excel_log_verdict, cancelled_batch_ids

router = APIRouter(prefix="/batch", tags=["Batch"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v", ".webm", ".ts"}
MAX_FILES_PER_CHUNK = 25
MAX_CHUNK_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_SINGLE_FILE_BYTES = 1000 * 1024 * 1024 # 1 GB - higher ceiling for solo oversized files


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_batch_or_404(batch_id: str, db: Session) -> models.Batch:
    batch = db.query(models.Batch).filter(models.Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return batch


def _current_video(batch_id: str, db: Session) -> Optional[dict]:
    """Return earliest PROCESSING video, else earliest QUEUED, else None."""
    for s in ("PROCESSING", "QUEUED"):
        v = (
            db.query(models.Video)
            .filter(models.Video.batch_id == batch_id, models.Video.status == s)
            .order_by(models.Video.uploaded_at.asc())
            .first()
        )
        if v:
            return {"video_id": v.id, "filename": v.filename, "status": v.status}
    return None


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CreateBatchResponse(BaseModel):
    batch_id: str
    batch_number: int

class EnqueuedItem(BaseModel):
    filename: str
    video_id: str

class RejectedItem(BaseModel):
    filename: str
    reason: str

class EnqueueBulkResponse(BaseModel):
    enqueued: List[EnqueuedItem]
    rejected: List[RejectedItem]

class BatchStatusResponse(BaseModel):
    batch_id: str
    folder_name: Optional[str]
    total: int
    queued: int
    processing: int
    completed: int
    failed: int
    stopped: int
    current_video: Optional[Any]

class VideoItem(BaseModel):
    id: str
    filename: str
    relative_path: Optional[str]
    file_size: int
    status: str
    verdict: Optional[str]

class QueuePageResponse(BaseModel):
    items: List[VideoItem]
    page: int
    page_size: int
    total: int

class ManifestItem(BaseModel):
    filename: str
    relative_path: Optional[str]
    file_size: int
    last_modified: Optional[int]

class ManifestResponse(BaseModel):
    batch_id: str
    items: List[ManifestItem]

class RetryRequest(BaseModel):
    statuses: List[str] = ["FAILED", "STOPPED"]

class StopResponse(BaseModel):
    success: bool
    message: str

class RetryResponse(BaseModel):
    re_queued: int


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/create", response_model=CreateBatchResponse)
def create_batch(
    folder_name: Optional[str] = None,
    db: Session = Depends(database.get_db),
):
    """Create a new empty batch. Returns batch_id for subsequent enqueue calls."""
    active_storage = crud.get_active_storage(db)
    if not active_storage:
        raise HTTPException(status_code=500, detail="No active storage configured")

    date_str = datetime.now().strftime("%Y-%m-%d")
    batch = crud.create_batch(db, active_storage.id, date_str)

    # Patch folder_name (crud.create_batch doesn't accept it yet)
    if folder_name:
        batch.folder_name = folder_name
        db.commit()

    return CreateBatchResponse(batch_id=batch.id, batch_number=batch.batch_number)


@router.post("/{batch_id}/enqueue-bulk", response_model=EnqueueBulkResponse)
async def enqueue_bulk(
    request: Request,
    batch_id: str,
    files: List[UploadFile] = File(...),
    relative_paths: Optional[str] = Form(None),  # JSON array of relative paths
    last_modifieds: Optional[str] = Form(None),   # JSON array of timestamps (ms)
    db: Session = Depends(database.get_db),
):
    """
    Accept up to MAX_FILES_PER_CHUNK video files. Validate, save, record in DB,
    add to inference queue. Return per-file enqueued/rejected breakdown.
    Never runs inference inside this request.
    """
    content_length = int(request.headers.get("content-length", 0))
    
    # Give solo large files a higher ceiling to prevent dropping oversized videos
    limit = MAX_SINGLE_FILE_BYTES if len(files) == 1 else MAX_CHUNK_BYTES
    
    if content_length > limit:
        raise HTTPException(status_code=413, detail=f"Request size {content_length} exceeds limit of {limit}")

    if len(files) > MAX_FILES_PER_CHUNK:
        raise HTTPException(status_code=400, detail=f"Cannot upload more than {MAX_FILES_PER_CHUNK} files at once")

    import json

    batch = _get_batch_or_404(batch_id, db)
    active_storage = crud.get_active_storage(db)
    if not active_storage:
        raise HTTPException(status_code=500, detail="No active storage configured")

    if batch_id in cancelled_batch_ids:
        for upload in files:
            fname = Path((upload.filename or "video.mp4").replace("\\", "/")).name
            rejected.append(RejectedItem(filename=fname, reason="Batch upload cancelled by user"))
        return EnqueueBulkResponse(enqueued=[], rejected=rejected)

    # Parse optional metadata arrays
    rel_paths: List[Optional[str]] = []
    lm_list: List[Optional[int]] = []
    if relative_paths:
        try:
            rel_paths = json.loads(relative_paths)
        except Exception:
            rel_paths = []
    if last_modifieds:
        try:
            lm_list = json.loads(last_modifieds)
        except Exception:
            lm_list = []

    # Build the batch output directory
    batch_dir = (
        Path(active_storage.root_path)
        / "raw"
        / batch.batch_date
        / f"batch_{batch.batch_number}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    enqueued: List[EnqueuedItem] = []
    rejected: List[RejectedItem] = []

    for idx, upload in enumerate(files):
        raw_fname = upload.filename or f"video_{uuid.uuid4()}.mp4"
        # Sanitize: always use just the base name (no path separators)
        fname = Path(raw_fname.replace("\\", "/")).name
        if not fname:
            fname = f"video_{uuid.uuid4()}.mp4"
        ext = Path(fname).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(RejectedItem(filename=fname, reason="Unsupported file type"))
            continue

        video_id = str(uuid.uuid4())
        dest = batch_dir / fname

        # Handle duplicate filenames in the same batch folder
        if dest.exists():
            stem = Path(fname).stem
            dest = batch_dir / f"{stem}_{video_id[:8]}{ext}"

        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
        except Exception as e:
            rejected.append(RejectedItem(filename=fname, reason=f"Save error: {e}"))
            continue

        rel_path = rel_paths[idx] if idx < len(rel_paths) else None
        last_mod = lm_list[idx] if idx < len(lm_list) else None

        db_video = crud.create_video(
            db=db,
            video_id=video_id,
            batch_id=batch_id,
            filename=fname,
            source_path=str(dest.absolute()),
            file_size=os.path.getsize(dest),
            relative_path=rel_path,
            last_modified=last_mod,
            commit=False
        )

        enqueued.append(EnqueuedItem(filename=fname, video_id=video_id))

    db.commit()
    
    for item in enqueued:
        inference_queue.put_nowait(item.video_id)

    return EnqueueBulkResponse(enqueued=enqueued, rejected=rejected)


@router.get("/{batch_id}/status", response_model=BatchStatusResponse)
def get_batch_status(batch_id: str, db: Session = Depends(database.get_db)):
    """Aggregate counts + current_video. Frontend polls this every 2s."""
    batch = _get_batch_or_404(batch_id, db)

    counts = (
        db.query(models.Video.status, func.count(models.Video.id))
        .filter(models.Video.batch_id == batch_id)
        .group_by(models.Video.status)
        .all()
    )
    status_map = {s: c for s, c in counts}
    total = sum(status_map.values())

    return BatchStatusResponse(
        batch_id=batch_id,
        folder_name=batch.folder_name,
        total=total,
        queued=status_map.get("QUEUED", 0),
        processing=status_map.get("PROCESSING", 0),
        completed=status_map.get("COMPLETED", 0),
        failed=status_map.get("FAILED", 0),
        stopped=status_map.get("STOPPED", 0),
        current_video=_current_video(batch_id, db),
    )


@router.get("/{batch_id}/queue", response_model=QueuePageResponse)
def get_batch_queue(
    batch_id: str,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(database.get_db),
):
    """Paginated list of videos in a batch. Never returns >page_size rows."""
    _get_batch_or_404(batch_id, db)
    page = max(1, page)
    page_size = min(100, max(1, page_size))
    offset = (page - 1) * page_size

    total = db.query(func.count(models.Video.id)).filter(models.Video.batch_id == batch_id).scalar()
    videos = (
        db.query(models.Video)
        .filter(models.Video.batch_id == batch_id)
        .order_by(models.Video.uploaded_at.asc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = [
        VideoItem(
            id=v.id,
            filename=v.filename,
            relative_path=v.relative_path,
            file_size=v.file_size,
            status=v.status,
            verdict=v.verdict,
        )
        for v in videos
    ]

    return QueuePageResponse(items=items, page=page, page_size=page_size, total=total)


@router.get("/{batch_id}/manifest", response_model=ManifestResponse)
def get_batch_manifest(batch_id: str, db: Session = Depends(database.get_db)):
    """Lightweight manifest for resume/dedup — filename, relative_path, file_size, last_modified only."""
    _get_batch_or_404(batch_id, db)
    videos = (
        db.query(
            models.Video.filename,
            models.Video.relative_path,
            models.Video.file_size,
            models.Video.last_modified,
        )
        .filter(models.Video.batch_id == batch_id)
        .order_by(models.Video.uploaded_at.asc())
        .all()
    )

    items = [
        ManifestItem(
            filename=v.filename,
            relative_path=v.relative_path,
            file_size=v.file_size,
            last_modified=v.last_modified,
        )
        for v in videos
    ]

    return ManifestResponse(batch_id=batch_id, items=items)


@router.get("/{batch_id}/log", response_class=FileResponse)
def download_batch_log(batch_id: str, db: Session = Depends(database.get_db)):
    """Download the generated excel log for a batch."""
    batch = _get_batch_or_404(batch_id, db)
    active_storage = crud.get_active_storage(db)
    storage_root = Path(active_storage.root_path if active_storage else (batch.storage.root_path if batch.storage else "./storage"))
    
    batch_dir = (
        storage_root
        / "processed"
        / batch.batch_date
        / f"batch_{batch.batch_number}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "inspection_log.xlsx"
    
    if not log_path.exists():
        for video in batch.videos:
            verdict = video.verdict or ("ABORTED" if video.status == "STOPPED" else "UNKNOWN")
            update_excel_log_verdict(batch_dir, video.filename, verdict, video.output_path)

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Inference log could not be created for this batch")
        
    return FileResponse(
        path=str(log_path),
        filename=f"batch_{batch.batch_number}_inspection_log.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.post("/{batch_id}/stop", response_model=StopResponse)
def stop_batch(batch_id: str, db: Session = Depends(database.get_db)):
    """Stop all inference for this batch. Maps to existing stop_batch_inference."""
    _get_batch_or_404(batch_id, db)
    res = stop_batch_inference(batch_id=batch_id)
    return StopResponse(success=res.get("success", True), message=res.get("message", "Stopped"))


@router.post("/{batch_id}/retry", response_model=RetryResponse)
def retry_batch(
    batch_id: str,
    body: RetryRequest = RetryRequest(),
    db: Session = Depends(database.get_db),
):
    """Re-queue FAILED and/or STOPPED videos in this batch. Rejects invalid status lists."""
    _get_batch_or_404(batch_id, db)

    ALLOWED_RETRY = {"FAILED", "STOPPED"}
    for s in body.statuses:
        if s not in ALLOWED_RETRY:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry status '{s}'. Only FAILED and STOPPED are allowed."
            )

    videos = (
        db.query(models.Video)
        .filter(
            models.Video.batch_id == batch_id,
            models.Video.status.in_(body.statuses),
        )
        .order_by(models.Video.uploaded_at.asc())
        .all()
    )

    # Discard the batch from the cancelled set so inference can proceed
    cancelled_batch_ids.discard(batch_id)

    count = 0
    enqueued_ids = []
    from services.queue_service import cancelled_video_ids
    for v in videos:
        v.status = "QUEUED"
        v.verdict = None
        enqueued_ids.append(v.id)
        cancelled_video_ids.discard(v.id)
        count += 1
    
    db.commit()

    import asyncio
    try:
        loop = asyncio.get_running_loop()
        for vid in enqueued_ids:
            loop.call_soon_threadsafe(inference_queue.put_nowait, vid)
    except RuntimeError:
        for vid in enqueued_ids:
            inference_queue.put_nowait(vid)

    return RetryResponse(re_queued=count)
