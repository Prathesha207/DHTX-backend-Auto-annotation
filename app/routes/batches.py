from app.models.video_run import VideoRun
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.batch import BatchCreate, BatchResponse
from app.crud import batch as crud_batch
from app.services.batch_service import BatchService

router = APIRouter(
    prefix="/batches",
    tags=["Batches"]
)


@router.post("/", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
def create_batch(batch_in: BatchCreate, db: Session = Depends(get_db)):
    return BatchService.create(
        db=db,
        batch_name=batch_in.batch_name,
        input_type=batch_in.input_type,
        input_path=batch_in.input_path,
        output_path=batch_in.output_path,
        total_videos=batch_in.total_videos,
    )


from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

@router.post("/{batch_id}/start")
def start_batch(batch_id: int, stream_hud: bool = False, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID {batch_id} not found"
        )
    
    db_batch.status = "queued"
    db.commit()
    
    # Import here to avoid circular imports during startup
    from app.services.job_queue import job_queue
    
    return {"message": "Batch added to inference queue"}

@router.post("/{batch_id}/cancel")
def cancel_batch(batch_id: int, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID {batch_id} not found"
        )
    
    from app.services.job_queue import job_queue
    canceled = job_queue.cancel_batch(batch_id)
    
    if not canceled:
        db_batch.status = "cancelled"
        db.commit()
        
    return {"message": "Batch cancelled"}

@router.get("/", response_model=List[BatchResponse])
def read_batches(db: Session = Depends(get_db)):
    return crud_batch.get_batches(db)


@router.get("/{batch_id}", response_model=BatchResponse)
def read_batch(batch_id: int, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID {batch_id} not found"
        )
    return db_batch

@router.get("/{batch_id}/status")
def get_batch_status(
    batch_id: int,
    db: Session = Depends(get_db),
):

    batch = crud_batch.get_batch_status(
        db=db,
        batch_id=batch_id,
    )

    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found",
        )

    current_video = (
        db.query(VideoRun)
        .filter(
            VideoRun.batch_id == batch.id,
            VideoRun.status == "running",
        )
        .first()
    )

    return {

        "batch_id": batch.id,

        "status": batch.status,

        "progress": batch.progress,

        "completed_videos": batch.completed_videos,

        "failed_videos": batch.failed_videos,

        "total_videos": batch.total_videos,

        "current_video_id": (
            current_video.id
            if current_video
            else None
        ),

        "current_video_name": (
            current_video.input_filename
            if current_video
            else None
        ),
    }
    
@router.delete("/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch(batch_id: int, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID {batch_id} not found"
        )
    crud_batch.delete_batch(db, db_batch)
    return None

import os
from fastapi.responses import FileResponse

@router.get("/{batch_id}/open", status_code=status.HTTP_200_OK)
def open_batch_folder(batch_id: int, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch or not db_batch.output_path:
        raise HTTPException(status_code=404, detail="Batch folder not found")
        
    folder_path = os.path.abspath(db_batch.output_path)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder does not exist on disk")
        
    try:
        os.startfile(folder_path)
        return {"message": "Folder opened"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{batch_id}/summary", status_code=status.HTTP_200_OK)
def get_batch_summary(batch_id: int, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    from app.models.video_run import VideoRun
    from app.models.cycle import Cycle
    
    videos = db.query(VideoRun).filter(VideoRun.batch_id == batch_id).all()
    
    completed = 0
    failed = 0
    cancelled = 0
    pending = 0
    
    total_cycles = 0
    normal = 0
    anomaly = 0
    unknown = 0
    
    for v in videos:
        if v.status == "completed":
            completed += 1
        elif v.status == "failed":
            failed += 1
        elif v.status in ("cancelled", "interrupted"):
            cancelled += 1
        else:
            pending += 1
            
        cycles = db.query(Cycle).filter(Cycle.video_run_id == v.id).all()
        total_cycles += len(cycles)
        for c in cycles:
            verdict = c.final_verdict.upper() if c.final_verdict else "UNKNOWN"
            if verdict == "NORMAL":
                normal += 1
            elif verdict == "ANOMALY":
                anomaly += 1
            else:
                unknown += 1

    return {
        "batch": f"Batch_{batch_id}",
        "date": db_batch.created_at[:10] if db_batch.created_at else "",
        "videos_discovered": len(videos),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "pending": pending,
        "videos_processed": completed + failed,
        "total_cycles": total_cycles,
        "normal": normal,
        "anomaly": anomaly,
        "unknown": unknown,
        "started_at": db_batch.started_at,
        "completed_at": db_batch.completed_at,
    }
