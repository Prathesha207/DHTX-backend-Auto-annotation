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

@router.post("/{batch_id}/start", status_code=status.HTTP_202_ACCEPTED)
def start_batch(batch_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_batch = crud_batch.get_batch(db, batch_id)
    if not db_batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID {batch_id} not found"
        )
    from app.services.ml_runner import run_batch_inference_task
    background_tasks.add_task(run_batch_inference_task, batch_id)
    return {"message": "Batch inference started"}

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
