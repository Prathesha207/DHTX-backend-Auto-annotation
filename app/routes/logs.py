from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db

from app.schemas.log import (
    LogCreate,
    LogResponse,
)

from app.crud import log as crud_log
from app.services.log_service import LogService

router = APIRouter(
    prefix="/logs",
    tags=["Logs"],
)


@router.post(
    "/",
    response_model=LogResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_log(
    log_in: LogCreate,
    db: Session = Depends(get_db),
):

    if log_in.level.lower() == "error":
        return LogService.error(
            db=db,
            batch_id=log_in.batch_id,
            message=log_in.message,
            video_run_id=log_in.video_run_id,
        )

    if log_in.level.lower() == "warning":
        return LogService.warning(
            db=db,
            batch_id=log_in.batch_id,
            message=log_in.message,
            video_run_id=log_in.video_run_id,
        )

    return LogService.info(
        db=db,
        batch_id=log_in.batch_id,
        message=log_in.message,
        video_run_id=log_in.video_run_id,
    )


@router.get(
    "/",
    response_model=List[LogResponse],
)
def read_logs(
    batch_id: Optional[int] = None,
    video_run_id: Optional[int] = None,
    db: Session = Depends(get_db),
):

    if video_run_id is not None:
        return crud_log.get_video_logs(
            db,
            video_run_id,
        )

    if batch_id is not None:
        return crud_log.get_batch_logs(
            db,
            batch_id,
        )

    return crud_log.get_logs(db)


@router.get(
    "/{log_id}",
    response_model=LogResponse,
)
def read_log(
    log_id: int,
    db: Session = Depends(get_db),
):

    db_log = crud_log.get_log(
        db,
        log_id,
    )

    if db_log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log {log_id} not found",
        )

    return db_log

@router.get("/video/{video_run_id}")
def get_video_logs(
    video_run_id: int,
    db: Session = Depends(get_db),
):

    logs = crud_log.get_video_logs(
        db,
        video_run_id,
    )

    return [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "level": log.level,
            "message": log.message,
        }
        for log in logs
    ]

@router.delete(
    "/{log_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_log(
    log_id: int,
    db: Session = Depends(get_db),
):

    db_log = crud_log.get_log(
        db,
        log_id,
    )

    if db_log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log {log_id} not found",
        )

    crud_log.delete_log(
        db,
        db_log,
    )

    return None