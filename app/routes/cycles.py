from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db

from app.schemas.cycle import (
    CycleCreate,
    CycleResponse,
)

from app.crud import cycle as crud_cycle
from app.services.cycle_service import CycleService

router = APIRouter(
    prefix="/cycles",
    tags=["Cycles"],
)


@router.post(
    "/",
    response_model=CycleResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_cycle(
    cycle_in: CycleCreate,
    db: Session = Depends(get_db),
):

    return CycleService.save_cycle(
        db=db,
        video_run_id=cycle_in.video_run_id,
        cycle_number=cycle_in.cycle_number,
        start_frame=cycle_in.start_frame,
        end_frame=cycle_in.end_frame,
        duration_seconds=cycle_in.duration_seconds,
        final_verdict=cycle_in.final_verdict,
        output_video_path=cycle_in.output_video_path,
        tube_blue=cycle_in.tube_blue,
        transition_middle=cycle_in.transition_middle,
        transition_end=cycle_in.transition_end,
        detected_sequence=cycle_in.detected_sequence,
        tube_order_result=cycle_in.tube_order_result,
        anomaly_ratio=cycle_in.anomaly_ratio,
        ok_votes=cycle_in.ok_votes,
        anomaly_votes=cycle_in.anomaly_votes,
        total_frames=cycle_in.total_frames,
        warmup_frames=cycle_in.warmup_frames,
        inference_frames=cycle_in.inference_frames,
        average_fps=cycle_in.average_fps,
    )


@router.get(
    "/",
    response_model=List[CycleResponse],
)
def read_cycles(
    video_run_id: Optional[int] = None,
    db: Session = Depends(get_db),
):

    if video_run_id is None:
        return crud_cycle.get_cycles(db)

    return crud_cycle.get_video_cycles(
        db,
        video_run_id=video_run_id,
    )


@router.get(
    "/{cycle_id}",
    response_model=CycleResponse,
)
def read_cycle(
    cycle_id: int,
    db: Session = Depends(get_db),
):

    db_cycle = crud_cycle.get_cycle(
        db,
        cycle_id,
    )

    if db_cycle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cycle {cycle_id} not found",
        )

    return db_cycle


@router.delete(
    "/{cycle_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_cycle(
    cycle_id: int,
    db: Session = Depends(get_db),
):

    db_cycle = crud_cycle.get_cycle(
        db,
        cycle_id,
    )

    if db_cycle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cycle {cycle_id} not found",
        )

    crud_cycle.delete_cycle(
        db,
        db_cycle,
    )

    return None