from app.crud import cycle as crud_cycle
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.video_run import VideoRunCreate, VideoRunUpdate, VideoRunResponse
from app.crud import video_run as crud_video_run
from app.services.video_run_service import VideoRunService

router = APIRouter(
    prefix="/video-runs",
    tags=["Video Runs"]
)


@router.post("/", response_model=VideoRunResponse, status_code=status.HTTP_201_CREATED)
def create_video_run(video_run_in: VideoRunCreate, db: Session = Depends(get_db)):
    return VideoRunService.create(
        db=db,
        batch_id=video_run_in.batch_id,
        input_filename=video_run_in.input_filename,
        input_path=video_run_in.input_path,
        queue_position=video_run_in.queue_position,
    )


@router.get("/", response_model=List[VideoRunResponse])
def read_video_runs(
    batch_id: Optional[int] = None,
    db: Session = Depends(get_db),
):

    if batch_id is None:
        return crud_video_run.get_video_runs(db)

    return crud_video_run.get_batch_video_runs(
        db,
        batch_id=batch_id,
    )


@router.get("/{video_run_id}", response_model=VideoRunResponse)
def read_video_run(video_run_id: int, db: Session = Depends(get_db)):
    db_video_run = crud_video_run.get_video_run(db, video_run_id)
    if not db_video_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VideoRun with ID {video_run_id} not found"
        )
    return db_video_run

@router.get("/{video_run_id}/progress")
def get_video_progress(
    video_run_id: int,
    db: Session = Depends(get_db),
):

    video = crud_video_run.get_video_progress(
        db,
        video_run_id,
    )

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    return {

        "id": video.id,

        "status": video.status,

        "progress": video.progress,

        "current_frame": video.current_frame,

        "total_frames": video.total_frames,

        "fps": video.fps,

        "duration_seconds": video.duration_seconds,

        "width": video.width,

        "height": video.height,

        "output_video_path": video.output_video_path,

        "error_message": video.error_message,

        "started_at": video.started_at,

        "completed_at": video.completed_at,
    }

import cv2
import time
from fastapi.responses import StreamingResponse

@router.get("/{video_run_id}/stream-preview")
def stream_preview(
    video_run_id: int,
    db: Session = Depends(get_db),
):
    video = crud_video_run.get_video_run(db, video_run_id)
    if video is None or not video.input_path:
        raise HTTPException(status_code=404, detail="Video not found")

    def iter_frames():
        cap = cv2.VideoCapture(video.input_path)
        fps = video.fps or 30
        frame_time = 1.0 / fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for preview performance
            frame = cv2.resize(frame, (640, 360))
            
            success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                   
            time.sleep(frame_time)
            
        cap.release()

    return StreamingResponse(
        iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{video_run_id}/report")
def download_video_report(video_run_id: int, db: Session = Depends(get_db)):
    video = crud_video_run.get_video_run(db, video_run_id)

    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VideoRun with ID {video_run_id} not found",
        )

    if not video.excel_report_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not available yet — inference may still be running or "
                   "no inspection_log.xlsx was produced for this video.",
        )

    report_path = Path(video.excel_report_path)

    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file is missing on disk.",
        )

    filename = f"{Path(video.input_filename).stem}_inspection_log.xlsx"

    return FileResponse(
        path=str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


@router.patch("/{video_run_id}", response_model=VideoRunResponse)
def update_video_run(video_run_id: int, updates: VideoRunUpdate, db: Session = Depends(get_db)):
    db_video_run = crud_video_run.get_video_run(db, video_run_id)
    if not db_video_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VideoRun with ID {video_run_id} not found"
        )
    
    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_video_run, key, value)
        
    return crud_video_run.update_video_run(db, db_video_run)

@router.get("/{video_run_id}/result")
def get_video_result(
    video_run_id: int,
    db: Session = Depends(get_db),
):

    video = crud_video_run.get_video_run(
        db,
        video_run_id,
    )

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    cycles = crud_cycle.get_video_summary(
        db,
        video_run_id,
    )

    total_cycles = len(cycles)

    ok_cycles = sum(
        1
        for c in cycles
        if c.final_verdict == "OK"
    )

    anomaly_cycles = sum(
        1
        for c in cycles
        if c.final_verdict == "ANOMALY"
    )

    unknown_cycles = sum(
        1
        for c in cycles
        if c.final_verdict == "UNKNOWN"
    )

    return {

        "video": {

            "id": video.id,

            "filename": video.input_filename,

            "status": video.status,

            "progress": video.progress,

            "output_video": video.output_video_path,

            "duration_seconds": video.duration_seconds,

            "fps": video.fps,

            "total_frames": video.total_frames,

            "started_at": video.started_at,

            "completed_at": video.completed_at,
        },

        "summary": {

            "total_cycles": total_cycles,

            "ok_cycles": ok_cycles,

            "anomaly_cycles": anomaly_cycles,

            "unknown_cycles": unknown_cycles,
        },

        "cycles": [

            {

                "id": c.id,

                "cycle_number": c.cycle_number,

                "verdict": c.final_verdict,

                "sequence": c.detected_sequence,

                "tube_order": c.tube_order_result,

                "anomaly_ratio": c.anomaly_ratio,

                "output_video": c.output_video_path,

            }

            for c in cycles

        ]
    }
