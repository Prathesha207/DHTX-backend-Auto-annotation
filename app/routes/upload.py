from pathlib import Path
from typing import List
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from app.dependencies import get_db

from app.services.upload_service import UploadService
from app.services.batch_service import BatchService
from app.services.video_run_service import VideoRunService

from app.video_metadata import probe_video

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"


@router.post(
    "/video",
    status_code=status.HTTP_201_CREATED,
)
def upload_video(
    file: UploadFile = File(...),
    batch_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):

    try:

        saved_path = UploadService.save_video(
            file=file,
            upload_root=UPLOAD_DIR,
        )

        metadata = probe_video(saved_path)

        output_dir = UploadService.create_output_directory(
            OUTPUT_DIR,
        )

        batch = BatchService.create(
            db=db,
            batch_name=batch_name,
            input_type="video",
            input_path=saved_path,
            output_path=output_dir,
            total_videos=1,
        )

        video = VideoRunService.create(
            db=db,
            batch_id=batch.id,
            input_filename=Path(saved_path).name,
            input_path=saved_path,
            queue_position=1,
        )

        VideoRunService.update_metadata(
            db=db,
            video=video,
            width=metadata["width"],
            height=metadata["height"],
            fps=metadata["fps"],
            duration_seconds=metadata["duration_seconds"],
        )

        video.total_frames = metadata["total_frames"]

        db.commit()
        db.refresh(video)

        return {
            "batch_id": batch.id,
            "video_run_id": video.id,
            "message": "Video uploaded successfully",
        }

    except Exception as ex:

        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )


@router.post(
    "/folder",
    status_code=status.HTTP_201_CREATED,
)
def upload_folder(
    files: List[UploadFile] = File(...),
    batch_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):

    if len(files) == 0:

        raise HTTPException(
            status_code=400,
            detail="No files uploaded.",
        )

    saved_files = UploadService.save_videos(
        files,
        UPLOAD_DIR,
    )

    output_dir = UploadService.create_output_directory(
        OUTPUT_DIR,
    )

    batch = BatchService.create(
        db=db,
        batch_name=batch_name,
        input_type="folder",
        input_path=str(Path(saved_files[0]).parent),
        output_path=output_dir,
        total_videos=len(saved_files),
    )

    ids = []

    for index, path in enumerate(saved_files, start=1):

        metadata = probe_video(path)

        video = VideoRunService.create(
            db=db,
            batch_id=batch.id,
            input_filename=Path(path).name,
            input_path=path,
            queue_position=index,
        )

        VideoRunService.update_metadata(
            db=db,
            video=video,
            width=metadata["width"],
            height=metadata["height"],
            fps=metadata["fps"],
            duration_seconds=metadata["duration_seconds"],
        )

        video.total_frames = metadata["total_frames"]

        db.commit()
        db.refresh(video)

        ids.append(video.id)

    return {
        "batch_id": batch.id,
        "video_run_ids": ids,
        "message": "Folder uploaded successfully",
    }