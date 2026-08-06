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
    Body,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.dependencies import get_db

from app.services.upload_service import UploadService
from app.services.batch_service import BatchService
from app.services.video_run_service import VideoRunService
from app.crud.inference_config import get_config

from app.video_metadata import probe_video

from pydantic import BaseModel
import fastapi.responses

class BatchCreateRequest(BaseModel):
    batch_name: str | None = None
    total_videos: int

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v", ".webm", ".flv", ".3gp", ".ts",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"
}


class LocalFolderRequest(BaseModel):
    folder_path: str
    batch_name: str | None = None
    output_path: str | None = None


@router.post(
    "/video",
    status_code=status.HTTP_201_CREATED,
)
def upload_video(
    file: UploadFile = File(...),
    batch_name: str | None = Form(default=None),
    output_path: str | None = Form(default=None),
    db: Session = Depends(get_db),
):

    try:

        saved_info = UploadService.save_video(
            file=file,
            upload_root=UPLOAD_DIR,
        )
        saved_path = saved_info["path"]
        sanitized_name = saved_info["original_name"]

        metadata = probe_video(saved_path)

        output_dir = UploadService.create_batch_output_directory(
            output_path or get_config(db).default_output_dir,
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
            input_filename=sanitized_name,
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
            "batch_name": Path(batch.output_path).name if batch.output_path else None,
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
    output_path: str | None = Form(default=None),
    db: Session = Depends(get_db),
):

    if len(files) == 0:

        raise HTTPException(
            status_code=400,
            detail="No files uploaded.",
        )

    saved_files_info = UploadService.save_videos(
        files,
        UPLOAD_DIR,
    )

    output_dir = UploadService.create_batch_output_directory(
        output_path or get_config(db).default_output_dir,
    )

    batch = BatchService.create(
        db=db,
        batch_name=batch_name,
        input_type="folder",
        input_path=str(Path(saved_files_info[0]["path"]).parent),
        output_path=output_dir,
        total_videos=len(saved_files_info),
    )

    for index, file_info in enumerate(saved_files_info, start=1):

        path = file_info["path"]
        sanitized_name = file_info["original_name"]
        try:
            metadata = probe_video(path)
        except Exception:
            metadata = {"width": 0, "height": 0, "fps": 0, "duration_seconds": 0, "total_frames": 0}

        video = VideoRunService.create(
            db=db,
            batch_id=batch.id,
            input_filename=sanitized_name,
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

    from app.models.video_run import VideoRun
    ids = [vid for (vid,) in db.query(VideoRun.id).filter(VideoRun.batch_id == batch.id).order_by(VideoRun.queue_position.asc()).all()]

    return {
        "batch_id": batch.id,
        "batch_name": Path(batch.output_path).name if batch.output_path else None,
        "video_run_ids": ids,
        "message": "Folder uploaded successfully",
    }


@router.post(
    "/local-folder",
    status_code=status.HTTP_201_CREATED,
)
def upload_local_folder(
    request: LocalFolderRequest,
    db: Session = Depends(get_db),
):
    """
    Register videos from a local folder path (no file upload needed).
    Used by the Electron app where backend & frontend share the same machine.
    The folder_path must be an absolute path accessible to the backend process.
    """
    folder = Path(request.folder_path)

    if not folder.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Folder does not exist: {request.folder_path}",
        )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {request.folder_path}",
        )

    # Collect all supported video/image files (non-recursive to keep it predictable)
    video_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
    ])

    if len(video_files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No supported video or image files found in the selected folder.",
        )

    output_dir = UploadService.create_batch_output_directory(request.output_path or get_config(db).default_output_dir)

    batch = BatchService.create(
        db=db,
        batch_name=request.batch_name,
        input_type="folder",
        input_path=str(folder),
        output_path=output_dir,
        total_videos=len(video_files),
    )

    for index, video_path in enumerate(video_files, start=1):
        try:
            metadata = probe_video(str(video_path))
        except Exception:
            metadata = {"width": 0, "height": 0, "fps": 0, "duration_seconds": 0, "total_frames": 0}

        video = VideoRunService.create(
            db=db,
            batch_id=batch.id,
            input_filename=video_path.name,
            input_path=str(video_path),
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

    from app.models.video_run import VideoRun
    ids = [v for (v,) in db.query(VideoRun.id).filter(VideoRun.batch_id == batch.id).order_by(VideoRun.queue_position.asc()).all()]

    return {
        "batch_id": batch.id,
        "batch_name": Path(batch.output_path).name if batch.output_path else None,
        "video_run_ids": ids,
        "message": f"Local folder registered: {len(ids)} video(s) found",
    }


@router.post(
    "/local-video",
    status_code=status.HTTP_201_CREATED,
)
def upload_local_video(
    request: dict = Body(...),
    db: Session = Depends(get_db),
):
    """
    Register a single local video file by absolute path (no file upload needed).
    Used by the Electron app where backend & frontend share the same machine.
    """
    file_path_str = request.get("file_path")
    batch_name_val = request.get("batch_name")
    output_path_val = request.get("output_path")

    if not file_path_str:
        raise HTTPException(status_code=400, detail="file_path is required")

    file_path = Path(file_path_str)

    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File does not exist: {file_path_str}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path_str}")

    if file_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path.suffix}")

    try:
        metadata = probe_video(str(file_path))
    except Exception:
        metadata = {"width": 0, "height": 0, "fps": 0, "duration_seconds": 0, "total_frames": 0}

    output_dir = UploadService.create_batch_output_directory(output_path_val or get_config(db).default_output_dir)

    batch = BatchService.create(
        db=db,
        batch_name=batch_name_val,
        input_type="video",
        input_path=str(file_path),
        output_path=output_dir,
        total_videos=1,
    )

    video = VideoRunService.create(
        db=db,
        batch_id=batch.id,
        input_filename=file_path.name,
        input_path=str(file_path),
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
        "batch_name": Path(batch.output_path).name if batch.output_path else None,
        "video_run_id": video.id,
        "message": "Local video registered successfully",
    }


@router.post(
    "/batch/create",
    status_code=status.HTTP_201_CREATED,
)
def create_batch(
    request: BatchCreateRequest,
    db: Session = Depends(get_db),
):
    from app.crud.inference_config import get_config
    output_dir = UploadService.create_batch_output_directory(get_config(db).default_output_dir)

    batch = BatchService.create(
        db=db,
        batch_name=request.batch_name,
        input_type="folder",
        input_path="",
        output_path=output_dir,
        total_videos=request.total_videos,
    )
    return {"batch_id": batch.id}

@router.post(
    "/batch/{batch_id}/file",
    status_code=status.HTTP_201_CREATED,
)
def upload_batch_file(
    batch_id: int,
    queue_position: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    from app.models.batch import Batch
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    saved_info = UploadService.save_video(file=file, upload_root=UPLOAD_DIR)
    saved_path = saved_info["path"]
    sanitized_name = saved_info["original_name"]

    try:
        metadata = probe_video(saved_path)
    except Exception as e:
        video = VideoRunService.create(
            db=db,
            batch_id=batch.id,
            input_filename=sanitized_name,
            input_path=saved_path,
            queue_position=queue_position,
            file_size=saved_info.get("file_size"),
            checksum=saved_info.get("checksum"),
        )
        VideoRunService.fail(db, video, error_message=str(e), status="failed_upload")
        return fastapi.responses.JSONResponse(
            status_code=422,
            content={
                "accepted": False,
                "status": "FAILED_UPLOAD",
                "reason": str(e),
                "video_run_id": video.id
            }
        )

    video = VideoRunService.create(
        db=db,
        batch_id=batch.id,
        input_filename=sanitized_name,
        input_path=saved_path,
        queue_position=queue_position,
        file_size=saved_info.get("file_size"),
        checksum=saved_info.get("checksum"),
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
    
    if queue_position == 1 and not batch.input_path:
        batch.input_path = str(Path(saved_path).parent)
        
    db.commit()
    db.refresh(video)

    return {
        "batch_id": batch.id,
        "video_run_id": video.id,
        "queue_position": queue_position,
        "message": "Video attached to batch successfully",
    }


@router.get(
    "/batch/{batch_id}/manifest",
    status_code=status.HTTP_200_OK,
)
def get_batch_manifest(
    batch_id: int,
    db: Session = Depends(get_db),
):
    from app.models.batch import Batch
    from app.models.video_run import VideoRun
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    runs = db.query(VideoRun).filter(VideoRun.batch_id == batch_id).all()
    uploaded = [{"original_filename": r.input_filename, "status": r.status} for r in runs if r.status != "failed_upload"]
    
    return {
        "accepted": True,
        "batch_id": batch.id,
        "uploaded_files": uploaded
    }
