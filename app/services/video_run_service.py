from datetime import datetime

from sqlalchemy.orm import Session

from app.services.websocket_manager import manager

from app.crud.video_run import (
    create_video_run,
    update_video_run,
)


class VideoRunService:

    @staticmethod
    def create(
        db: Session,
        *,
        batch_id: int,
        input_filename: str,
        input_path: str,
        queue_position: int,
    ):

        return create_video_run(
            db=db,
            batch_id=batch_id,
            input_filename=input_filename,
            input_path=input_path,
            queue_position=queue_position,
            created_at=datetime.now().isoformat(),
        )

    @staticmethod
    def start(
        db: Session,
        video,
    ):

        video.status = "running"
        video.started_at = datetime.now().isoformat()

        return update_video_run(db, video)

    @staticmethod
    def update_progress(
        db: Session,
        video,
        *,
        current_frame: int,
        total_frames: int,
    ):

        video.current_frame = current_frame
        video.total_frames = total_frames

        if total_frames and total_frames > 0:
            video.progress = (current_frame / total_frames) * 100

        manager.send_threadsafe(
            video.batch_id,
            {
                "type": "progress",
                "video_id": video.id,
                "progress": video.progress,
                "current_frame": video.current_frame,
            },
        )

        return update_video_run(db, video)

    @staticmethod
    def update_metadata(
        db: Session,
        video,
        *,
        width: int,
        height: int,
        fps: float,
        duration_seconds: float,
    ):

        video.width = width
        video.height = height
        video.fps = fps
        video.duration_seconds = duration_seconds

        return update_video_run(db, video)

    @staticmethod
    def complete(
        db: Session,
        video,
        *,
        output_video_path: str,
        excel_report_path: str | None = None,
    ):

        video.status = "completed"
        video.progress = 100
        video.output_video_path = output_video_path
        video.excel_report_path = excel_report_path
        video.completed_at = datetime.now().isoformat()

        return update_video_run(db, video)

    @staticmethod
    def fail(
        db: Session,
        video,
        *,
        error_message: str,
    ):

        video.status = "failed"
        video.error_message = error_message
        video.completed_at = datetime.now().isoformat()

        return update_video_run(db, video)