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

    # FIX: Track how many times update_progress has been called so we can
    # throttle the expensive batch-level DB query + commit (2 SELECTs + 1 COMMIT)
    # to fire only every 5 progress events instead of every single one.
    _progress_call_count: int = 0

    @staticmethod
    def update_progress(
        db: Session,
        video,
        *,
        current_frame: int,
        total_frames: int,
        fps: float = 0.0,
        elapsed_seconds: float = 0.0,
        eta_seconds: float = 0.0,
    ):

        if not video:
            return None

        video.current_frame = current_frame
        video.total_frames = total_frames

        if total_frames and total_frames > 0:
            video.progress = (current_frame / total_frames) * 100

        batch_progress = video.progress

        VideoRunService._progress_call_count += 1
        # FIX: Only do the expensive batch-level query every 5 calls (was every call).
        # This avoids 2 SQL SELECTs + 1 db.commit() on every progress update during inference.
        if VideoRunService._progress_call_count % 5 == 0 or current_frame == total_frames or current_frame == 1:
            try:
                from app.models.video_run import VideoRun
                from app.models.batch import Batch
                batch = db.query(Batch).filter(Batch.id == video.batch_id).first()
                if batch and batch.total_videos > 0:
                    all_runs = db.query(VideoRun).filter(VideoRun.batch_id == video.batch_id).all()
                    total_prog = 0.0
                    for r in all_runs:
                        if r.id == video.id:
                            total_prog += (video.progress or 0.0)
                        elif r.status == "completed":
                            total_prog += 100.0
                        else:
                            total_prog += (r.progress or 0.0)
                    batch_progress = min(max(total_prog / batch.total_videos, 0.0), 100.0)
                    batch.progress = batch_progress
                    db.commit()
            except Exception:
                pass

        # WebSocket emit fires every call — frontend stays responsive.
        manager.send_threadsafe(
            video.batch_id,
            {
                "type": "progress",
                "video_id": video.id,
                "progress": video.progress,
                "batch_progress": batch_progress,
                "current_frame": video.current_frame,
                "fps": fps,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            },
        )

        # FIX: Removed the redundant second db.commit() that previously fired every 5 frames
        # on top of the batch commit above. The batch commit above already covers the video row
        # since both objects share the same session and the video.progress was modified above.
        # Only force a video-level commit at the very first and last frame.
        if current_frame == 1 or current_frame == total_frames:
            return update_video_run(db, video)
        return video

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