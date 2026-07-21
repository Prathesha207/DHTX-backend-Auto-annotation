from datetime import datetime

from app.services.websocket_manager import manager

from sqlalchemy.orm import Session

from app.crud.batch import (
    create_batch,
    update_batch,
)


class BatchService:

    @staticmethod
    def create(
        db: Session,
        *,
        batch_name: str | None,
        input_type: str,
        input_path: str,
        output_path: str,
        total_videos: int,
    ):

        return create_batch(
            db=db,
            batch_name=batch_name,
            input_type=input_type,
            input_path=input_path,
            output_path=output_path,
            total_videos=total_videos,
            created_at=datetime.now().isoformat(),
        )

    @staticmethod
    def start(
        db: Session,
        batch,
    ):

        batch.status = "running"
        batch.started_at = datetime.now().isoformat()

        manager.send_threadsafe(batch.id, {"type": "batch", "status": "running"})

        return update_batch(db, batch)

    @staticmethod
    def update_progress(
        db: Session,
        batch,
        *,
        completed_videos: int,
        failed_videos: int,
    ):

        batch.completed_videos = completed_videos
        batch.failed_videos = failed_videos

        if batch.total_videos > 0:
            batch.progress = (
                (completed_videos + failed_videos)
                / batch.total_videos
            ) * 100

        return update_batch(db, batch)

    @staticmethod
    def complete(
        db: Session,
        batch,
    ):

        batch.status = "completed"
        batch.progress = 100
        batch.completed_at = datetime.now().isoformat()

        manager.send_threadsafe(batch.id, {"type": "batch", "status": "completed"})
        manager.send_threadsafe(batch.id, {"type": "finished"})

        return update_batch(db, batch)

    @staticmethod
    def fail(
        db: Session,
        batch,
    ):

        batch.status = "failed"
        batch.completed_at = datetime.now().isoformat()

        manager.send_threadsafe(batch.id, {"type": "batch", "status": "failed"})
        manager.send_threadsafe(batch.id, {"type": "finished"})

        return update_batch(db, batch)