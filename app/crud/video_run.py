from sqlalchemy.orm import Session

from app.models.video_run import VideoRun


def create_video_run(
    db: Session,
    *,
    batch_id: int,
    input_filename: str,
    input_path: str,
    queue_position: int,
    file_size: int = None,
    checksum: str = None,
    created_at: str,
) -> VideoRun:

    video = VideoRun(
        batch_id=batch_id,
        input_filename=input_filename,
        input_path=input_path,
        queue_position=queue_position,
        file_size=file_size,
        checksum=checksum,
        created_at=created_at,
        status="queued",
        progress=0,
        current_frame=0,
    )

    db.add(video)
    db.commit()
    db.refresh(video)

    return video


def get_video_run(
    db: Session,
    video_run_id: int,
) -> VideoRun | None:

    return (
        db.query(VideoRun)
        .filter(VideoRun.id == video_run_id)
        .first()
    )


def get_video_runs(
    db: Session,
) -> list[VideoRun]:

    return (
        db.query(VideoRun)
        .order_by(VideoRun.id)
        .all()
    )


def get_batch_video_runs(
    db: Session,
    batch_id: int,
) -> list[VideoRun]:

    return (
        db.query(VideoRun)
        .filter(VideoRun.batch_id == batch_id)
        .order_by(VideoRun.queue_position)
        .all()
    )


def update_video_run(
    db: Session,
    video: VideoRun,
) -> VideoRun:

    db.commit()
    db.refresh(video)

    return video
    
def get_video_progress(
    db: Session,
    video_run_id: int,
):
    return (
        db.query(VideoRun)
        .filter(VideoRun.id == video_run_id)
        .first()
    )

def delete_video_run(
    db: Session,
    video: VideoRun,
) -> None:

    db.delete(video)
    db.commit()