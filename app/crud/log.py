from sqlalchemy.orm import Session

from app.models.log import Log


def create_log(
    db: Session,
    batch_id: int,
    message: str,
    timestamp: str,
    level: str = "info",
    video_run_id: int | None = None,
    cycle_id: int | None = None,
    frame_number: int | None = None,
    state: str | None = None,
) -> Log:

    log = Log(
        batch_id=batch_id,
        video_run_id=video_run_id,
        cycle_id=cycle_id,
        frame_number=frame_number,
        state=state,
        timestamp=timestamp,
        level=level,
        message=message,
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log


def create_log_no_commit(
    db: Session,
    batch_id: int,
    message: str,
    timestamp: str,
    level: str = "info",
    video_run_id: int | None = None,
    cycle_id: int | None = None,
    frame_number: int | None = None,
    state: str | None = None,
) -> Log:
    """
    Add a log row to the session WITHOUT committing.
    Used by the buffered flush path for debug/perf logs.
    """
    log = Log(
        batch_id=batch_id,
        video_run_id=video_run_id,
        cycle_id=cycle_id,
        frame_number=frame_number,
        state=state,
        timestamp=timestamp,
        level=level,
        message=message,
    )
    db.add(log)
    return log


def get_log(
    db: Session,
    log_id: int,
) -> Log | None:

    return (
        db.query(Log)
        .filter(Log.id == log_id)
        .first()
    )

def get_logs(
    db: Session,
):

    return (
        db.query(Log)
        .order_by(Log.id)
        .all()
    )
    
def get_batch_logs(
    db: Session,
    batch_id: int,
):

    return (
        db.query(Log)
        .filter(Log.batch_id == batch_id)
        .order_by(Log.id)
        .all()
    )


def get_video_logs(
    db: Session,
    video_run_id: int,
):

    return (
        db.query(Log)
        .filter(Log.video_run_id == video_run_id)
        .order_by(Log.id)
        .all()
    )


def delete_log(
    db: Session,
    log: Log,
):

    db.delete(log)
    db.commit()