from sqlalchemy.orm import Session

from app.models.batch import Batch


def create_batch(
    db: Session,
    *,
    batch_name: str | None = None,
    input_type: str,
    input_path: str,
    output_path: str,
    total_videos: int,
    created_at: str,
) -> Batch:

    batch = Batch(
        batch_name=batch_name,
        input_type=input_type,
        input_path=input_path,
        output_path=output_path,
        total_videos=total_videos,
        completed_videos=0,
        failed_videos=0,
        progress=0,
        status="queued",
        created_at=created_at,
    )

    db.add(batch)
    db.commit()
    db.refresh(batch)

    return batch


def get_batch(
    db: Session,
    batch_id: int,
) -> Batch | None:

    return (
        db.query(Batch)
        .filter(Batch.id == batch_id)
        .first()
    )


def get_batches(
    db: Session,
) -> list[Batch]:

    return (
        db.query(Batch)
        .order_by(Batch.id.desc())
        .all()
    )


def get_batch_status(
    db: Session,
    batch_id: int,
):
    return (
        db.query(Batch)
        .filter(Batch.id == batch_id)
        .first()
    )


def update_batch(
    db: Session,
    batch: Batch,
) -> Batch:

    db.commit()
    db.refresh(batch)

    return batch


def delete_batch(
    db: Session,
    batch: Batch,
):

    db.delete(batch)
    db.commit()