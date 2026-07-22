from datetime import datetime

from sqlalchemy.orm import Session

from app.models.inference_config import InferenceConfig


_DEFAULTS = dict(
    model1_frame_count=30,
    model1_pass_frames=28,
    model2_start_skip_frame=10,
    model2_frame_count=20,
    model2_pass_frames=18,
    socket_absent_frames=10,
    socket_loss_abort_frames=15,
    enable_debug_logging=False,
    enable_perf_logging=True,
)


def get_config(db: Session) -> InferenceConfig:
    """
    Return the singleton inference config row (id=1).
    Creates one with defaults if it doesn't exist yet.
    """
    row = db.query(InferenceConfig).filter(InferenceConfig.id == 1).first()
    if row is None:
        row = InferenceConfig(
            id=1,
            **_DEFAULTS,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def update_config(db: Session, **kwargs) -> InferenceConfig:
    """
    Partial update of the singleton config row.
    Caller is responsible for validation (see route layer).
    """
    row = get_config(db)
    for key, value in kwargs.items():
        if hasattr(row, key) and key not in ("id", "created_at"):
            setattr(row, key, value)
    row.updated_at = datetime.now().isoformat()
    db.commit()
    db.refresh(row)
    return row
