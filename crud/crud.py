import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from database import models


# ============================================================
# STORAGE
# ============================================================

def get_active_storage(db: Session):
    from utils.storage_discovery import is_path_writable, get_writable_default_location

    storage = (
        db.query(models.StorageSetting)
        .filter(models.StorageSetting.is_active.is_(True))
        .first()
    )

    if storage and is_path_writable(storage.root_path):
        return storage

    # Auto-heal: If no storage exists or the stored path was from a different machine
    default_path = get_writable_default_location()
    if storage:
        storage.root_path = default_path
        storage.is_active = True
        db.commit()
        db.refresh(storage)
        return storage

    new_setting = models.StorageSetting(
        root_path=default_path,
        is_active=True
    )
    db.add(new_setting)
    db.commit()
    db.refresh(new_setting)
    return new_setting



def set_active_storage(db: Session, new_root_path: str):
    # Deactivate current storage
    (
        db.query(models.StorageSetting)
        .filter(models.StorageSetting.is_active.is_(True))
        .update(
            {"is_active": False},
            synchronize_session=False
        )
    )

    # Create new active storage
    new_setting = models.StorageSetting(
        root_path=new_root_path,
        is_active=True
    )

    db.add(new_setting)
    db.commit()
    db.refresh(new_setting)

    return new_setting


import threading

_batch_lock = threading.Lock()

# ============================================================
# BATCH
# ============================================================

def create_batch(
    db: Session,
    storage_id: int,
    date_str: str
) -> models.Batch:

    # --------------------------------------------------------
    # IMPORTANT:
    # Batch numbering comes from FILESYSTEM.
    # --------------------------------------------------------
    import pathlib
    
    active_storage = db.query(models.StorageSetting).filter(models.StorageSetting.id == storage_id).first()
    if not active_storage:
        raise Exception("Active storage not found")
        
    base_raw_dir = pathlib.Path(active_storage.root_path) / "raw" / date_str
    base_processed_dir = pathlib.Path(active_storage.root_path) / "processed" / date_str
    
    existing = set()
    for d in (base_raw_dir, base_processed_dir):
        if d.exists():
            for item in d.iterdir():
                if item.is_dir() and item.name.startswith("batch_"):
                    try:
                        existing.add(int(item.name.replace("batch_", "")))
                    except ValueError:
                        pass

    # Find the first missing batch number (reuse gaps)
    with _batch_lock:
        next_num = 1
        while next_num in existing:
            next_num += 1

        # Create DB record
        db_batch = models.Batch(
            id=str(uuid.uuid4()),
            batch_number=next_num,
            batch_date=date_str,
            storage_root_id=storage_id
        )

        db.add(db_batch)
        db.commit()
        db.refresh(db_batch)

    return db_batch


# ============================================================
# VIDEO
# ============================================================

def create_video(
    db: Session,
    video_id: str,
    batch_id: str,
    filename: str,
    source_path: str,
    file_size: int,
    relative_path: Optional[str] = None,
    last_modified: Optional[int] = None,
    commit: bool = True
):
    db_video = models.Video(
        id=video_id,
        batch_id=batch_id,
        filename=filename,
        source_path=source_path,
        file_size=file_size,
        relative_path=relative_path,
        last_modified=last_modified,
        status="QUEUED"
    )

    db.add(db_video)
    if commit:
        db.commit()
        db.refresh(db_video)

    return db_video


def get_videos(
    db: Session,
    skip: int = 0,
    limit: int = 100
):
    return (
        db.query(models.Video)
        .order_by(models.Video.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_video_result(
    db: Session,
    video_id: str,
    status: str,
    verdict: str = None,
    output_path: str = None,
    commit: bool = True
):
    video = (
        db.query(models.Video)
        .filter(models.Video.id == video_id)
        .first()
    )

    if video:
        video.status = status

        if verdict is not None:
            video.verdict = verdict

        if output_path is not None:
            video.output_path = output_path

        if commit:
            db.commit()
            db.refresh(video)

    return video


# ============================================================
# MODEL SETTINGS
# ============================================================

def get_active_model_setting(db: Session):

    setting = (
        db.query(models.ModelSetting)
        .filter(models.ModelSetting.is_active.is_(True))
        .first()
    )

    if not setting:
        setting = models.ModelSetting(
            is_active=True
        )

        db.add(setting)
        db.commit()
        db.refresh(setting)

    return setting


def update_model_setting(
    db: Session,
    m1_total: int,
    m1_pass: int,
    m2_total: int,
    m2_pass: int
):
    if not (1 <= m1_pass <= m1_total):
        raise ValueError("Model 1 pass_frames must be between 1 and total_frames")
    if not (1 <= m2_pass <= m2_total):
        raise ValueError("Model 2 pass_frames must be between 1 and total_frames")

    setting = get_active_model_setting(db)

    setting.m1_total = m1_total
    setting.m1_pass = m1_pass
    setting.m2_total = m2_total
    setting.m2_pass = m2_pass

    db.commit()
    db.refresh(setting)

    return setting
