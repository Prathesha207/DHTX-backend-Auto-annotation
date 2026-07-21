from datetime import datetime

from sqlalchemy.orm import Session

from app.crud.log import create_log
from app.services.websocket_manager import manager


class LogService:

    @staticmethod
    def _write(
        db: Session,
        level: str,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
    ):

        log = create_log(
            db=db,
            batch_id=batch_id,
            message=message,
            timestamp=datetime.now().isoformat(),
            level=level,
            video_run_id=video_run_id,
        )

        msg_str = f"[{level.upper()}] {message}"
        print(msg_str)

        manager.send_threadsafe(
            batch_id,
            {"type": "log", "level": level, "message": message},
        )

        return log

    @staticmethod
    def info(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
    ):
        return LogService._write(
            db=db,
            level="info",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
        )

    @staticmethod
    def warning(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
    ):
        return LogService._write(
            db=db,
            level="warning",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
        )

    @staticmethod
    def error(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
    ):
        return LogService._write(
            db=db,
            level="error",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
        )