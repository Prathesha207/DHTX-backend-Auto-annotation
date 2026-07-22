import threading
import time
from datetime import datetime

from sqlalchemy.orm import Session

from app.crud.log import create_log, create_log_no_commit
from app.services.websocket_manager import manager


# ============================================================
# Buffered Flush for debug/perf Logs
# ============================================================
# debug and perf logs are high-frequency. Writing each one to
# SQLite with an individual commit would bottleneck the pipeline.
# They are accumulated in memory and flushed every _FLUSH_INTERVAL
# seconds or every _FLUSH_BATCH_SIZE entries, whichever fires first.
# Console print and WebSocket send are ALWAYS immediate.
# ============================================================

_FLUSH_BATCH_SIZE = 50
_FLUSH_INTERVAL = 2.0  # seconds

_buffer_lock = threading.Lock()
_pending_logs: list[dict] = []
_flush_timer: threading.Timer | None = None


def _schedule_flush():
    global _flush_timer
    if _flush_timer is not None:
        return  # already scheduled
    _flush_timer = threading.Timer(_FLUSH_INTERVAL, _do_flush)
    _flush_timer.daemon = True
    _flush_timer.start()


def _do_flush():
    global _flush_timer
    with _buffer_lock:
        batch = list(_pending_logs)
        _pending_logs.clear()
        _flush_timer = None

    if not batch:
        return

    try:
        from app.database.database import SessionLocal
        with SessionLocal() as db:
            for entry in batch:
                create_log_no_commit(
                    db=db,
                    batch_id=entry["batch_id"],
                    message=entry["message"],
                    timestamp=entry["timestamp"],
                    level=entry["level"],
                    video_run_id=entry.get("video_run_id"),
                    cycle_id=entry.get("cycle_id"),
                    frame_number=entry.get("frame_number"),
                    state=entry.get("state"),
                )
            db.commit()
    except Exception:
        pass  # best-effort; logs are already on console + WS


def _buffer_log(entry: dict):
    with _buffer_lock:
        _pending_logs.append(entry)
        if len(_pending_logs) >= _FLUSH_BATCH_SIZE:
            _schedule_flush_now = True
        else:
            _schedule_flush_now = False
            _schedule_flush()

    if _schedule_flush_now:
        _do_flush()


def flush_pending_logs():
    """
    Flush any buffered debug/perf logs immediately.
    Call this at video-run or batch boundaries to ensure nothing is lost.
    """
    _do_flush()


# ============================================================
# LogService — the SINGLE logging pipeline
# ============================================================
# Every log in the inference path goes through here.
# Nothing else in the backend calls print() for inference events.
# ============================================================

class LogService:

    @staticmethod
    def _write(
        db: Session | None,
        level: str,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        ts = datetime.now().isoformat()

        # ── 1. Console (the ONLY print() in the pipeline) ────────
        prefix = f"[{level.upper()}]"
        if frame_number is not None:
            prefix += f" [F#{frame_number:05d}]"
        if state:
            prefix += f" [{state}]"
        print(f"{prefix} {message}")

        # ── 2. WebSocket — ALWAYS immediate, every level ─────────
        ws_payload = {"type": "log", "level": level, "message": message}
        if cycle_id is not None:
            ws_payload["cycle_id"] = cycle_id
        if frame_number is not None:
            ws_payload["frame_number"] = frame_number
        if state:
            ws_payload["state"] = state

        manager.send_threadsafe(batch_id, ws_payload)

        # ── 3. SQLite ────────────────────────────────────────────
        if level in ("debug", "perf"):
            # Buffered write for high-frequency levels
            _buffer_log(dict(
                batch_id=batch_id,
                message=message,
                timestamp=ts,
                level=level,
                video_run_id=video_run_id,
                cycle_id=cycle_id,
                frame_number=frame_number,
                state=state,
            ))
            return None
        else:
            # Immediate write for info/warning/error
            if db is not None:
                log = create_log(
                    db=db,
                    batch_id=batch_id,
                    message=message,
                    timestamp=ts,
                    level=level,
                    video_run_id=video_run_id,
                    cycle_id=cycle_id,
                    frame_number=frame_number,
                    state=state,
                )
                return log
            else:
                # Caller didn't provide a session — buffer it
                _buffer_log(dict(
                    batch_id=batch_id,
                    message=message,
                    timestamp=ts,
                    level=level,
                    video_run_id=video_run_id,
                    cycle_id=cycle_id,
                    frame_number=frame_number,
                    state=state,
                ))
                return None

    @staticmethod
    def info(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        return LogService._write(
            db=db,
            level="info",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
            cycle_id=cycle_id,
            frame_number=frame_number,
            state=state,
        )

    @staticmethod
    def warning(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        return LogService._write(
            db=db,
            level="warning",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
            cycle_id=cycle_id,
            frame_number=frame_number,
            state=state,
        )

    @staticmethod
    def error(
        db: Session,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        return LogService._write(
            db=db,
            level="error",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
            cycle_id=cycle_id,
            frame_number=frame_number,
            state=state,
        )

    @staticmethod
    def debug(
        db: Session | None,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        return LogService._write(
            db=db,
            level="debug",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
            cycle_id=cycle_id,
            frame_number=frame_number,
            state=state,
        )

    @staticmethod
    def perf(
        db: Session | None,
        batch_id: int,
        message: str,
        video_run_id: int | None = None,
        cycle_id: int | None = None,
        frame_number: int | None = None,
        state: str | None = None,
    ):
        return LogService._write(
            db=db,
            level="perf",
            batch_id=batch_id,
            message=message,
            video_run_id=video_run_id,
            cycle_id=cycle_id,
            frame_number=frame_number,
            state=state,
        )