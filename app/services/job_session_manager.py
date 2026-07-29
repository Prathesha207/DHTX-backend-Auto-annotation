from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

@dataclass
class JobSession:
    batch_id: int
    status: str = "running"
    progress: float = 0.0
    current_video: int = 1
    total_videos: int = 1
    current_frame: int = 0
    fps: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    latest_frame: Optional[str] = None
    latest_terminal_logs: List[Dict[str, Any]] = field(default_factory=list)
    latest_statistics: Dict[str, Any] = field(default_factory=dict)
    current_cycle: Optional[int] = None
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    connected_clients: int = 0
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_update: float = field(default_factory=time.time)

    def to_snapshot_dict(self) -> Dict[str, Any]:
        return {
            "type": "batch_state",
            "status": self.status,
            "progress": self.progress,
            "current_video": self.current_video,
            "total_videos": self.total_videos,
            "current_frame": self.current_frame,
            "fps": round(self.fps, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1),
            "latest_frame": self.latest_frame,
            "latest_logs": self.latest_terminal_logs[-100:],  # send last 100 logs in snapshot
            "statistics": self.latest_statistics,
            "current_cycle": self.current_cycle,
            "cycles": self.cycles,
            "connected_clients": self.connected_clients,
            "started_at": self.started_at,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


class JobSessionManager:
    _instance = None

    def __init__(self):
        self.sessions: Dict[int, JobSession] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_session(self, batch_id: int, total_videos: int = 1) -> JobSession:
        session = JobSession(batch_id=batch_id, total_videos=total_videos)
        self.sessions[batch_id] = session
        print(f"[JobSession] JobSession Created for Batch {batch_id}")
        return session

    def get_session(self, batch_id: int) -> Optional[JobSession]:
        return self.sessions.get(batch_id)

    def destroy_session(self, batch_id: int):
        if batch_id in self.sessions:
            del self.sessions[batch_id]
            print(f"[JobSession] JobSession Destroyed for Batch {batch_id}")

    def update_client_count(self, batch_id: int, count: int):
        session = self.get_session(batch_id)
        if session:
            session.connected_clients = count
            session.last_update = time.time()

    def update_from_payload(self, batch_id: int, payload: Dict[str, Any]):
        session = self.get_session(batch_id)
        if not session:
            # If a payload is sent for a session that wasn't created yet, create it on the fly
            session = self.create_session(batch_id)

        session.last_update = time.time()
        msg_type = payload.get("type")

        if msg_type == "batch":
            status = payload.get("status")
            if status:
                if status == "cancelled":
                    status = "interrupted"
                session.status = status
                if status in ("completed", "failed", "interrupted"):
                    if status == "completed":
                        session.progress = 100.0
                    print(f"[JobSession] Job {status.capitalize()} for Batch {batch_id}")
                    # Note: we keep the session in memory so any reconnect immediately after completion gets the snapshot!
        elif msg_type == "progress":
            session.progress = payload.get("batch_progress", payload.get("progress", session.progress))
            session.current_frame = payload.get("current_frame", session.current_frame)
            session.fps = payload.get("fps", session.fps)
            session.elapsed_seconds = payload.get("elapsed_seconds", session.elapsed_seconds)
            session.eta_seconds = payload.get("eta_seconds", session.eta_seconds)
        elif msg_type == "frame":
            base64_img = payload.get("base64")
            if base64_img:
                session.latest_frame = base64_img
        elif msg_type == "log":
            log_item = {
                "level": payload.get("level", "info"),
                "message": payload.get("message", ""),
                "timestamp": payload.get("timestamp", datetime.utcnow().isoformat() + "Z")
            }
            session.latest_terminal_logs.append(log_item)
            if len(session.latest_terminal_logs) > 1000:
                session.latest_terminal_logs = session.latest_terminal_logs[-1000:]
        elif msg_type == "status":
            session.latest_statistics = payload
            if "cycle_no" in payload:
                session.current_cycle = payload.get("cycle_no")
        elif msg_type == "cycle":
            if "cycle" in payload:
                session.current_cycle = payload.get("cycle")
                cycle_num = payload.get("cycle")
                verdict = payload.get("verdict", "UNKNOWN")
                found = False
                for c in session.cycles:
                    if c.get("cycle") == cycle_num:
                        c["verdict"] = verdict
                        found = True
                        break
                if not found:
                    session.cycles.append({"cycle": cycle_num, "verdict": verdict})
        elif msg_type == "video_run":
            if "queue_position" in payload:
                session.current_video = payload.get("queue_position", session.current_video)
            if "total_videos" in payload:
                session.total_videos = payload.get("total_videos", session.total_videos)

    def get_snapshot_dict(self, batch_id: int, db: Any = None) -> Dict[str, Any]:
        session = self.get_session(batch_id)
        if session:
            return session.to_snapshot_dict()

        # If no active in-memory session exists, query SQLite for completed / failed / interrupted jobs
        if db is not None:
            from app.crud.batch import get_batch
            batch = get_batch(db, batch_id)
            if batch:
                cycles_list = []
                try:
                    from app.models.video_run import VideoRun
                    from app.models.cycle import Cycle
                    runs = db.query(VideoRun).filter(VideoRun.batch_id == batch_id).all()
                    for r in runs:
                        db_cycles = db.query(Cycle).filter(Cycle.video_run_id == r.id).all()
                        for c in db_cycles:
                            cycles_list.append({"cycle": c.cycle_number, "verdict": c.final_verdict})
                except Exception as e:
                    print(f"[JobSession] Error loading cycles for snapshot: {e}")

                status_val = "interrupted" if batch.status == "cancelled" else batch.status
                return {
                    "type": "batch_state",
                    "status": status_val,
                    "progress": 100.0 if status_val == "completed" else batch.progress,
                    "current_video": batch.completed_videos + 1,
                    "total_videos": batch.total_videos,
                    "current_frame": 0,
                    "fps": 0.0,
                    "elapsed_seconds": 0.0,
                    "eta_seconds": 0.0,
                    "latest_frame": None,
                    "latest_logs": [],
                    "statistics": {},
                    "current_cycle": cycles_list[-1]["cycle"] if cycles_list else None,
                    "cycles": cycles_list,
                    "connected_clients": 0,
                    "started_at": batch.started_at or "",
                    "output_path": batch.output_path,
                    "summary": f"Batch {batch.status}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }

        # Fallback if nothing found
        return {
            "type": "batch_state",
            "status": "unknown",
            "progress": 0.0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

job_session_mgr = JobSessionManager.get_instance()
