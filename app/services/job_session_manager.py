from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

import uuid

@dataclass
class JobSession:
    batch_id: int
    session_uuid: str = field(default_factory=lambda: uuid.uuid4().hex)
    ws_sequence_number: int = 0
    status: str = "running"
    progress: float = 0.0
    current_video: int = 1
    current_video_run_id: Optional[int] = None
    total_videos: int = 1
    completed_videos: int = 0
    last_completed_video_id: Optional[int] = None
    last_completed_queue_position: Optional[int] = None
    current_filename: Optional[str] = None
    current_frame: int = 0
    total_frames: int = 0
    batch_progress: float = 0.0
    video_progress: float = 0.0
    fps: float = 0.0
    decode_fps: float = 0.0
    processing_fps: float = 0.0
    streaming_fps: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    decoded_frame: int = 0
    processed_frame: int = 0
    rendered_frame: int = 0
    sent_frame: int = 0
    heartbeat: float = field(default_factory=time.time)
    latest_frame: Optional[str] = None
    latest_terminal_logs: List[Dict[str, Any]] = field(default_factory=list)
    latest_statistics: Dict[str, Any] = field(default_factory=dict)
    current_cycle: Optional[int] = None
    current_verdict: Optional[str] = None
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    connected_clients: int = 0
    show_roi: bool = False
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_update: float = field(default_factory=time.time)

    def to_snapshot_dict(self) -> Dict[str, Any]:
        return {
            "type": "batch_state",
            "session_uuid": self.session_uuid,
            "status": self.status,
            "progress": self.progress, # Deprecated, use batch_progress
            "batch_progress": self.batch_progress,
            "video_progress": self.video_progress,
            "current_video": self.current_video,
            "current_video_run_id": self.current_video_run_id,
            "total_videos": self.total_videos,
            "completed_videos": self.completed_videos,
            "last_completed_video_id": self.last_completed_video_id,
            "last_completed_queue_position": self.last_completed_queue_position,
            "current_filename": self.current_filename,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 1),
            "decode_fps": round(self.decode_fps, 1),
            "processing_fps": round(self.processing_fps, 1),
            "streaming_fps": round(self.streaming_fps, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1),
            "decoded_frame": self.decoded_frame,
            "processed_frame": self.processed_frame,
            "rendered_frame": self.rendered_frame,
            "sent_frame": self.sent_frame,
            "heartbeat": self.heartbeat,
            "latest_frame": self.latest_frame,
            "latest_logs": self.latest_terminal_logs[-100:],  # send last 100 logs in snapshot
            "statistics": self.latest_statistics,
            "current_cycle": self.current_cycle,
            "current_verdict": self.current_verdict,
            "cycles": self.cycles,
            "connected_clients": self.connected_clients,
            "show_roi": self.show_roi,
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
        session = self.sessions.get(batch_id)
        if session:
            session.connected_clients = count
            session.last_update = time.time()

    def set_show_roi(self, batch_id: int, show_roi: bool) -> None:
        if batch_id in self.sessions:
            self.sessions[batch_id].show_roi = show_roi

    def update_from_payload(self, batch_id: int, payload: Dict[str, Any]):
        session = self.get_session(batch_id)
        if not session:
            # If a payload is sent for a session that wasn't created yet, create it on the fly
            session = self.create_session(batch_id)

        msg_type = payload.get("type")

        if msg_type == "batch":
            status = payload.get("status")
            if status:
                if status == "cancelled":
                    status = "interrupted"
                session.status = status
                if status in ("completed", "failed", "interrupted"):
                    session.progress = 100.0
                    session.batch_progress = 100.0
                    session.video_progress = 100.0
            
            if "videos_queued" in payload: session.videos_queued = payload["videos_queued"]
            if "videos_processing" in payload: session.videos_processing = payload["videos_processing"]
            if "videos_completed" in payload: 
                session.completed_videos = payload["videos_completed"]
                session.videos_completed = payload["videos_completed"]
                if session.total_videos > 0:
                    session.batch_progress = (session.completed_videos / session.total_videos) * 100
            if "videos_failed" in payload: session.videos_failed = payload["videos_failed"]
            if "videos_cancelled" in payload: session.videos_cancelled = payload["videos_cancelled"]
            
            if status in ("completed", "failed", "interrupted"):
                print(f"[JobSession] Job {status.capitalize()} for Batch {batch_id}")
                # Note: we keep the session in memory so any reconnect immediately after completion gets the snapshot!
        elif msg_type == "video_started":
            session.current_filename = payload.get("filename", session.current_filename)
            session.total_frames = payload.get("total_frames", session.total_frames)
            session.current_video = payload.get("video_index", session.current_video)
            session.current_video_run_id = payload.get("video_id", session.current_video_run_id)
            session.total_videos = payload.get("total_videos", session.total_videos)
            session.fps = payload.get("fps", session.fps)
            session.current_frame = 0
            session.video_progress = 0.0
            
            # 1. Hard reset transient session states (USER REQUIREMENT)
            session.cycles = []
            session.current_cycle = None
            session.current_verdict = None
            session.latest_statistics = {}
            session.latest_frame = None
            session.latest_terminal_logs = []
            session.elapsed_seconds = 0.0
            session.eta_seconds = 0.0
            
            if session.total_videos > 0:
                session.batch_progress = (session.completed_videos / session.total_videos) * 100
        elif msg_type == "progress":
            if "current_frame" in payload: session.current_frame = payload["current_frame"]
            if "total_frames" in payload: session.total_frames = payload["total_frames"]
            if "fps" in payload: session.fps = payload["fps"]
            if "elapsed_seconds" in payload: session.elapsed_seconds = payload["elapsed_seconds"]
            if "eta_seconds" in payload: session.eta_seconds = payload["eta_seconds"]
            if "progress" in payload: session.video_progress = payload["progress"]
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
            if "final_verdict" in payload:
                session.current_verdict = payload.get("final_verdict")
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
                    "session_uuid": "",
                    "sequence": 0,
                    "progress": 100.0 if status_val == "completed" else batch.progress, # Deprecated
                    "batch_progress": 100.0 if status_val == "completed" else batch.progress,
                    "video_progress": 100.0 if status_val == "completed" else 0.0,
                    "current_video": batch.completed_videos + 1,
                    "total_videos": batch.total_videos,
                    "completed_videos": batch.completed_videos,
                    "current_filename": None,
                    "current_frame": 0,
                    "total_frames": 0,
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
