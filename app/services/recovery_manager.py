from sqlalchemy.orm import Session
from app.models.batch import Batch
from app.models.video_run import VideoRun
from app.services.job_session_manager import job_session_mgr

class RecoveryManager:
    @staticmethod
    def recover_orphaned_jobs(db: Session):
        """
        On every startup, cancel all orphaned/unfinished jobs from the
        previous session so the queue is completely empty.
        This ensures that when a user uploads a new batch after restarting
        the server, it runs immediately instead of waiting behind crashed
        or interrupted jobs.

        DB status values are constrained to:
            queued | running | completed | cancelled | failed

        Recovery policy
        ───────────────
        - ANY batch that is 'running', 'queued', or 'pending' is marked 'queued'.
        - ANY video run that is 'running', 'queued', or 'pending' is marked 'pending'.
        """
        try:
            # ── Clean up orphaned processing video files ──
            import os
            from pathlib import Path
            output_dir = Path("outputs")
            if output_dir.exists():
                for p in output_dir.rglob("__processing__*"):
                    try:
                        p.unlink()
                        print(f"[RECOVERY] Deleted orphaned temp video: {p.name}")
                    except Exception as e:
                        print(f"[RECOVERY ERROR] Could not delete {p}: {e}")

            # ── Cancel unfinished batches so the queue is empty ──
            unfinished_batches = db.query(Batch).filter(
                Batch.status.in_(["running", "queued", "pending", "interrupted"])
            ).all()
            for b in unfinished_batches:
                old_status = b.status
                b.status = "interrupted"
                b.interrupted_reason = "BACKEND_CRASH"
                print(f"[RECOVERY] Batch {b.id}: {old_status} -> interrupted")
                
                # Load persisted state if available
                from app.models.job_session_state import JobSessionState
                state_obj = db.query(JobSessionState).filter(JobSessionState.batch_id == b.id).first()
                session = job_session_mgr.create_session(b.id, b.total_videos)
                
                if state_obj and state_obj.state_json:
                    state = state_obj.state_json
                    session.progress = state.get("progress", b.progress)
                    session.batch_progress = state.get("batch_progress", b.progress)
                    session.video_progress = state.get("video_progress", 0)
                    session.current_video = state.get("current_video", 1)
                    session.completed_videos = state.get("completed_videos", 0)
                    session.last_completed_video_id = state.get("last_completed_video_id")
                    session.last_completed_queue_position = state.get("last_completed_queue_position")
                    session.current_filename = state.get("current_filename")
                    session.current_frame = state.get("current_frame", 0)
                    session.total_frames = state.get("total_frames", 0)
                    session.latest_statistics = state.get("statistics", {})
                    
                session.status = "interrupted"

            # ── Cancel unfinished video runs ──
            unfinished_runs = db.query(VideoRun).filter(
                VideoRun.status.in_(["running", "queued", "pending", "interrupted"])
            ).all()
            for r in unfinished_runs:
                r.status = "interrupted"
                r.interrupted_reason = "BACKEND_CRASH"
                print(f"[RECOVERY] VideoRun {r.id}: -> interrupted")

            if unfinished_batches or unfinished_runs:
                db.commit()
                print(f"[RECOVERY] Done: {len(unfinished_batches)} batches and {len(unfinished_runs)} video runs interrupted.")
            else:
                print("[RECOVERY] No orphaned jobs found.")
        except Exception as e:
            print(f"[RECOVERY ERROR] Error during startup recovery: {e}")
            db.rollback()
