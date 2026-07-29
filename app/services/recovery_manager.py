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
            # ── Reset unfinished batches so they can be resumed ──
            unfinished_batches = db.query(Batch).filter(
                Batch.status.in_(["running", "queued", "pending"])
            ).all()
            for b in unfinished_batches:
                old_status = b.status
                b.status = "queued"
                print(f"[RECOVERY] Batch {b.id}: {old_status} -> queued (ready to resume)")
                # Update in-memory session for UI sync
                session = job_session_mgr.create_session(b.id, b.total_videos)
                session.status = "queued"
                session.progress = b.progress

            # ── Reset unfinished video runs so they will be retried ──
            unfinished_runs = db.query(VideoRun).filter(
                VideoRun.status.in_(["running", "queued", "pending"])
            ).all()
            for r in unfinished_runs:
                r.status = "pending"
                print(f"[RECOVERY] VideoRun {r.id}: -> pending (ready to retry)")

            if unfinished_batches or unfinished_runs:
                db.commit()
                print(f"[RECOVERY] Done: {len(unfinished_batches)} batches and {len(unfinished_runs)} video runs recovered for resume.")
            else:
                print("[RECOVERY] No orphaned jobs found.")
        except Exception as e:
            print(f"[RECOVERY ERROR] Error during startup recovery: {e}")
            db.rollback()
