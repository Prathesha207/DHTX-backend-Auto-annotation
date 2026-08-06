from sqlalchemy.orm import Session
import logging

logger = logging.getLogger("DBVerifier")

class DBVerifier:
    def __init__(self, db_session: Session):
        self.db = db_session

    def verify_batch(self, batch_id: int):
        from app.models.batch import Batch
        from app.models.video_run import VideoRun
        
        batch = self.db.query(Batch).filter(Batch.id == batch_id).first()
        if not batch:
            logger.error(f"Batch {batch_id} not found in DB.")
            return False
            
        videos = self.db.query(VideoRun).filter(VideoRun.batch_id == batch_id).all()
        
        status_counts = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed_upload": 0,
            "failed_inference": 0,
            "interrupted": 0,
            "cancelled": 0
        }
        
        for v in videos:
            if v.status in status_counts:
                status_counts[v.status] += 1
            else:
                logger.warning(f"Unknown status '{v.status}' on VideoRun {v.id}")
                
        # Assertion 1: Sum of statuses == total_videos
        total_calculated = sum(status_counts.values())
        if total_calculated != batch.total_videos:
            logger.error(f"Assertion failed: sum of statuses ({total_calculated}) != total_videos ({batch.total_videos})")
            return False
            
        # Assertion 2: Batch.completed_videos == actual completed count
        if batch.completed_videos != status_counts["completed"]:
            logger.error(f"Assertion failed: Batch.completed_videos ({batch.completed_videos}) != actual completed count ({status_counts['completed']})")
            return False
            
        logger.info(f"Batch {batch_id} DB consistency PASSED. {total_calculated} total videos.")
        return True
        
    def check_orphan_rows(self):
        # A quick check for orphan VideoRuns with no matching Batch
        from app.models.batch import Batch
        from app.models.video_run import VideoRun
        from app.models.log import Log
        
        # Check orphan VideoRuns
        orphans = self.db.query(VideoRun).filter(~VideoRun.batch_id.in_(self.db.query(Batch.id))).count()
        if orphans > 0:
            logger.error(f"Found {orphans} orphan VideoRun rows!")
            return False
            
        # Check orphan Logs
        log_orphans = self.db.query(Log).filter(~Log.batch_id.in_(self.db.query(Batch.id))).count()
        if log_orphans > 0:
            logger.error(f"Found {log_orphans} orphan Log rows!")
            return False
            
        logger.info("Orphan DB checks PASSED.")
        return True
