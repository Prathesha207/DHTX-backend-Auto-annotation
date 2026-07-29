import threading
import time
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.models.batch import Batch
from app.services.ml_runner import MLRunner
from app.services.log_service import LogService

class BackgroundJobQueue:
    _instance = None

    def __init__(self):
        self.lock = threading.Lock()
        self.running_batch_id = None
        self.cancel_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BackgroundJobQueue()
        return cls._instance

    def _worker_loop(self):
        while True:
            try:
                batch_id_to_run = None
                with SessionLocal() as db:
                    batch = db.query(Batch).filter(
                        Batch.status.in_(["queued", "interrupted"])
                    ).order_by(Batch.created_at.asc()).first()
                    
                    if batch:
                        batch_id_to_run = batch.id
                        self.running_batch_id = batch.id
                        self.cancel_event.clear()

                if batch_id_to_run:
                    try:
                        runner = MLRunner()
                        runner.run_batch(batch_id_to_run, stream_hud=False, cancel_event=self.cancel_event)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        try:
                            with SessionLocal() as db:
                                b = db.query(Batch).filter(Batch.id == batch_id_to_run).first()
                                if b and b.status not in ("completed", "cancelled"):
                                    b.status = "failed"
                                    db.commit()
                        except Exception as db_e:
                            print(f"[JOB QUEUE ERROR] Failed to update batch {batch_id_to_run} to failed state: {db_e}")
                    finally:
                        with self.lock:
                            self.running_batch_id = None
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            time.sleep(2)

    def cancel_batch(self, batch_id: int):
        with self.lock:
            if self.running_batch_id == batch_id:
                self.cancel_event.set()
                return True
            return False

job_queue = BackgroundJobQueue.get_instance()
