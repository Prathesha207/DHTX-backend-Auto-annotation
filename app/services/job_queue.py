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
        self.is_unhealthy = False
        
        self.worker_thread = None
        self.watchdog_thread = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BackgroundJobQueue()
        return cls._instance

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

    def _worker_loop(self):
        while True:
            if getattr(self, "is_unhealthy", False):
                time.sleep(10)
                continue
                
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
                            print(f"[JOB QUEUE] Error during session destruction for batch {batch_id_to_run}: {db_e}")
                    finally:
                        with self.lock:
                            self.running_batch_id = None
                        
                        try:
                            from app.services.job_session_manager import job_session_mgr
                            job_session_mgr.destroy_session(batch_id_to_run)
                        except Exception:
                            pass
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            time.sleep(2)

    def _watchdog_loop(self):
        last_frame_no = -1
        stalled_time = 0
        current_stalled_batch = None
        
        while True:
            time.sleep(5)
            
            with self.lock:
                batch_id = self.running_batch_id
                
            if not batch_id:
                stalled_time = 0
                current_stalled_batch = None
                continue
                
            if current_stalled_batch != batch_id:
                current_stalled_batch = batch_id
                stalled_time = 0
                last_frame_no = -1
                
            try:
                from app.services.job_session_manager import job_session_mgr
                session = job_session_mgr.get_session(batch_id)
                if not session or session.status != "running":
                    stalled_time = 0
                    continue
                    
                frame = session.current_frame
                if frame == last_frame_no and frame > 0:
                    stalled_time += 5
                else:
                    last_frame_no = frame
                    stalled_time = 0
                    
                try:
                    from app.models.job_session_state import JobSessionState
                    from app.crud.inference_config import get_config
                    from datetime import datetime
                    with SessionLocal() as db:
                        state_obj = db.query(JobSessionState).filter(JobSessionState.batch_id == batch_id).first()
                        if not state_obj:
                            state_obj = JobSessionState(batch_id=batch_id)
                            db.add(state_obj)
                        state_obj.state_json = session.to_snapshot_dict()
                        state_obj.updated_at = datetime.utcnow().isoformat()
                        
                        config = get_config(db)
                        watchdog_enabled = config.watchdog_enabled
                        watchdog_warning_seconds = config.watchdog_warning_seconds
                        watchdog_fail_seconds = config.watchdog_fail_seconds
                        
                        db.commit()
                except Exception as persist_e:
                    print(f"[WATCHDOG ERROR] Persisting session: {persist_e}")
                    watchdog_enabled = True
                    watchdog_warning_seconds = 30
                    watchdog_fail_seconds = 60
                    
                if not watchdog_enabled:
                    continue
                    
                if stalled_time == watchdog_warning_seconds:
                    print(f"[WATCHDOG] Batch {batch_id} appears stalled at frame {frame}. Requesting graceful cancellation.")
                    with SessionLocal() as db:
                        LogService.warning(db, batch_id, None, "Video appears stalled. Requesting graceful cancellation via watchdog.")
                    self.cancel_event.set()
                    
                elif stalled_time == watchdog_fail_seconds:
                    print(f"[WATCHDOG] Batch {batch_id} hard-hung. Marking backend unhealthy.")
                    self.is_unhealthy = True
                    try:
                        with SessionLocal() as db:
                            from app.models.batch import Batch
                            b = db.query(Batch).filter(Batch.id == batch_id).first()
                            if b and b.status not in ("completed", "cancelled", "failed"):
                                b.status = "failed"
                                db.commit()
                                LogService.error(db, batch_id, None, "Critical worker hang detected. Backend disabled until restarted.")
                    except:
                        pass
                        
                    try:
                        from app.services.websocket_manager import manager
                        manager.send_threadsafe(batch_id, {"type": "batch", "status": "failed"})
                        manager.send_threadsafe(batch_id, {"type": "finished"})
                    except:
                        pass
            except Exception as e:
                print(f"[WATCHDOG ERROR] {e}")


    def cancel_batch(self, batch_id: int):
        with self.lock:
            if self.running_batch_id == batch_id:
                self.cancel_event.set()
                return True
            return False

    def graceful_shutdown(self):
        """Called on application exit to stop any active video cleanly."""
        print("[SHUTDOWN] Stopping background queue and cancelling active batch...")
        self.cancel_event.set()
        # Give the worker thread a few seconds to exit the loop and flush files
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)
        print("[SHUTDOWN] Background queue stopped.")

job_queue = BackgroundJobQueue.get_instance()
