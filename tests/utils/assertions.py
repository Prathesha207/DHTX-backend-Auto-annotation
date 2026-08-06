import logging
from sqlalchemy.orm import Session
from tests.utils.db_verifier import DBVerifier
from tests.utils.filesystem_verifier import FilesystemVerifier
from tests.utils.leak_detector import LeakDetector
import requests
import time

logger = logging.getLogger("Assertions")

def assert_backend_running(port=8000):
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        assert resp.status_code == 200, "Health endpoint returned non-200"
        logger.info("Backend is running.")
    except Exception as e:
        logger.error(f"Backend is not running: {e}")
        raise AssertionError(f"Backend is not running: {e}")

def assert_database_consistent(db: Session, batch_id: int):
    verifier = DBVerifier(db)
    assert verifier.verify_batch(batch_id) == True, f"DB consistency failed for batch {batch_id}"

def assert_no_orphan_rows(db: Session):
    verifier = DBVerifier(db)
    assert verifier.check_orphan_rows() == True, "Orphan rows detected in DB"

def assert_no_orphan_files(output_dir: str, batch_name: str, expected_videos: list):
    verifier = FilesystemVerifier(output_dir)
    assert verifier.verify_artifacts(batch_name, expected_videos) == True, "Filesystem check failed (orphans or missing files)"

def assert_batch_completed(db: Session, batch_id: int):
    from app.models.batch import Batch
    b = db.query(Batch).filter(Batch.id == batch_id).first()
    assert b is not None, "Batch not found"
    assert b.total_videos == b.completed_videos, f"Batch not completed. {b.completed_videos}/{b.total_videos}"
    assert b.status == "completed", f"Batch status is {b.status}, expected completed"

def assert_gpu_clean(leak_detector: LeakDetector):
    assert leak_detector.check_leaks() == True, "Leaks detected (GPU, Threads, RAM, or Files)"

def assert_job_session_consistent():
    # In a real environment, query the application state.
    # We can fetch /health and check the worker status
    resp = requests.get("http://127.0.0.1:8000/health")
    data = resp.json()
    worker = data.get("worker", {})
    assert worker.get("status") in ["idle", "processing"], f"Worker in bad state: {worker.get('status')}"
