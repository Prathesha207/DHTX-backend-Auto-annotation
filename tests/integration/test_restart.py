import os
import sys
import time
import subprocess
import signal
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
from app.database.database import SessionLocal
from app.models.batch import Batch
from app.models.video_run import VideoRun

def start_backend():
    # Start the backend via uvicorn in a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8001"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    # Wait for startup
    client = APIClient()
    client.base_url = "http://127.0.0.1:8001"
    for _ in range(20):
        try:
            resp = requests.get(f"{client.base_url}/health")
            if resp.status_code == 200:
                return process, client
        except:
            time.sleep(0.5)
    raise RuntimeError("Backend failed to start")

def kill_backend(process):
    # SIGKILL equivalent on Windows is TerminateProcess / taskkill
    if os.name == 'nt':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
    else:
        os.kill(process.pid, signal.SIGKILL)
    process.wait()

def run():
    print("--- Subtest A: SIGKILL during upload ---")
    process, client = start_backend()
    generator = AssetGenerator()
    
    resp = requests.post(f"{client.base_url}/upload/batch/create", json={
        "batch_name": "Restart Upload Test",
        "total_videos": 2
    })
    batch_id = resp.json()["batch_id"]
    
    # Upload one video, then kill
    golden = generator.get_golden_file("normal_short.mp4")
    client.upload_file(batch_id, golden, queue_position=1)
    
    # Kill the backend!
    kill_backend(process)
    
    # Verify DB state manually
    db = SessionLocal()
    b = db.query(Batch).filter(Batch.id == batch_id).first()
    assert b.status == "waiting", f"Batch should be waiting, got {b.status}"
    
    # Restart backend
    process2, client2 = start_backend()
    
    # The batch should remain waiting. Upload 2nd video.
    client2.upload_file(batch_id, generator.get_golden_file("anomaly_short.mp4"), queue_position=2)
    
    manifest = client2.get_manifest(batch_id)
    assert len(manifest["uploaded_files"]) == 2, "Should have exactly 2 files in manifest, no duplicates"
    
    # Clean up
    kill_backend(process2)
    db.close()
    print("Subtest A PASSED.")
    
    print("--- Subtest B: SIGKILL during inference ---")
    process3, client3 = start_backend()
    
    resp = requests.post(f"{client3.base_url}/upload/batch/create", json={
        "batch_name": "Restart Inference Test",
        "total_videos": 1
    })
    batch_id_b = resp.json()["batch_id"]
    client3.upload_file(batch_id_b, generator.get_golden_file("normal_long.mp4"), queue_position=1)
    client3.start_batch(batch_id_b)
    
    time.sleep(2) # Let it start processing
    
    # Kill it
    kill_backend(process3)
    
    # Verify DB manually says running (since it was brutally killed without cleanup)
    db2 = SessionLocal()
    b2 = db2.query(Batch).filter(Batch.id == batch_id_b).first()
    assert b2.status == "running", "Batch should still say running in DB due to ungraceful kill"
    db2.close()
    
    # Restart backend
    process4, client4 = start_backend()
    
    # Backend startup should have recovered it to "interrupted"
    status = client4.get_batch_status(batch_id_b)
    assert status.get("status") == "interrupted", f"Batch did not recover to interrupted: {status.get('status')}"
    
    # Manual retry
    client4.resume_batch(batch_id_b)
    time.sleep(1)
    status2 = client4.get_batch_status(batch_id_b)
    assert status2.get("status") in ["running", "completed"], "Batch did not resume"
    
    # Cleanup
    kill_backend(process4)
    print("Subtest B PASSED.")
    
if __name__ == "__main__":
    run()
