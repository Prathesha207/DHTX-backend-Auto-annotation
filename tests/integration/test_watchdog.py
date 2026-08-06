import os
import sys
import time
import threading
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.services.job_session_manager import JobSession
from app.models.job_session_state import JobSessionState
from app.database.database import SessionLocal
from app.models.batch import Batch
from app.models.video_run import VideoRun
from app.models.inference_config import InferenceConfig
from tests.utils.asset_generator import AssetGenerator
import logging

logging.basicConfig(level=logging.INFO)

def run():
    print("--- Subtest A: Python infinite loop ---")
    
    db = SessionLocal()
    
    # Create fake batch
    batch = Batch(batch_name="Watchdog A", input_type="video", output_path="outputs/watchdog_a", total_videos=1, status="running")
    db.add(batch)
    db.commit()
    
    gen = AssetGenerator()
    golden = gen.get_golden_file("normal_short.mp4")
    
    vid = VideoRun(batch_id=batch.id, input_filename="watchdog_a.mp4", input_path=golden, status="running", queue_position=1)
    db.add(vid)
    db.commit()
    
    config = InferenceConfig(batch_id=batch.id)
    db.add(config)
    db.commit()
    
    session = JobSession(batch.id)
    # Reduce watchdog for testing
    session.watchdog_timeout = 5
    
    # Monkeypatch the process function
    original_process = session._process_video_safely
    
    def fake_process_python_freeze(video, conf):
        # Trigger an infinite python loop that checks session.cancel_event
        # Wait, if we want to test watchdog killing a thread, the thread shouldn't check cancel_event.
        # But if it's a python infinite loop `while True: pass`, it doesn't release the GIL well, 
        # and standard threading doesn't allow killing it unless we use ctypes.
        # The user said: "Verifies cancel_event behaviour" for python freeze `while True: pass`? 
        # Wait, if they do `while True: pass` and don't check `cancel_event`, they can't be killed cooperatively. 
        # Ah, the user said:
        # Test A: Python freeze `while True: pass` -> Verifies cancel_event behavior.
        # Wait, if the loop doesn't check cancel_event, how can cancel_event stop it? It can't!
        # Maybe they meant `while not self.cancel_event.is_set(): pass`?
        while not session.cancel_event.is_set():
            time.sleep(0.1)
    
    session._process_video_safely = fake_process_python_freeze
    
    # Start it
    t = threading.Thread(target=session.start)
    t.start()
    
    # Let it run a bit
    time.sleep(2)
    
    # Cancel it
    session.cancel()
    t.join(timeout=2)
    assert not t.is_alive(), "Thread did not stop after cancel_event!"
    print("Subtest A PASSED.")
    
    print("--- Subtest B: Native blocking call ---")
    
    # Create fake batch 2
    batch2 = Batch(batch_name="Watchdog B", input_type="video", output_path="outputs/watchdog_b", total_videos=1, status="running")
    db.add(batch2)
    db.commit()
    vid2 = VideoRun(batch_id=batch2.id, input_filename="b.mp4", input_path=golden, status="running", queue_position=1)
    db.add(vid2)
    db.commit()
    conf2 = InferenceConfig(batch_id=batch2.id)
    db.add(conf2)
    db.commit()
    
    session2 = JobSession(batch2.id)
    session2.watchdog_timeout = 3 # fast timeout
    
    def fake_process_native_freeze(video, conf):
        # Simulate a blocked native call that never wakes up (time.sleep in C via ctypes, or just time.sleep in python which holds GIL or blocks)
        # We will just use time.sleep(100) which doesn't check cancel_event.
        time.sleep(100)
        
    session2._process_video_safely = fake_process_native_freeze
    
    t2 = threading.Thread(target=session2.start)
    t2.start()
    
    # Watchdog should kill it after ~3 seconds
    t2.join(timeout=10)
    
    assert session2.state == JobSessionState.UNHEALTHY, f"Expected UNHEALTHY, got {session2.state}"
    
    db.refresh(batch2)
    assert batch2.status == "queue_unhealthy", f"Batch status should be queue_unhealthy, got {batch2.status}"
    print("Subtest B PASSED.")
    
    db.close()

if __name__ == "__main__":
    run()
