import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.websocket_client import WSClient
from tests.integration.test_upload_resume import run as upload_resume_run

def run():
    client = APIClient()
    
    # Run the upload resume test to get a batch with 3 videos
    batch_id = upload_resume_run()
    
    ws = WSClient()
    ws.connect()
    ws.subscribe(batch_id)
    
    client.start_batch(batch_id)
    
    start_time = time.time()
    events = []
    
    # Collect events for 5 seconds to ensure we get a few heartbeats/updates
    while time.time() - start_time < 5:
        event = ws.recv(timeout=1.0)
        if event:
            events.append(event)
            
    ws.close()
    
    assert len(events) > 0, "No websocket events received"
    
    last_ts = 0
    last_seq = -1
    session_uuid = None
    
    for e in events:
        # Check sequence
        seq = e.get("sequence", -1)
        assert seq >= last_seq, f"Sequence decreased from {last_seq} to {seq}"
        last_seq = seq
        
        # Check session UUID persistence
        e_uuid = e.get("session_uuid")
        assert e_uuid is not None, "Missing session_uuid"
        if session_uuid is None:
            session_uuid = e_uuid
        else:
            assert session_uuid == e_uuid, "session_uuid changed during single batch"
            
        # Check timestamp
        ts = e.get("timestamp", 0)
        assert ts >= last_ts, "Timestamp decreased"
        last_ts = ts
        
    # Now create a NEW batch to ensure session_uuid changes
    client.cancel_batch(batch_id) # Cancel previous just in case
    
    # Create batch 2
    import requests
    resp = requests.post(f"{client.base_url}/upload/batch/create", json={"batch_name": "Batch 2", "total_videos": 1})
    batch_id_2 = resp.json()["batch_id"]
    client.upload_file(batch_id_2, "backend/tests/golden/normal_short.mp4", queue_position=1)
    
    ws2 = WSClient()
    ws2.connect()
    ws2.subscribe(batch_id_2)
    client.start_batch(batch_id_2)
    
    event2 = ws2.recv(timeout=5.0)
    ws2.close()
    client.cancel_batch(batch_id_2)
    
    assert event2 is not None, "Failed to receive events from second batch"
    uuid2 = event2.get("session_uuid")
    assert uuid2 is not None, "Missing session_uuid on second batch"
    assert session_uuid != uuid2, "session_uuid did not change for the new batch"

    print("Heartbeat telemetry test PASSED.")
    
if __name__ == "__main__":
    run()
