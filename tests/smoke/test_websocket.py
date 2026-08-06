import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.websocket_client import WSClient
from tests.smoke.test_upload import run as upload_run

def run():
    client = APIClient()
    batch_id = upload_run()
    
    ws = WSClient(f"ws://127.0.0.1:8000/ws/{batch_id}")
    ws.connect()
    
    client.start_batch(batch_id)
    
    # 2. Collect events
    start_time = time.time()
    events = []
    while time.time() - start_time < 5:
        event = ws.recv(timeout=1.0)
        if event:
            events.append(event)
            # Verify session UUID and sequence
            assert "session_uuid" in event, "Missing session_uuid"
            assert "sequence" in event, "Missing sequence"
    
    assert len(events) > 0, "No websocket events received"
    
    # 3. Test reconnect (simulate frontend refresh)
    ws.close()
    ws2 = WSClient(f"ws://127.0.0.1:8000/ws/{batch_id}")
    ws2.connect()
    
    event2 = ws2.recv(timeout=2.0)
    assert event2 is not None, "Failed to receive events after reconnect"
    
    ws2.close()
    
if __name__ == "__main__":
    run()
