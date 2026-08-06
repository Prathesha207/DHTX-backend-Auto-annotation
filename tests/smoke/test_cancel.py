import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.smoke.test_upload import run as upload_run

def run():
    client = APIClient()
    batch_id = upload_run()
    
    # 1. Start inference
    client.start_batch(batch_id)
    time.sleep(2) # let it start
    
    # 2. Cancel midway
    cancel_resp = client.cancel_batch(batch_id)
    assert cancel_resp.get("message") == "Batch cancelled", f"Cancel failed: {cancel_resp}"
    
    # Wait for status to reflect
    time.sleep(2)
    status = client.get_batch_status(batch_id)
    assert status.get("status") == "cancelled", f"Status is not cancelled: {status.get('status')}"
    
    # 3. Verify cleanup / DB state / GPU release
    # We will rely on LeakDetector in the runner to catch GPU/Thread leaks.

if __name__ == "__main__":
    run()
