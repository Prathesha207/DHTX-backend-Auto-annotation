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
    
    # 2. Ping health every 500ms
    start_time = time.time()
    healthy_pings = 0
    while time.time() - start_time < 5:
        try:
            health = client.get_health()
            assert health.get("status") == "healthy", f"Backend unhealthy during inference: {health.get('status')}"
            worker = health.get("worker", {})
            # Verify worker is processing or idle (if finished fast)
            assert worker.get("status") in ["idle", "processing"], f"Worker bad state: {worker.get('status')}"
            healthy_pings += 1
        except Exception as e:
            raise AssertionError(f"Health check failed during inference: {e}")
        time.sleep(0.5)
        
    assert healthy_pings >= 5, "Did not get enough health pings"
    
if __name__ == "__main__":
    run()
