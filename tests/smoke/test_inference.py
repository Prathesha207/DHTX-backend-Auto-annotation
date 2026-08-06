import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
from tests.smoke.test_upload import run as upload_run

def run():
    client = APIClient()
    
    # 1. Upload one normal video (reuse test_upload logic)
    batch_id = upload_run()
    
    # 2. Start inference
    client.start_batch(batch_id)
    
    # Wait for completion (timeout 90s)
    start_time = time.time()
    while time.time() - start_time < 90:
        status = client.get_batch_status(batch_id)
        if status.get("status") == "completed":
            break
        elif status.get("status") in ["failed", "failed_inference", "cancelled"]:
            raise AssertionError(f"Batch failed during inference: {status}")
        time.sleep(1)
        
    final_status = client.get_batch_status(batch_id)
    assert final_status.get("status") == "completed", "Batch did not complete within timeout"
    
    # 3. Verify DB updated
    assert final_status.get("completed_videos") == 1, "completed_videos should be 1"
    
    # 4. Verify Filesystem valid (Output, Excel, etc)
    # This will be handled in assertions or we can do it directly
    from app.database.database import SessionLocal
    from tests.utils.assertions import assert_no_orphan_files
    from app.core.config import config_manager
    OUTPUT_DIR = config_manager.get("paths").get("output_dir", "outputs")
    
    output_path = final_status.get("output_path")
    if not output_path:
        raise AssertionError("Batch output_path is missing")
    
    parent_dir = os.path.dirname(output_path)
    b_name = os.path.basename(output_path)
    
    assert_no_orphan_files(parent_dir, b_name, ["normal_short.mp4"])
    
if __name__ == "__main__":
    run()
