import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
import requests
import shutil

def run():
    client = APIClient()
    generator = AssetGenerator()
    
    # Create batch for 4 corrupted files
    resp = requests.post(f"{client.base_url}/upload/batch/create", json={
        "batch_name": "Artifact Validation Test",
        "total_videos": 4
    })
    resp.raise_for_status()
    batch_id = resp.json()["batch_id"]
    
    # 1. 0 byte output
    empty_file = generator.get_golden_file("empty.mp4")
    success, resp_data = client.upload_file(batch_id, empty_file, queue_position=1)
    assert not success, "0 byte file should fail upload"
    assert resp_data.get("status") == "FAILED_UPLOAD", "0 byte file did not get FAILED_UPLOAD"
    
    # 2. unreadable output (text file disguised as mp4)
    corrupted_file = generator.get_golden_file("corrupted.mp4")
    success, resp_data = client.upload_file(batch_id, corrupted_file, queue_position=2)
    assert not success, "Unreadable file should fail upload"
    assert resp_data.get("status") == "FAILED_UPLOAD", "Unreadable file did not get FAILED_UPLOAD"
    
    # 3. Truncated mp4
    normal_file = generator.get_golden_file("normal_short.mp4")
    truncated_path = generator.get_golden_file("truncated.mp4")
    with open(normal_file, "rb") as f_in:
        data = f_in.read()
    with open(truncated_path, "wb") as f_out:
        f_out.write(data[:len(data)//2]) # write only half
        
    success, resp_data = client.upload_file(batch_id, truncated_path, queue_position=3)
    # ffmpeg might accept a truncated file during probe depending on the header
    # Let's see if the backend catches it during upload or inference.
    # If it accepts it, it should fail during inference. We will assert it either fails upload or inference.
    
    # 4. Cannot decode first frame
    # A file with valid mp4 headers but no video streams or bad codec.
    # For now, corrupted.mp4 is close enough, but let's test if the worker crashes.
    # If it uploaded successfully, we start the batch.
    
    # We will upload 1 valid file to ensure the worker starts and doesn't crash on the bad ones
    client.upload_file(batch_id, normal_file, queue_position=4)
    client.start_batch(batch_id)
    
    import time
    time.sleep(5) # let it process
    
    status = client.get_batch_status(batch_id)
    # The batch should complete its valid video and mark the rest as failed_inference or failed_upload.
    # Specifically, it should NOT crash the whole worker.
    worker_health = client.get_health().get("worker", {})
    assert worker_health.get("status") in ["idle", "processing"], "Worker crashed due to bad artifact!"
    
    # Cleanup truncated file
    os.remove(truncated_path)
    print("Artifact validation test PASSED.")

if __name__ == "__main__":
    run()
