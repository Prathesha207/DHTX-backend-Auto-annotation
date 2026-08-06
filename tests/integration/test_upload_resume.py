import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
import requests

def run():
    client = APIClient()
    generator = AssetGenerator()
    
    # 1. Create a batch for 3 videos
    # Note: create_batch must send total_videos as well.
    resp = requests.post(f"{client.base_url}/upload/batch/create", json={
        "batch_name": "Resume Test",
        "total_videos": 3
    })
    resp.raise_for_status()
    batch_id = resp.json()["batch_id"]
    
    golden1 = generator.get_golden_file("normal_short.mp4")
    golden2 = generator.get_golden_file("normal_long.mp4")
    golden3 = generator.get_golden_file("anomaly_short.mp4")
    
    # 2. Upload first video
    client.upload_file(batch_id, golden1, queue_position=1)
    
    # 3. Simulate disconnect (we just stop uploading)
    
    # 4. Reconnect and get manifest
    manifest = client.get_manifest(batch_id)
    uploaded = manifest.get("uploaded_files", [])
    
    assert len(uploaded) == 1, f"Expected 1 file in manifest, got {len(uploaded)}"
    assert uploaded[0] == "normal_short.mp4", "Manifest filename mismatch"
    
    # 5. Upload the rest
    client.upload_file(batch_id, golden2, queue_position=2)
    client.upload_file(batch_id, golden3, queue_position=3)
    
    manifest_final = client.get_manifest(batch_id)
    assert len(manifest_final["uploaded_files"]) == 3, "Manifest should have 3 files now"
    
    print("Upload resume via manifest test PASSED.")
    return batch_id

if __name__ == "__main__":
    run()
