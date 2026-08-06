import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.api_verifier import APIVerifier
from tests.utils.asset_generator import AssetGenerator

def run():
    client = APIClient()
    generator = AssetGenerator()
    
    # 1. Create batch
    batch_resp = client.create_batch(batch_name="Smoke Test Upload")
    assert APIVerifier.verify_batch_create_response(batch_resp), "Batch create schema invalid"
    batch_id = batch_resp["batch_id"]
    
    # 2. Upload exactly 1 golden video
    golden_file = generator.get_golden_file("normal_short.mp4")
    success, upload_resp = client.upload_file(batch_id, golden_file, queue_position=0)
    assert success, f"Upload failed: {upload_resp}"
    assert APIVerifier.verify_upload_response(upload_resp), "Upload response schema invalid"
    
    # 3. Verify manifest updated
    manifest = client.get_manifest(batch_id)
    assert APIVerifier.verify_manifest_response(manifest), "Manifest schema invalid"
    assert len(manifest["uploaded_files"]) == 1, "Manifest should have 1 file"
    assert manifest["uploaded_files"][0]["original_filename"] == "normal_short", f"Manifest filename mismatch: {manifest}"
    
    # 4. Verify status is queued (before start)
    # The actual status might be queued immediately.
    status = client.get_batch_status(batch_id)
    assert status["total_videos"] == 1, "Total videos should be 1"
    
    return batch_id

if __name__ == "__main__":
    run()
