import argparse
import time
import sys
import random
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
from tests.utils.leak_detector import LeakDetector

def simulate_user_upload(client, generator, batch_id, queue_pos, randomize=True):
    if randomize:
        time.sleep(random.uniform(0.1, 1.0)) # Latency
    golden = generator.get_golden_file("normal_short.mp4")
    client.upload_file(batch_id, golden, queue_position=queue_pos)
    
def run_scale(scale):
    print(f"--- Running Stress Test: Scale {scale} ---")
    client = APIClient()
    gen = AssetGenerator()
    
    # 1. Create Batch
    resp = client.create_batch(batch_name=f"Stress_{scale}")
    batch_id = resp["batch_id"]
    
    print(f"Batch {batch_id} created. Uploading {scale} videos (with randomized network latency)...")
    
    # 2. Uploads
    start_up = time.time()
    for i in range(1, scale + 1):
        simulate_user_upload(client, gen, batch_id, i, randomize=True)
    up_dur = time.time() - start_up
    print(f"Uploads completed in {up_dur:.1f}s (Avg {up_dur/scale:.3f}s per video)")
    
    # 3. Inference
    print("Starting Inference...")
    client.start_batch(batch_id)
    
    start_inf = time.time()
    while True:
        status = client.get_batch_status(batch_id)
        if status.get("status") == "completed":
            break
        elif status.get("status") in ["failed", "cancelled"]:
            raise RuntimeError(f"Stress test failed: {status}")
        time.sleep(5) # poll every 5s
        
    inf_dur = time.time() - start_inf
    print(f"Inference completed in {inf_dur:.1f}s (Avg {inf_dur/scale:.3f}s per video)")
    
    # 4. Metrics
    # In a real run, report_generator would be used here to record DB commit latency, etc.
    print(f"Scale {scale} PASSED.")

def main():
    parser = argparse.ArgumentParser(description="Run Stress Suite")
    parser.add_argument("--scale", type=int, choices=[10, 50, 200, 500, 1000], default=10, help="Number of videos")
    args = parser.parse_args()
    
    leak = LeakDetector()
    leak.capture_baseline()
    
    run_scale(args.scale)
    
    if not leak.check_leaks():
        print("FAIL: Leaks detected during stress run!")
        sys.exit(1)
        
    print("Stress suite completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
