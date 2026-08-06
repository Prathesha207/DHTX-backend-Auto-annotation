import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils.api_client import APIClient
from tests.utils.asset_generator import AssetGenerator
from tests.utils.leak_detector import LeakDetector

def parse_duration(dur_str):
    if dur_str.endswith('m'):
        return int(dur_str[:-1]) * 60
    elif dur_str.endswith('h'):
        return int(dur_str[:-1]) * 3600
    return int(dur_str)

def main():
    parser = argparse.ArgumentParser(description="Run Soak Suite")
    parser.add_argument("--duration", type=str, default="15m", help="Duration (15m, 1h, 6h, 24h)")
    args = parser.parse_args()
    
    total_seconds = parse_duration(args.duration)
    print(f"Starting Soak Test for {args.duration} ({total_seconds} seconds)...")
    
    leak = LeakDetector()
    leak.capture_baseline()
    
    client = APIClient()
    gen = AssetGenerator()
    golden = gen.get_golden_file("normal_short.mp4")
    
    start_time = time.time()
    loops = 0
    
    while time.time() - start_time < total_seconds:
        loops += 1
        print(f"[{time.strftime('%H:%M:%S')}] Loop {loops}...")
        
        # 1. Create and upload
        resp = client.create_batch(batch_name=f"Soak_Loop_{loops}")
        batch_id = resp["batch_id"]
        client.upload_file(batch_id, golden, queue_position=1)
        
        # 2. Start
        client.start_batch(batch_id)
        
        # 3. Wait
        while True:
            status = client.get_batch_status(batch_id)
            if status.get("status") == "completed":
                break
            time.sleep(2)
            
    print(f"Soak completed {loops} loops.")
    print("Verifying leak baseline...")
    
    if not leak.check_leaks():
        print("FAIL: Leaks detected at end of soak test!")
        sys.exit(1)
        
    print(f"Soak Suite ({args.duration}) completed successfully with ZERO LEAKS.")
    sys.exit(0)

if __name__ == "__main__":
    main()
