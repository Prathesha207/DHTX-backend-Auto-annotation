import argparse
import time
import sys
import json
import os
import importlib

# Ensure tests use the test database
os.environ["DATABASE_URL"] = "sqlite:///./db/test.db"

# Delete the test database to ensure a clean slate
test_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "test.db"))
if os.path.exists(test_db_path):
    try:
        os.remove(test_db_path)
    except Exception as e:
        print(f"Warning: Could not delete {test_db_path}: {e}")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import smoke tests
from tests.smoke import (
    test_environment,
    test_upload,
    test_inference,
    test_websocket,
    test_cancel,
    test_recovery,
    test_health
)
from tests.utils.leak_detector import LeakDetector
from tests.utils.assertions import assert_gpu_clean

TESTS = [
    ("Environment", test_environment.run, 5.0), # (name, func, threshold)
    ("Upload", test_upload.run, 5.0),
    ("Inference", test_inference.run, 90.0),
    ("WebSocket", test_websocket.run, 15.0),
    ("Cancel", test_cancel.run, 5.0),
    ("Recovery", test_recovery.run, 5.0),
    ("Health", test_health.run, 10.0),
]

def run_single_suite():
    print("-" * 40)
    leak_detector = LeakDetector()
    leak_detector.capture_baseline()
    
    results = []
    failed = False
    
    for name, func, threshold in TESTS:
        if failed:
            results.append((name, "SKIP", 0.0, threshold))
            continue
            
        start = time.time()
        try:
            func()
            dur = time.time() - start
            status = "PASS" if dur <= threshold else "WARN"
            results.append((name, status, dur, threshold))
            print(f"{name:<15} {status:<6} {dur:.1f} s")
        except Exception as e:
            dur = time.time() - start
            results.append((name, "FAIL", dur, threshold))
            print(f"{name:<15} FAIL   {dur:.1f} s")
            print(f"  -> Error: {e}")
            failed = True
            
    # Check Leaks
    if not failed:
        try:
            assert_gpu_clean(leak_detector)
        except Exception as e:
            print(f"Leak Check      FAIL   0.0 s")
            print(f"  -> Error: {e}")
            failed = True
            
    return not failed, results

def main():
    parser = argparse.ArgumentParser(description="Run Enterprise Smoke Tests")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to run the suite")
    args = parser.parse_args()
    
    print(f"Starting Smoke Suite (Repeats: {args.repeat})")
    
    metrics = []
    run_statuses = []
    
    start_all = time.time()
    
    for i in range(args.repeat):
        print(f"\nRun {i+1}/{args.repeat}")
        passed, results = run_single_suite()
        run_statuses.append("PASS" if passed else "FAIL")
        
        # Save metrics for the first run or overall
        if passed:
            for name, status, dur, _ in results:
                metrics.append({
                    "run": i+1,
                    "test": name,
                    "duration": dur,
                    "status": status
                })
        else:
            break # Stop on first failure in sequence
            
    total_dur = time.time() - start_all
    
    # Save metrics
    metrics_path = os.path.join(os.path.dirname(__file__), "smoke_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\n" + "-" * 40)
    for i, st in enumerate(run_statuses):
        print(f"Run {i+1} {st}")
        
    flaky = False
    if args.repeat > 1 and len(run_statuses) > 1 and len(set(run_statuses)) > 1:
        flaky = True
        
    print(f"Flaky Tests      {1 if flaky else 0}")
    print("-" * 40)
    
    overall = "PASS" if all(s == "PASS" for s in run_statuses) else "FAIL"
    print(f"Overall          {overall}")
    print(f"Duration         {total_dur:.1f} s")
    
    if overall == "FAIL" or flaky:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
