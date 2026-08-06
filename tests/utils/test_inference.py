import sys
import os
import cv2

sys.path.append(os.path.abspath("backend"))

from models.inference_video_full_detection import process_video_cycles
from models.renderers.renderer_factory import get_renderer

def run_test():
    video_path = "backend/session_17764421_2026-04-18_00-08-31_Bad.avi"
    output_dir = "backend/test_output"
    
    if not os.path.exists(video_path):
        print(f"Test video not found at: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print("Test video found. Instantiating process_video_cycles...")
    
    # We use None for models; it will fail when inference attempts to run, 
    # but we can verify the pipeline setup doesn't crash on instantiation.
    # To truly verify end-to-end, it's better to run the actual backend server.
    try:
        gen = process_video_cycles(
            video_path=video_path,
            resolved_output_dir=output_dir,
            seg_net=None,
            yolo_socket=None,
            yolo_pose=None,
            print_summary=False,
            enable_debug=False,
            yield_mode=True,
            renderer=get_renderer("production")
        )
        print("Generator created successfully.")
    except Exception as e:
        print(f"Failed to create generator: {e}")

if __name__ == "__main__":
    run_test()
