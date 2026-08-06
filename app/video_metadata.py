import cv2
import os

def probe_video(path: str) -> dict:
    """
    Strict validation of the video file before adding to queue.
    """
    if not os.path.exists(path):
        raise ValueError(f"File does not exist: {path}")
        
    if os.path.getsize(path) == 0:
        raise ValueError(f"File size is 0: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video (is it corrupted?): {path}")

    # Explicitly test decoding the first frame to catch 'moov atom' errors
    success, frame = cap.read()
    if not success or frame is None:
        cap.release()
        raise ValueError(f"Failed to decode first frame of video: {path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if fps > 0 else 0.0
    
    if width <= 0 or height <= 0 or fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata (w:{width}, h:{height}, fps:{fps}, frames:{total_frames})")

    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
    }