import cv2
import time
import base64
from typing import Optional
from sqlalchemy.orm import Session
from app.services.websocket_manager import manager
from app.plugins.base_plugin import BaseMLPlugin
from app.crud.cycle import create_cycle, update_cycle
from app.models.inference_config import InferenceConfig
from app.services.ui_state_manager import UIStateManager
from app.services.job_session_manager import job_session_mgr
class ApplicationController:
    def __init__(self, db: Session, batch_id: int, video_run_id: int, video_path: str, output_dir: str):
        self.db = db
        self.batch_id = batch_id
        self.video_run_id = video_run_id
        self.video_path = video_path
        self.output_dir = output_dir

    def run_inference(self, plugin: BaseMLPlugin, cancel_event=None) -> bool:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        from pathlib import Path
        import os
        
        base_name = Path(self.video_path).stem
        
        # (Removed dummy excel report generation)

        from app.services.video_run_service import VideoRunService

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        t0 = time.time()
        
        last_cycle_no = 0
        db_cycle_id = None
        
        last_heartbeat_time = 0
        last_ws_frame_time = 0.0
        
        decoded_frame = 0
        processed_frame = 0
        rendered_frame = 0
        sent_frame = 0

        # Fetch UI State Config
        config = self.db.query(InferenceConfig).first()
        ui_state_manager = UIStateManager(config)
        preview_interval = 1.0 / config.preview_fps if config and config.preview_fps > 0 else 0.05

        # Fetch VideoRun from DB once before the loop
        from app.models.video_run import VideoRun
        video = self.db.query(VideoRun).filter(VideoRun.id == self.video_run_id).first()

        success = True
        try:
            while cap.isOpened():
                if cancel_event and cancel_event.is_set():
                    print("[DEBUG] cancel_event.is_set() is TRUE inside application_controller loop!")
                    manager.send_threadsafe(self.batch_id, {"type": "batch", "status": "STOPPING"})
                    success = "CANCELLED"
                    break

                frame_capture_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_frame += 1

                decoded_frame += 1

                session = job_session_mgr.get_session(self.batch_id)
                show_roi = session.show_roi if session else False

                # The Plugin isolates all ML logic!
                t_plugin_start = time.time()
                result = plugin.process_frame(frame, show_roi=show_roi)
                if not result:
                    continue
                
                processed_frame += 1

                vis = result.get("frame", frame)
                
                # Use the pre-fetched video object instead of querying on every frame
                if video:
                    VideoRunService.update_progress(
                        db=self.db,
                        video=video,
                        current_frame=current_frame,
                        total_frames=total_frames
                    )
                
                elapsed = time.time() - t0
                current_fps = current_frame / elapsed if elapsed > 0 else 0
                decode_fps = decoded_frame / elapsed if elapsed > 0 else 0
                processing_fps = processed_frame / elapsed if elapsed > 0 else 0
                streaming_fps = sent_frame / elapsed if elapsed > 0 else 0
                
                eta = (total_frames - current_frame) / current_fps if current_fps > 0 else 0

                manager.send_threadsafe(self.batch_id, {
                    "type": "progress",
                    "current_frame": result.get("frame_idx", current_frame),
                    "progress": (current_frame / total_frames) * 100 if total_frames else 0,
                    "fps": result.get("fps", current_fps),
                    "decode_fps": decode_fps,
                    "processing_fps": processing_fps,
                    "streaming_fps": streaming_fps,
                    "decoded_frame": decoded_frame,
                    "processed_frame": processed_frame,
                    "rendered_frame": rendered_frame,
                    "sent_frame": sent_frame,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "frame_capture_time": frame_capture_time,
                })
                
                # Dedicated Heartbeat Event (every 3 seconds)
                now = time.time()
                if now - last_heartbeat_time >= 3.0:
                    import torch
                    from app.database.database import SessionLocal
                    from app.models.batch import Batch
                    from app.core.config import SEGMENTATION_MODEL_NAME
                    try:
                        with SessionLocal() as _db:
                            queue_length = _db.query(Batch).filter(Batch.status.in_(["queued", "interrupted"])).count()
                    except:
                        queue_length = 0
                        
                    manager.send_threadsafe(self.batch_id, {
                        "type": "heartbeat",
                        "status": "Backend Healthy",
                        "worker_state": "Processing",
                        "gpu_busy": torch.cuda.is_available(),
                        "queue_length": queue_length,
                        "timestamp": now,
                        "backend_version": "1.7.3",
                        "pipeline_version": "v46",
                        "yolo_version": "8.2",
                        "pose_version": "v46",
                        "segmentation_model": SEGMENTATION_MODEL_NAME
                    })
                    last_heartbeat_time = now

                # Broadcast live preview via WebSockets
                now = time.time()
                if now - last_ws_frame_time >= preview_interval:
                    try:
                        # Use the annotated 'vis' frame instead of the raw 'frame'
                        _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        manager.send_threadsafe(self.batch_id, {
                            "type": "frame",
                            "video_run_id": self.video_run_id,
                            "base64": b64_str,
                            "invisible_roi": result.get("invisible_roi"),
                            "frame_capture_time": frame_capture_time,
                            "frame_encoded_time": time.time()
                        })
                        last_ws_frame_time = now
                        sent_frame += 1
                    except Exception:
                        pass
                
                # Update the pure UI-only state machine
                ui_state = ui_state_manager.update(result)
                rendered_frame += 1
                
                # Send Status over WebSocket
                manager.send_threadsafe(self.batch_id, {
                    "type": "status",
                    "video_run_id": self.video_run_id,
                    "state": ui_state["state"],
                    "progress": ui_state["progress"],
                    "cycle_stats": {
                        "cycle": result.get("cycle_no", 0),
                        "current_verdict": result.get("final_verdict", "UNKNOWN"),
                        "ok_votes": result.get("normal_votes", 0),
                        "anomaly_votes": result.get("anomaly_votes", 0),
                        "elapsed_s": 0,
                    },
                    "perf": {
                        "fps": result.get("fps", 0),
                        "frame_time_ms": result.get("frame_ms", 0),
                        "avg_latency": result.get("frame_ms_avg", 0),
                        "cpu": 0,
                        "encode_ms": 0,
                    },
                    "models": ui_state["models"],
                    "decision": {
                        "socket": "Present" if result.get("socket_hit") and result.get("socket_hit", {}).get("class") == 1 else "Absent",
                        "hand": "Present" if result.get("hand_hit") else "Absent",
                        "tubes": {t: "Present" for t in result.get("detected_seq", [])},
                        "sequence": result.get("detected_seq", []),
                        "vote": result.get("final_verdict", "N/A"),
                    }
                })

                # DB Logic for Cycles
                active_cycle = result.get("active_cycle", False)
                cycle_no = result.get("cycle_no", 0)

                if active_cycle and cycle_no > last_cycle_no:
                    # New cycle started
                    last_cycle_no = cycle_no
                    db_cycle = create_cycle(self.db, self.video_run_id, cycle_no)
                    db_cycle_id = db_cycle.id

                if not active_cycle and db_cycle_id is not None:
                    # Cycle finished
                    update_cycle(
                        db=self.db,
                        cycle_id=db_cycle_id,
                        final_verdict=result.get("final_verdict", "UNKNOWN"),
                        anomaly_ratio=result.get("anomaly_ratio", 0.0),
                        normal_votes=result.get("normal_votes", 0),
                        anomaly_votes=result.get("anomaly_votes", 0)
                    )
                    db_cycle_id = None
                
                # Yield GIL to ensure the asyncio event loop can process WebSockets and heartbeats
                time.sleep(0.005)
        finally:
            if success is True and not (cancel_event and cancel_event.is_set()):
                # Correct the total frames in case OpenCV overestimated it
                from app.models.video_run import VideoRun
                video = self.db.query(VideoRun).filter(VideoRun.id == self.video_run_id).first()
                if video and current_frame > 0:
                    VideoRunService.update_progress(
                        db=self.db,
                        video=video,
                        current_frame=current_frame,
                        total_frames=current_frame
                    )
                manager.send_threadsafe(self.batch_id, {
                    "type": "progress",
                    "current_frame": current_frame,
                    "progress": 100.0,
                    "fps": 0,
                    "elapsed_seconds": time.time() - t0 if 't0' in locals() else 0,
                    "eta_seconds": 0,
                    "total_frames": current_frame
                })
                
            cap.release()
            print(f"[DEBUG] application_controller returning success={success}")
            
        return success
