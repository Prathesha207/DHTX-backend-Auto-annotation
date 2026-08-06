import os
import sys
from datetime import datetime
import threading
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session
from app.database.database import SessionLocal

from app.crud.batch import get_batch
from app.crud.video_run import get_batch_video_runs
from app.crud.cycle import get_video_cycles
from app.models.video_run import VideoRun
import json

from app.services.batch_service import BatchService
from app.services.video_run_service import VideoRunService
from app.services.log_service import LogService
from app.services.excel_parser import ExcelParser
from app.services.websocket_manager import manager
from app.video_metadata import probe_video
from app.crud.inference_config import get_config
from app.services.application_controller import ApplicationController
from app.services.plugin_manager import PluginManager

# ============================================================
# Configuration
# ============================================================

from app.services.inference_settings import settings

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "ml"

if settings.ml_pipeline == "new":
    SCRIPT_PATH = BASE_DIR / "models" / "inference_video_full_detection_new.py"
else:
    SCRIPT_PATH = BASE_DIR / "models" / "inference_video_full_detection.py"
YOLO_MODEL = MODEL_DIR / "best.pt"
POSE_MODEL = MODEL_DIR / "yolov8n-pose.pt"
SEQUENCE_MODEL = MODEL_DIR / "best_model_optimized.pth"

# ============================================================
# ML Runner
# ============================================================

class MLRunner:

    def run_batch(self, batch_id: int, stream_hud: bool = False, cancel_event=None):
        from app.core.config import config_manager
        max_retries = config_manager.get("queue_settings", {}).get("max_retries_recoverable", 3)
        
        with SessionLocal() as db:
            batch = get_batch(db, batch_id)

            if batch is None:
                return

            BatchService.start(db, batch)

            LogService.info(
                db=db,
                batch_id=batch.id,
                message=f"Starting Batch {batch.id}",
            )

        # The static video_ids loop is removed in favor of a dynamic consumer loop
        import time
        wait_seconds = 0
        from app.crud.inference_config import get_config
        with SessionLocal() as config_db:
            conf = get_config(config_db)
            timeout_seconds = conf.upload_timeout_seconds if hasattr(conf, 'upload_timeout_seconds') else 300

        while True:
            if cancel_event and cancel_event.is_set():
                break

            with SessionLocal() as db:
                batch = get_batch(db, batch_id)
                if not batch:
                    break

                # 1. Fetch next READY or INTERRUPTED video
                next_video = db.query(VideoRun).filter(
                    VideoRun.batch_id == batch_id,
                    VideoRun.status.in_(["queued", "interrupted"])
                ).order_by(VideoRun.queue_position.asc()).first()

                if not next_video:
                    # 2. If no READY videos, check if we're completely finished
                    all_v = get_batch_video_runs(db, batch.id)
                    comp = sum(1 for v in all_v if v.status == "completed")
                    fail = sum(1 for v in all_v if v.status in ["failed", "failed_upload", "failed_inference"])
                    canc = sum(1 for v in all_v if v.status == "cancelled")
                    
                    if (comp + fail + canc) >= batch.total_videos:
                        break # Batch is fully processed
                    else:
                        if wait_seconds >= timeout_seconds:
                            LogService.warning(db=db, batch_id=batch.id, video_run_id=None, message="Batch timed out waiting for uploads. Finishing batch.")
                            BatchService.upload_timeout(db, batch)
                            break
                            
                        time.sleep(0.5) # Waiting for uploader to attach more files
                        wait_seconds += 0.5
                        continue
                
                wait_seconds = 0
                video_id = next_video.id
                video_idx = next_video.queue_position
                
            # Process the found video
            with SessionLocal() as db:
                batch = get_batch(db, batch_id)
                video = db.query(VideoRun).filter(VideoRun.id == video_id).first()
                if not video or video.status not in ("queued", "interrupted"):
                    continue
                
                if cancel_event and cancel_event.is_set():
                    video.status = "cancelled"
                    db.commit()
                    continue
                
                success = False
                for attempt in range(max_retries):
                    if attempt > 0:
                        LogService.warning(
                            db=db,
                            batch_id=batch.id,
                            video_run_id=video.id,
                            message=f"Retrying video... Attempt {attempt + 1}/{max_retries}"
                        )
                    
                    try:
                        success = self.run_video(
                            db=db,
                            batch=batch,
                            video=video,
                            batch_index=video_idx,
                            batch_total=batch.total_videos,
                            stream_hud=stream_hud,
                            cancel_event=cancel_event
                        )
                        if success == "CANCELLED":
                            video.status = "cancelled"
                            db.commit()
                            success = False
                            break
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"Exception during run_video: {e}\n{error_trace}")
                        LogService.error(
                            db=db,
                            batch_id=batch.id,
                            video_run_id=video.id,
                            message=f"Fatal exception during batch processing: {e}"
                        )
                        success = False
                    
                    if success or (cancel_event and cancel_event.is_set()):
                        break
                
                # Re-query all to get true counts
                all_v = get_batch_video_runs(db, batch.id)
                comp = sum(1 for v in all_v if v.status == "completed")
                fail = sum(1 for v in all_v if v.status == "failed")
                canc = sum(1 for v in all_v if v.status == "cancelled")
                proc = sum(1 for v in all_v if v.status == "processing")
                queued = sum(1 for v in all_v if v.status == "queued")
                
                BatchService.update_progress(
                    db=db,
                    batch=batch,
                    completed_videos=comp,
                    failed_videos=fail,
                )
                
                from app.services.websocket_manager import manager
                manager.send_threadsafe(batch.id, {
                    "type": "batch",
                    "status": "running",
                    "videos_queued": queued,
                    "videos_processing": proc,
                    "videos_completed": comp,
                    "videos_failed": fail,
                    "videos_cancelled": canc
                })

        with SessionLocal() as db:
            batch = get_batch(db, batch_id)

            if cancel_event and cancel_event.is_set():
                queued_videos = db.query(VideoRun).filter(
                    VideoRun.batch_id == batch_id,
                    VideoRun.status == "queued"
                ).all()
                for qv in queued_videos:
                    qv.status = "cancelled"
                db.commit()

            final_videos = get_batch_video_runs(db, batch.id)
            final_comp = sum(1 for v in final_videos if v.status == "completed")
            final_fail = sum(1 for v in final_videos if v.status == "failed")
            final_canc = sum(1 for v in final_videos if v.status == "cancelled")
            final_pend = len(final_videos) - (final_comp + final_fail + final_canc)

            db.refresh(batch)
            
            # 1. Update batch status based on video statuses
            if cancel_event and cancel_event.is_set():
                BatchService.cancel(db=db, batch=batch)
            elif final_pend == 0:
                if final_fail == len(final_videos) and len(final_videos) > 0:
                    BatchService.fail(db=db, batch=batch)
                else:
                    BatchService.complete(db=db, batch=batch)
            else:
                # If there are still pending videos, but the loop exited without cancel, it's a failure
                BatchService.fail(db=db, batch=batch)

            # 2. Gather cycles and generate JSON summary
            total_cycles = 0
            normal_ct = 0
            anomaly_ct = 0
            unknown_ct = 0
            aborted_ct = 0
            metadata_videos = []
            
            total_duration_sec = 0.0
            total_processed_frames = 0

            for v in final_videos:
                metadata_videos.append({
                    "video_uuid": Path(v.input_path).stem,
                    "original_name": v.input_filename
                })
                
                # Try to sum frames and time for FPS calculation
                if v.status in ("completed", "failed") and v.started_at and v.completed_at:
                    try:
                        start_time = datetime.fromisoformat(v.started_at)
                        end_time = datetime.fromisoformat(v.completed_at)
                        dur = (end_time - start_time).total_seconds()
                        if dur > 0:
                            total_duration_sec += dur
                            if v.total_frames:
                                total_processed_frames += v.total_frames
                    except:
                        pass
                
                cycles = get_video_cycles(db, v.id)
                total_cycles += len(cycles)
                for c in cycles:
                    if c.final_verdict == "NORMAL": normal_ct += 1
                    elif c.final_verdict == "ANOMALY": anomaly_ct += 1
                    elif c.final_verdict == "ABORTED": aborted_ct += 1
                    else: unknown_ct += 1
            
            avg_fps = (total_processed_frames / total_duration_sec) if total_duration_sec > 0 else 0
            
            summary_data = {
                "batch": f"Batch_{batch.id}",
                "date": batch.created_at[:10],
                "videos_discovered": len(final_videos),
                "completed": final_comp,
                "failed": final_fail,
                "cancelled": final_canc,
                "pending": final_pend,
                "videos_processed": final_comp + final_fail,
                "total_cycles": total_cycles,
                "normal": normal_ct,
                "anomaly": anomaly_ct,
                "unknown": unknown_ct,
                "aborted": aborted_ct,
                "processing_time_seconds": total_duration_sec,
                "average_fps": round(avg_fps, 2)
            }
            
            metadata_data = {
                "batch_uuid": Path(batch.output_path).stem if not batch.output_path.startswith("outputs") else "",
                "videos": metadata_videos
            }

            # Calculate exact failure reasons
            failed_upload = sum(1 for v in final_videos if v.status == "failed_upload")
            failed_inference = sum(1 for v in final_videos if v.status == "failed_inference")
            upload_timeout = sum(1 for v in final_videos if v.status == "upload_timeout")
            interrupted = sum(1 for v in final_videos if v.status == "interrupted")
            
            # Hardware properties
            import psutil
            import torch
            gpu_name = "None"
            cuda_ver = "None"
            peak_gpu_mb = 0
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_ver = torch.version.cuda
                peak_gpu_mb = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
            peak_ram_mb = psutil.Process().memory_info().rss // (1024 * 1024)
            
            avg_processing_time = (total_duration_sec / final_comp) if final_comp > 0 else 0

            out_path = Path(batch.output_path)
            if out_path.exists():
                with open(out_path / "summary.json", "w") as f:
                    json.dump(summary_data, f, indent=2)
                with open(out_path / "metadata.json", "w") as f:
                    json.dump(metadata_data, f, indent=2)
                
                txt_summary = f"""==================================================
Batch Summary Report
==================================================
Batch Name       : {batch.batch_name or f'Batch_{batch.id}'}
Batch ID         : {batch.id}
Date             : {batch.created_at[:10]}
Start Time       : {batch.started_at or "N/A"}
End Time         : {batch.completed_at or "N/A"}
Duration         : {total_duration_sec:.1f}s

Hardware & Environment
--------------------------------------------------
GPU              : {gpu_name}
CUDA             : {cuda_ver}
Peak GPU Memory  : {peak_gpu_mb} MB
Peak RAM         : {peak_ram_mb} MB

Versions
--------------------------------------------------
DHTX Desktop     : 2.4.1
Backend          : 1.7.3
YOLO             : 8.2
Pose             : v46

Inference Statistics
--------------------------------------------------
Videos Discovered: {len(final_videos)}
Completed        : {final_comp}
Failed Upload    : {failed_upload}
Failed Inference : {failed_inference}
Upload Timeout   : {upload_timeout}
Interrupted      : {interrupted}
Cancelled        : {final_canc}
Pending          : {final_pend}

Performance
--------------------------------------------------
Total Cycles     : {total_cycles} (Normal: {normal_ct}, Anomaly: {anomaly_ct}, Unknown: {unknown_ct}, Aborted: {aborted_ct})
Average FPS      : {avg_fps:.2f}
Avg Process Time : {avg_processing_time:.2f}s
==================================================
"""
                with open(out_path / "batch_summary.txt", "w") as f:
                    f.write(txt_summary)

            LogService.info(
                db=db,
                batch_id=batch.id,
                message="Batch Finished"
            )

            from app.services.job_session_manager import job_session_mgr
            job_session_mgr.destroy_session(batch.id)

    # def _build_command(
    #     self,
    #     *,
    #     video_path: str,
    #     output_dir: Path,
    # ):
    #     return [
    #         sys.executable,
    #         str(SCRIPT_PATH),
    #         "--video", video_path,
    #         "--model", str(SEQUENCE_MODEL),
    #         "--out_base", str(output_dir),
    #         "--yolo", str(YOLO_MODEL),
    #         "--hand_yolo", str(POSE_MODEL),
    #         "--print_summary",
    #     ]

    def _find_excel(
        self,
        output_dir: Path,
    ):
        files = list(output_dir.rglob("inspection_log.xlsx"))
        if files:
            return files[0]
        return None

    def _find_output_video(
        self,
        output_dir: Path,
    ):
        priority = [
            "*processed*.mp4",
            "*output*.mp4",
            "*.mp4",
            "*.avi",
            "*.mov",
        ]
        for pattern in priority:
            files = list(output_dir.rglob(pattern))
            if files:
                return str(files[0])
        return ""

    def run_video(
        self,
        db: Session,
        batch,
        video,
        batch_index: int = None,
        batch_total: int = None,
        stream_hud: bool = False,
        **kwargs
    ) -> bool:
        try:
            try:
                metadata = probe_video(video.input_path)
                VideoRunService.update_metadata(
                    db=db,
                    video=video,
                    width=metadata["width"],
                    height=metadata["height"],
                    fps=metadata["fps"],
                    duration_seconds=metadata["duration_seconds"],
                )
                video.total_frames = metadata["total_frames"]
            except Exception as ex:
                LogService.warning(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message=f"Unable to read video metadata : {ex}",
                )

            VideoRunService.start(
                db=db,
                video=video,
            )

            # Broadcast video_started first so frontend clears old logs BEFORE we send new ones
            fps_src = video.fps if hasattr(video, 'fps') and video.fps else 30.0
            src_w = video.width if hasattr(video, 'width') and video.width else 1920
            src_h = video.height if hasattr(video, 'height') and video.height else 1080
            
            from app.services.websocket_manager import manager
            manager.send_threadsafe(batch.id, {
                "type": "video_started",
                "filename": video.input_filename,
                "video_id": video.id,
                "total_frames": video.total_frames or 0,
                "fps": fps_src,
                "width": src_w,
                "height": src_h,
                "video_index": batch_index if batch_index is not None else 1,
                "total_videos": batch_total if batch_total is not None else 1,
                "started_at": datetime.now().isoformat() + "Z",
                "status": "RUNNING"
            })

            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message=f"Starting inference : {video.input_filename}",
            )

            output_dir = Path(batch.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message="Launching ML process in-memory...",
            )

            from app.services.model_manager import ModelManager
            models_tuple = ModelManager.get_instance().get_models()

            config_db = get_config(db)
            
            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message=(
                    f"Loaded Config - "
                    f"M1: {config_db.model1_pass_frames}/{config_db.model1_frame_count} "
                    f"| M2 Skip: {config_db.model2_start_skip_frame} "
                    f"| M2: {config_db.model2_pass_frames}/{config_db.model2_frame_count} "
                    f"| SockAbst: {config_db.socket_absent_frames} "
                    f"| SockAbort: {config_db.socket_loss_abort_frames}"
                )
            )

            pm = PluginManager()
            plugin = pm.load_plugin("dhtx_inspection")
            

            if kwargs.get("cancel_event") and kwargs.get("cancel_event").is_set():
                return "CANCELLED"
                
            # config_snap can be passed as dict
            plugin.initialize({
                "seg_net": models_tuple[0],
                "yolo_socket": models_tuple[1],
                "yolo_pose": models_tuple[2],
                "video_path": video.input_path,
                "output_dir": str(output_dir),
                "fps_src": fps_src,
                "src_w": src_w,
                "src_h": src_h,
                "model1_frame_count": config_db.model1_frame_count,
                "model1_pass_frames": config_db.model1_pass_frames,
                "model2_start_skip_frame": config_db.model2_start_skip_frame,
                "model2_frame_count": config_db.model2_frame_count,
                "model2_pass_frames": config_db.model2_pass_frames,
                "socket_absent_frames": config_db.socket_absent_frames,
                "socket_loss_abort_frames": config_db.socket_loss_abort_frames,
            })

            controller = ApplicationController(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                video_path=video.input_path,
                output_dir=str(output_dir)
            )

            returncode = 0
            is_aborted = False
            try:
                if kwargs.get("cancel_event") and kwargs.get("cancel_event").is_set():
                    is_aborted = True
                    return "CANCELLED"

                success = controller.run_inference(plugin, cancel_event=kwargs.get("cancel_event"))
                print(f"[DEBUG] ml_runner run_inference returned: {success}")
                
                if success == "CANCELLED":
                    is_aborted = True
                    print("[DEBUG] ml_runner returning CANCELLED")
                    return "CANCELLED"
                if not success:
                    is_aborted = True
                    returncode = 1
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(error_trace, file=sys.stderr)
                LogService.error(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message=f"Pipeline exception: {type(e).__name__} - {str(e)}\n{error_trace}"
                )
                is_aborted = True
                returncode = 1
            finally:
                try:
                    plugin.shutdown(aborted=is_aborted)
                except Exception as shutdown_err:
                    print(f"[WARN] Error during plugin shutdown: {shutdown_err}", file=sys.stderr)
                
                # Guaranteed memory cleanup
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            if returncode != 0:
                VideoRunService.fail(
                    db=db,
                    video=video,
                    error_message=f"ML exited with code {returncode}",
                )
                LogService.error(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message=f"Inference failed ({returncode})",
                )
                return False

            # ========================================================
            # Finalize: Parse Excel and Update SQLite (Atomic)
            # ========================================================
            try:
                excel_file = self._find_excel(output_dir)
                output_video = self._find_output_video(output_dir)
                
                if excel_file:
                    LogService.info(
                        db=db,
                        batch_id=batch.id,
                        video_run_id=video.id,
                        message="Parsing inspection_log.xlsx",
                    )
                    cycles = ExcelParser.parse(
                        db=db,
                        video_run_id=video.id,
                        excel_path=str(excel_file),
                        video_filename=video.input_filename,
                    )
                    if not cycles:
                        LogService.warning(db=db, batch_id=batch.id, video_run_id=video.id, message="Excel parsed but 0 rows matched.")
                else:
                    LogService.warning(db=db, batch_id=batch.id, video_run_id=video.id, message="inspection_log.xlsx not found.")
                
                if not output_video:
                    raise ValueError("Artifact validation failed: Processed video not found.")
                
                import os
                import cv2
                
                if os.path.getsize(output_video) == 0:
                    raise ValueError("Artifact validation failed: Processed video size is 0 bytes.")
                    
                cap_val = cv2.VideoCapture(str(output_video))
                if not cap_val.isOpened():
                    raise ValueError("Artifact validation failed: Cannot open processed video.")
                    
                ret, _ = cap_val.read()
                if not ret:
                    cap_val.release()
                    raise ValueError("Artifact validation failed: Cannot decode first frame of processed video.")
                    
                frame_count = int(cap_val.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap_val.get(cv2.CAP_PROP_FPS)
                cap_val.release()
                
                if frame_count <= 0:
                    raise ValueError("Artifact validation failed: Processed video has 0 frames.")
                if fps <= 0 or (frame_count / fps) <= 0:
                    raise ValueError("Artifact validation failed: Processed video has 0 duration.")

                VideoRunService.complete(
                    db=db,
                    video=video,
                    output_video_path=output_video,
                    excel_report_path=str(excel_file) if excel_file else None,
                )
                # VideoRunService.complete automatically performs db.commit()
                
            except Exception as finalize_ex:
                db.rollback()
                LogService.error(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message=f"Finalize failed: {finalize_ex}",
                )
                VideoRunService.fail(
                    db=db,
                    video=video,
                    error_message=f"Finalize failed: {finalize_ex}",
                )
                return False

            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message="Inference completed and validated successfully.",
            )

            return True

        except Exception as ex:
            VideoRunService.fail(
                db=db,
                video=video,
                error_message=str(ex),
            )
            LogService.error(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message=str(ex),
            )
            return False
            
        finally:
            import gc
            gc.collect()
            
            from app.core.config import config_manager
            sweep_freq = config_manager.get("gpu_options", {}).get("vram_sweep_frequency", 10)
            
            if sweep_freq > 0 and batch_index and (batch_index % sweep_freq == 0):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        LogService.info(
                            db=db,
                            batch_id=batch.id,
                            video_run_id=video.id,
                            message=f"VRAM sweep executed (Video {batch_index})"
                        )
                except ImportError:
                    pass

    # String callbacks are removed since InferenceStateMachine logs natively.




# ============================================================
# Background Entry
# ============================================================
def run_batch_inference_task(
    batch_id: int,
    stream_hud: bool = False,
):
    runner = MLRunner()
    runner.run_batch(batch_id, stream_hud=stream_hud)