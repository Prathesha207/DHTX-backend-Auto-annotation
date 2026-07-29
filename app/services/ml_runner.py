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
from app.services.inference_state_machine import InferenceStateMachine, InferenceConfigSnapshot

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
SEQUENCE_MODEL = MODEL_DIR / "best_model_finetuned_manual.pth"

# ============================================================
# ML Runner
# ============================================================

class MLRunner:

    def run_batch(self, batch_id: int, stream_hud: bool = False, cancel_event=None):
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

            videos = get_batch_video_runs(db, batch.id)
            
            for idx in range(len(videos)):
                video_id = videos[idx].id
                
                # Re-query the video run in each iteration to avoid DetachedInstanceError
                video = db.query(VideoRun).filter(VideoRun.id == video_id).first()
                if not video or video.status in ("completed", "failed", "cancelled"):
                    continue
                
                if cancel_event and cancel_event.is_set():
                    video.status = "cancelled"
                    db.commit()
                    continue
                
                success = self.run_video(
                    db=db,
                    batch=batch,
                    video=video,
                    batch_index=idx + 1,
                    batch_total=len(videos),
                    stream_hud=stream_hud,
                    cancel_event=cancel_event
                )
                
                # Re-query all to get true counts
                all_v = get_batch_video_runs(db, batch.id)
                comp = sum(1 for v in all_v if v.status == "completed")
                fail = sum(1 for v in all_v if v.status == "failed")
                
                BatchService.update_progress(
                    db=db,
                    batch=batch,
                    completed_videos=comp,
                    failed_videos=fail,
                )

            final_videos = get_batch_video_runs(db, batch.id)
            final_comp = sum(1 for v in final_videos if v.status == "completed")
            final_fail = sum(1 for v in final_videos if v.status == "failed")
            final_canc = sum(1 for v in final_videos if v.status == "cancelled")
            final_pend = len(final_videos) - (final_comp + final_fail + final_canc)

            db.refresh(batch)
            
            # 1. Update batch status based on video statuses
            if final_pend == 0:
                if final_fail > 0:
                    BatchService.fail(db=db, batch=batch)
                elif final_canc > 0:
                    batch.status = "cancelled"
                    batch.completed_at = datetime.now().isoformat()
                    db.commit()
                    db.refresh(batch)
                    from app.services.websocket_manager import manager
                    manager.send_threadsafe(batch.id, {"type": "batch", "status": "cancelled"})
                    manager.send_threadsafe(batch.id, {"type": "finished"})
                else:
                    BatchService.complete(db=db, batch=batch)
            else:
                # If there are still pending videos, but the loop exited, it's a failure
                BatchService.fail(db=db, batch=batch)

            # 2. Gather cycles and generate JSON summary
            total_cycles = 0
            normal_ct = 0
            anomaly_ct = 0
            unknown_ct = 0
            metadata_videos = []

            for v in final_videos:
                metadata_videos.append({
                    "video_uuid": Path(v.input_path).stem,
                    "original_name": v.input_filename
                })
                cycles = get_video_cycles(db, v.id)
                total_cycles += len(cycles)
                for c in cycles:
                    if c.final_verdict == "NORMAL": normal_ct += 1
                    elif c.final_verdict == "ANOMALY": anomaly_ct += 1
                    else: unknown_ct += 1
            
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
            }
            
            metadata_data = {
                "batch_uuid": Path(batch.output_path).stem if not batch.output_path.startswith("outputs") else "",
                "videos": metadata_videos
            }

            out_path = Path(batch.output_path)
            if out_path.exists():
                with open(out_path / "summary.json", "w") as f:
                    json.dump(summary_data, f, indent=2)
                with open(out_path / "metadata.json", "w") as f:
                    json.dump(metadata_data, f, indent=2)

            LogService.info(
                db=db,
                batch_id=batch.id,
                message="Batch Finished"
            )

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
            config_snap = InferenceConfigSnapshot(
                model1_frame_count=config_db.model1_frame_count,
                model1_pass_frames=config_db.model1_pass_frames,
                model2_start_skip_frame=config_db.model2_start_skip_frame,
                model2_frame_count=config_db.model2_frame_count,
                model2_pass_frames=config_db.model2_pass_frames,
                socket_absent_frames=config_db.socket_absent_frames,
                socket_loss_abort_frames=config_db.socket_loss_abort_frames,
                enable_debug_logging=config_db.enable_debug_logging,
                enable_perf_logging=config_db.enable_perf_logging,
            )

            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message=(
                    f"Loaded Config - "
                    f"M1: {config_snap.model1_pass_frames}/{config_snap.model1_frame_count} "
                    f"| M2 Skip: {config_snap.model2_start_skip_frame} "
                    f"| M2: {config_snap.model2_pass_frames}/{config_snap.model2_frame_count} "
                    f"| SockAbst: {config_snap.socket_absent_frames} "
                    f"| SockAbort: {config_snap.socket_loss_abort_frames}"
                )
            )

            state_machine = InferenceStateMachine(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                video_path=video.input_path,
                output_dir=str(output_dir),
                original_name=video.input_filename,
                models=models_tuple,
                config=config_snap,
                enable_debug=os.environ.get("ENABLE_DEBUG", "0") == "1",
                stream_hud=stream_hud,
            )

            returncode = 0
            try:
                success = state_machine.run(cancel_event=kwargs.get("cancel_event"))
                if not success:
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
                returncode = 1

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
            # Locate inspection_log.xlsx
            # ========================================================
            excel_file = self._find_excel(output_dir)

            if excel_file:
                LogService.info(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message="Parsing inspection_log.xlsx",
                )
                try:
                    cycles = ExcelParser.parse(
                        db=db,
                        video_run_id=video.id,
                        excel_path=str(excel_file),
                        video_filename=Path(video.input_path).name,
                    )
                    if not cycles:
                        LogService.warning(
                            db=db,
                            batch_id=batch.id,
                            video_run_id=video.id,
                            message=(
                                f"Excel file was parsed but 0 rows matched filename "
                                f"'{Path(video.input_path).name}'. Check the 'Video File' column in "
                                f"{excel_file} against this value."
                            ),
                        )
                    else:
                        LogService.info(
                            db=db,
                            batch_id=batch.id,
                            video_run_id=video.id,
                            message=f"Saved {len(cycles)} cycle(s) from inspection_log.xlsx",
                        )
                except Exception as ex:
                    LogService.error(
                        db=db,
                        batch_id=batch.id,
                        video_run_id=video.id,
                        message=f"Excel Parser Error : {ex}",
                    )
            else:
                LogService.warning(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message="inspection_log.xlsx not found.",
                )

            # ========================================================
            # Locate processed video
            # ========================================================
            output_video = self._find_output_video(output_dir)

            if output_video:
                LogService.info(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message=f"Output Video : {output_video}",
                )
            else:
                LogService.warning(
                    db=db,
                    batch_id=batch.id,
                    video_run_id=video.id,
                    message="Processed video not found.",
                )

            VideoRunService.complete(
                db=db,
                video=video,
                output_video_path=output_video,
                excel_report_path=str(excel_file) if excel_file else None,
            )

            LogService.info(
                db=db,
                batch_id=batch.id,
                video_run_id=video.id,
                message="Inference completed successfully.",
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
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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