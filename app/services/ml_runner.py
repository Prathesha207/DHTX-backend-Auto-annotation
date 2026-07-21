import os
import sys
import threading
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session
from app.database.database import SessionLocal

from app.crud.batch import get_batch
from app.crud.video_run import get_batch_video_runs
from app.crud.cycle import get_video_cycles
import json

from app.services.batch_service import BatchService
from app.services.video_run_service import VideoRunService
from app.services.log_service import LogService

from app.services.excel_parser import ExcelParser
from app.services.progress_parser import ProgressParser
from app.services.status_parser import StatusParser
from app.services.frame_parser import FrameParser
from app.services.inference_engine import InferenceEngine
from app.video_metadata import probe_video

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "ml"
SCRIPT_PATH = BASE_DIR / "models" / "inference_video_full_detection.py"
YOLO_MODEL = MODEL_DIR / "best.pt"
POSE_MODEL = MODEL_DIR / "yolov8n-pose.pt"
SEQUENCE_MODEL = MODEL_DIR / "best_model_finetuned_manual.pth"

# ============================================================
# ML Runner
# ============================================================

class MLRunner:

    def run_batch(self, batch_id: int):
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
            completed = 0
            failed = 0

            for video in videos:
                success = self.run_video(
                    db=db,
                    batch=batch,
                    video=video,
                )
                if success:
                    completed += 1
                else:
                    failed += 1

                BatchService.update_progress(
                    db=db,
                    batch=batch,
                    completed_videos=completed,
                    failed_videos=failed,
                )

            if failed == 0:
                BatchService.complete(db=db, batch=batch)
            else:
                BatchService.fail(db=db, batch=batch)

            # Generate summary.json and metadata.json
            total_cycles = 0
            normal_ct = 0
            anomaly_ct = 0
            unknown_ct = 0
            metadata_videos = []

            for v in videos:
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
                "videos_processed": completed + failed,
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
            import models.inference_video_full_detection as inf_mod
            
            models_tuple = ModelManager.get_instance().get_models()

            def on_message_callback(msg):
                self._handle_inference_message(msg, batch.id, video.id)

            returncode = 0
            try:
                inf_mod.run_single_video(
                    video_path=video.input_path,
                    seg_model_path="", 
                    out_base=str(output_dir),
                    yolo_socket_path="",
                    hand_pose_path="",
                    print_summary=True,
                    render_mode="frontend",
                    original_name=video.input_filename,
                    models=models_tuple,
                    on_message=on_message_callback
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
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
                    ExcelParser.parse(
                        db=db,
                        video_run_id=video.id,
                        excel_path=str(excel_file),
                        video_filename=video.input_filename,
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

    # ============================================================
    # Read stdout
    # ============================================================
    def _handle_inference_message(self, line: str, batch_id: int, video_run_id: int):
        try:
            line = line.strip()
            if not line:
                return
            
            with SessionLocal() as thread_db:
                video = get_batch_video_runs(thread_db, batch_id)
                video_obj = next((v for v in video if v.id == video_run_id), None)
                
                if video_obj:
                    ProgressParser.parse(
                        db=thread_db,
                        line=line,
                        video=video_obj,
                    )

                is_status_line = StatusParser.parse(
                    line=line,
                    batch_id=batch_id,
                    video_id=video_run_id,
                )

                is_frame_line = FrameParser.parse(
                    line=line,
                    batch_id=batch_id,
                    video_id=video_run_id,
                )

                if not is_status_line and not is_frame_line:
                    LogService.info(
                        db=thread_db,
                        batch_id=batch_id,
                        video_run_id=video_run_id,
                        message=line,
                    )
        except Exception as e:
            with SessionLocal() as thread_db:
                LogService.error(
                    db=thread_db,
                    batch_id=batch_id,
                    video_run_id=video_run_id,
                    message=f"Error in inference callback: {e}",
                )




# ============================================================
# Background Entry
# ============================================================
def run_batch_inference_task(
    batch_id: int,
):
    runner = MLRunner()
    runner.run_batch(batch_id)