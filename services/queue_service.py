import asyncio
from pathlib import Path
from fastapi.concurrency import run_in_threadpool
import logging
import shutil
import datetime

from database import database, models
from crud import crud
from services.inference_service import run_inference, request_stop_inference, is_stop_requested, get_active_video

logger = logging.getLogger(__name__)



inference_queue = asyncio.Queue()
cancelled_video_ids = set()
cancelled_batch_ids = set()


def recover_interrupted_jobs(is_shutdown: bool = False) -> list[str]:
    """
    Phase 9 — Crash recovery / Shutdown cleanup.
    Any video that was left in PROCESSING status when the backend died or is shutting down
    is reset back to QUEUED.
    Also deletes any orphaned __processing__ files on disk.
    
    Returns a list of video IDs that were recovered to QUEUED.
    """
    db = database.SessionLocal()
    enqueued_ids = []
    try:
        stuck = (
            db.query(models.Video)
            .filter(models.Video.status == "PROCESSING")
            .order_by(models.Video.uploaded_at.asc())
            .all()
        )
        for v in stuck:
            v.status = "QUEUED"
            v.processing_started_at = None
            enqueued_ids.append(v.id)
            
            # Clean up temporary in-progress cycle files for this stuck job
            if v.batch and v.batch.storage:
                out_base_dir = Path(v.batch.storage.root_path) / "processed" / v.batch.batch_date / f"batch_{v.batch.batch_number}"
                
                from utils.storage_discovery import resolve_file_location
                resolved_src = resolve_file_location(
                    stored_path=v.source_path,
                    active_root=v.batch.storage.root_path,
                    filename=v.filename,
                    subfolder=f"raw/{v.batch.batch_date}/batch_{v.batch.batch_number}"
                )
                source_path = resolved_src if (resolved_src and Path(resolved_src).is_file()) else v.source_path
                video_stem = Path(source_path).stem
                
                for f in out_base_dir.glob(f"__processing__{video_stem}_cycle*.mp4"):
                    try:
                        f.unlink()
                        print(f"🧹 Cleaned up orphaned file on recovery: {f.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to clean up {f.name}: {e}")
                        
            logger.info(f"🔁 Recovered interrupted job: {v.filename} ({v.id})")
        if stuck:
            db.commit()
            logger.info(f"✅ Recovered {len(stuck)} interrupted video(s) back to QUEUED.")
    except Exception as e:
        logger.error(f"⚠️ Job recovery failed: {e}")
    finally:
        db.close()
    
    return enqueued_ids if not is_shutdown else []

def ensure_unknown_output(source_path: str, out_base_dir: Path, filename: str) -> str | None:
    """Create a playable UNKNOWN result when the pipeline produced no cycle output."""
    unknown_dir = out_base_dir / "UNKNOWN"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    destination = unknown_dir / f"{Path(filename).stem}_UNKNOWN{source.suffix or '.mp4'}"
    if destination.exists():
        return str(destination.absolute())
    try:
        shutil.copy2(source, destination)
        return str(destination.absolute())
    except Exception as e:
        logger.error("Could not create UNKNOWN output for %s: %s", filename, e)
        return None

_async_excel_lock = None

def get_excel_lock():
    global _async_excel_lock
    if _async_excel_lock is None:
        import asyncio
        _async_excel_lock = asyncio.Lock()
    return _async_excel_lock

def update_excel_log_verdict(out_base_dir: Path, filename: str, verdict: str, output_path: str | None, only_if_missing: bool = False):
    """Create or update the batch Excel log for an aborted, failed, or fallback video."""
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        out_base_dir.mkdir(parents=True, exist_ok=True)
        excel_path = out_base_dir / "inspection_log.xlsx"
        headers = ["Sr No.", "Timestamp", "Cycle No.", "Video File", "Status", "Video Folder Saving Path"]
        fills = {
            "NORMAL": ("C6EFCE", "006100"),
            "ANOMALY": ("FFC7CE", "9C0006"),
            "ABORTED": ("FFE0B2", "B78103"),
            "UNKNOWN": ("FFEB9C", "9C6500"),
            "FAILED": ("FFC7CE", "9C0006"),
            "TIMEOUT": ("FFEB9C", "9C6500"),
        }

        if excel_path.exists():
            workbook = openpyxl.load_workbook(str(excel_path))
            sheet = workbook.active
        else:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Inspection Log"
            thin = Side(style="thin", color="BFBFBF")
            for column, header in enumerate(headers, 1):
                cell = sheet.cell(row=1, column=column, value=header)
                cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            sheet.freeze_panes = "A2"

        fill_hex, text_hex = fills.get(verdict, ("F2F2F2", "000000"))
        
        max_cycle = -1
        last_row_idx = None
        old_output_path = None
        
        for row in range(2, sheet.max_row + 1):
            if sheet.cell(row=row, column=4).value == filename:
                cycle_val = sheet.cell(row=row, column=3).value
                if isinstance(cycle_val, (int, float)):
                    if int(cycle_val) > max_cycle:
                        max_cycle = int(cycle_val)
                        last_row_idx = row
                        old_output_path = sheet.cell(row=row, column=6).value

        if last_row_idx is not None and only_if_missing:
            return

        # ALWAYS append a new row for failures to avoid destroying history
        target_row = sheet.max_row + 1

        sr_no = max((int(sheet.cell(row=r, column=1).value or 0) for r in range(2, sheet.max_row + 1)), default=0) + 1
        cycle_str = "-" if verdict in ["UNKNOWN", "FAILED", "ABORTED"] else max_cycle + 1
        
        sheet.append([sr_no, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cycle_str,
                      filename, verdict, output_path or "N/A"])
                
        status_cell = sheet.cell(row=target_row, column=5)
        status_cell.fill = PatternFill(start_color=fill_hex, end_color=fill_hex, fill_type="solid")
        status_cell.font = Font(name="Arial", size=10, bold=True, color=text_hex)

        thin = Side(style="thin", color="BFBFBF")
        for column in range(1, len(headers) + 1):
            cell = sheet.cell(row=target_row, column=column)
            if column != 5:
                cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            
            width = max(len(str(sheet.cell(row=row, column=column).value or ""))
                        for row in range(1, sheet.max_row + 1)) + 2
            sheet.column_dimensions[get_column_letter(column)].width = max(14, width)
        workbook.save(str(excel_path))
        print(f"📊 Updated Excel log row to {verdict}: {excel_path}")
    except Exception as e:
        print(f"⚠️ Could not update Excel log: {e}")

def stop_batch_inference(video_id: str | None = None, batch_id: str | None = None) -> dict:
    """
    Stops all inference for the target BATCH:
    1. Resolves batch_id (from batch_id parameter, video_id, or active video).
    2. If the currently running video belongs to this batch, halts it and routes output to ABORTED.
    3. Finds all remaining videos belonging strictly to this batch with status 'QUEUED' and marks them as 'STOPPED' (verdict 'ABORTED').
    4. PRESERVES already COMPLETED videos in this batch.
    5. PRESERVES any queued videos belonging to OTHER batches!
    """
    db = database.SessionLocal()
    cancelled_count = 0
    try:
        target_batch_id = batch_id
        active_id = get_active_video()

        if not target_batch_id:
            lookup_id = video_id or active_id
            if lookup_id:
                v = db.query(models.Video).filter(models.Video.id == lookup_id).first()
                if v:
                    target_batch_id = v.batch_id

        if not target_batch_id:
            return {"success": False, "status": "IDLE", "message": "No active batch found to stop."}

        # Track that this entire batch was stopped
        cancelled_batch_ids.add(target_batch_id)

        # 1. Halt active running video if it belongs to this batch
        if active_id:
            active_vid = db.query(models.Video).filter(models.Video.id == active_id).first()
            if active_vid and active_vid.batch_id == target_batch_id:
                cancelled_video_ids.add(active_id)
                request_stop_inference(active_id)
                crud.update_video_result(db, active_id, "STOPPING", verdict="ABORTED")
                cancelled_count += 1

        # 2. Cancel only QUEUED videos belonging strictly to this batch
        batch_queued_videos = db.query(models.Video).filter(
            models.Video.batch_id == target_batch_id,
            models.Video.status == "QUEUED"
        ).all()

        for v in batch_queued_videos:
            crud.update_video_result(db, v.id, "STOPPED", verdict="ABORTED", commit=False)
            cancelled_video_ids.add(v.id)
            cancelled_count += 1
            
        db.commit()
        batch = db.query(models.Batch).filter(models.Batch.id == target_batch_id).first()
        batch_num = batch.batch_number if batch else "?"
        print(f"🛑 Batch #{batch_num} stopped: halted active video and cancelled {cancelled_count} pending video(s).")

        return {
            "success": True,
            "batch_id": target_batch_id,
            "status": "STOPPED",
            "cancelled_count": cancelled_count,
            "message": f"Batch #{batch_num} stopped: halted active execution and cancelled {cancelled_count} pending video(s)."
        }
    finally:
        db.close()

async def inference_worker():
    print("🚀 Inference worker started and waiting for videos!")
    while True:
        video_id = await inference_queue.get()
        try:
            # Check if this queued item was cancelled while waiting
            if video_id in cancelled_video_ids:
                cancelled_video_ids.discard(video_id)
                print(f"⏩ Video {video_id} was cancelled before starting. Skipping.")
                continue

            db = database.SessionLocal()
            try:
                # Retrieve the video and its batch relationship
                video = db.query(models.Video).filter(models.Video.id == video_id).first()
                if not video:
                    print(f"⚠️ Video {video_id} not found in DB.")
                    continue

                if video.status in ("STOPPED", "FAILED", "ABORTED") or (video.batch_id and video.batch_id in cancelled_batch_ids):
                    print(f"⏩ Video {video_id} status is {video.status} (or batch cancelled). Skipping.")
                    continue
                
                batch = video.batch
                storage = batch.storage if (batch and batch.storage) else None
                active_storage = crud.get_active_storage(db)
                storage_root = Path(active_storage.root_path if active_storage else (storage.root_path if storage else "./storage"))
                
                from utils.storage_discovery import resolve_file_location
                resolved_src = resolve_file_location(
                    stored_path=video.source_path,
                    active_root=str(storage_root),
                    filename=video.filename,
                    subfolder=f"raw/{batch.batch_date}/batch_{batch.batch_number}" if batch else ""
                )
                source_path = resolved_src if (resolved_src and Path(resolved_src).is_file()) else video.source_path

                # Construct the processed batch output directory
                if batch:
                    out_base_dir = storage_root / "processed" / batch.batch_date / f"batch_{batch.batch_number}"
                else:
                    out_base_dir = storage_root / "processed" / "unknown_batch"
                out_base_dir.mkdir(parents=True, exist_ok=True)
                
                # Clean up any orphaned processing temp files from previous interrupted runs
                video_stem = Path(source_path).stem
                for f in out_base_dir.glob(f"__processing__{video_stem}_cycle*.mp4"):
                    try:
                        f.unlink()
                        print(f"🧹 Cleaned up orphaned file: {f.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to clean up {f.name}: {e}")

                print(f"▶️ Starting inference for {video.filename}")
                crud.update_video_result(db, video_id, "PROCESSING")

                # Track when processing started
                import datetime as _dt
                video.processing_started_at = _dt.datetime.utcnow()
                db.commit()
                
                # Fetch the active model setting
                val_setting = crud.get_active_model_setting(db)
                model_settings = {
                    "m1_total": val_setting.m1_total,
                    "m1_pass": val_setting.m1_pass,
                    "m2_total": val_setting.m2_total,
                    "m2_pass": val_setting.m2_pass,
                }
                
                # Execute inference in threadpool with stop support
                result_dict = await run_in_threadpool(
                    run_inference,
                    video_path=source_path,
                    out_base_dir=str(out_base_dir),
                    video_id=video_id,
                    model_settings=model_settings
                )
                
                completed_normally = result_dict.get("completed", False)
                final_verdict = result_dict.get("verdict", "UNKNOWN")
                final_output = result_dict.get("output_path")

                # Check if stop was requested during execution
                if not completed_normally or video_id in cancelled_video_ids:
                    reason = result_dict.get("reason")
                    if reason == "WRITER_FAILED":
                        raise RuntimeError("VideoWriter failed to write output frames.")
                        
                    cancelled_video_ids.discard(video_id)
                    
                    # 1. Create ABORTED folder
                    aborted_dir = out_base_dir / "ABORTED"
                    aborted_dir.mkdir(parents=True, exist_ok=True)
                    
                    aborted_video_path = None
                    video_stem = Path(source_path).stem

                    # Move any temporary in-progress file to ABORTED
                    for temp_mp4 in out_base_dir.glob(f"__processing__{video_stem}_cycle*.mp4"):
                        try:
                            import re
                            cycle_match = re.search(r"cycle(\d+)", temp_mp4.name)
                            cycle_str = cycle_match.group(1) if cycle_match else "001"
                            target_dest = aborted_dir / f"{video_stem}_ABORTED_cycle{cycle_str}.mp4"
                            
                            shutil.move(str(temp_mp4), str(target_dest))
                            if not aborted_video_path:
                                aborted_video_path = str(target_dest.absolute())
                        except Exception:
                            pass

                    crud.update_video_result(db, video_id, "STOPPED", verdict="ABORTED", output_path=aborted_video_path)
                    async with get_excel_lock():
                        await run_in_threadpool(update_excel_log_verdict, out_base_dir, video.filename, "ABORTED", aborted_video_path)
                    print(f"🛑 Inference stopped for {video.filename} -> saved to: {aborted_video_path}")
                    continue
                
                # Check DB for race condition where API STOP request occurred during final cycle completion
                db.refresh(video)
                if video.status in ("STOPPING", "STOPPED") or is_stop_requested(video_id) or video_id in cancelled_video_ids:
                    print(f"⚠️ Caught late stop request for {video.filename}. Ignoring COMPLETED status.")
                    continue
                
                if final_verdict == "UNKNOWN" and not final_output:
                    final_output = ensure_unknown_output(source_path, out_base_dir, video.filename)
                    async with get_excel_lock():
                        await run_in_threadpool(update_excel_log_verdict, out_base_dir, video.filename, "UNKNOWN", final_output, True)

                # Ensure atomic completion check
                rows = db.query(models.Video).filter(
                    models.Video.id == video_id,
                    models.Video.status == "PROCESSING"
                ).update({
                    "status": "COMPLETED", 
                    "verdict": final_verdict, 
                    "output_path": final_output
                })
                
                if rows == 0:
                    print(f"⚠️ Video {video.filename} was no longer processing. Ignoring COMPLETED update.")
                else:
                    db.commit()
                    print(f"✅ Inference finished for {video.filename} -> {final_verdict} ({final_output})")

            except Exception as e:
                print(f"❌ Inference failed for {video_id}: {e}")
                
                is_abort = video_id in cancelled_video_ids or is_stop_requested(video_id)
                
                # Ensure the DB status accurately checks STOPPING as well to handle the race condition safely
                current_status = db.query(models.Video.status).filter(models.Video.id == video_id).scalar()
                if current_status in ("STOPPING", "STOPPED") or is_abort:
                    new_status = "STOPPED"
                    new_verdict = "ABORTED"
                else:
                    new_status = "FAILED"
                    new_verdict = "UNKNOWN"
                
                fallback_path = None
                if 'source_path' in locals() and 'out_base_dir' in locals() and source_path and out_base_dir and 'video' in locals() and video:
                    video_stem = Path(source_path).stem
                    for temp_mp4 in out_base_dir.glob(f"__processing__{video_stem}_cycle*.mp4"):
                        try:
                            temp_mp4.unlink()
                        except Exception:
                            pass
                    
                    if new_status != "STOPPED":
                        fallback_path = ensure_unknown_output(source_path, out_base_dir, video.filename)
                        async with get_excel_lock():
                            await run_in_threadpool(update_excel_log_verdict, out_base_dir, video.filename, "UNKNOWN", fallback_path)
                    
                crud.update_video_result(db, video_id, new_status, verdict=new_verdict, output_path=fallback_path)
            finally:
                db.close()
        finally:
            inference_queue.task_done()
