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

INFERENCE_TIMEOUT_SECONDS = 60 * 60  # 60 minutes (safe for CPU inference)
CANCEL_GRACE_SECONDS = 3.0
inference_queue = asyncio.Queue()
cancelled_video_ids: set[str] = set()
cancelled_batch_ids: set[str] = set()


# ── Ensure Storage Location ───────────────────────────
# This function/class is responsible for ensure storage location operations.
def ensure_storage_location(db) -> str:
    """Ensure an active storage entry exists in DB and return its root_path."""
    active = crud.get_active_storage(db)
    if active and is_path_writable(active.root_path):
        return active.root_path
    
    writable_dir = get_writable_default_location()
    new_storage = crud.create_or_update_storage(db, root_path=writable_dir, is_default=True)
    return new_storage.root_path


# ── Recover Interrupted Jobs ──────────────────────────
# This function/class is responsible for recover interrupted jobs operations.
def recover_interrupted_jobs(is_shutdown: bool = False):
    """
    Recovers videos stuck in 'PROCESSING' state across server restarts.
    """
    db = database.SessionLocal()
    try:
        stuck = db.query(models.Video).filter(models.Video.status == "PROCESSING").all()
        enqueued_ids = []
        for v in stuck:
            v.status = "QUEUED"
            enqueued_ids.append(v.id)
            
            # Clean up orphaned temporary files for this video
            if v.batch:
                active_storage = crud.get_active_storage(db)
                storage_root = Path(active_storage.root_path if active_storage else "./storage")
                out_base_dir = storage_root / "processed" / v.batch.batch_date / f"batch_{v.batch.batch_number}"
                video_stem = Path(v.filename).stem
                for f in out_base_dir.glob(f"__processing__{video_stem}_cycle*.mp4"):
                    try:
                        f.unlink()
                        print(f"🧹 Cleaned up orphaned file: {f.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to clean up {f.name}: {e}")
                        
            logger.info(f"🔁 Recovered interrupted job: {v.filename} ({v.id})")
        if stuck:
            db.commit()
            if not is_shutdown:
                for vid in enqueued_ids:
                    inference_queue.put_nowait(vid)
            logger.info(f"✅ Recovered {len(stuck)} interrupted video(s) back to QUEUED.")
    except Exception as e:
        logger.error(f"⚠️ Job recovery failed: {e}")
    finally:
        db.close()

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


def update_excel_log_verdict(out_base_dir: Path, filename: str, verdict: str, output_path: str | None, only_if_missing: bool = True):
    """
    Create or update the batch Excel log for an aborted, failed, or fallback video.
    PRESERVES existing cycle rows logged by the ML pipeline so valid cycles are never overwritten.
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        out_base_dir.mkdir(parents=True, exist_ok=True)
        excel_path = out_base_dir / "inspection_log.xlsx"
        headers = ["Sr No.", "Timestamp", "Cycle No.", "Video File", "Status", "Video Folder Saving Path"]
        fills = {
            "NORMAL":  ("C6EFCE", "006100"),
            "ANOMALY": ("FFC7CE", "9C0006"),
            "ABORTED": ("FFE0B2", "B78103"),
            "UNKNOWN": ("FFEB9C", "9C6500"),
            "FAILED":  ("FFC7CE", "9C0006"),
            "TIMEOUT": ("FFEB9C", "9C6500"),
        }

        thin = Side(style="thin", color="BFBFBF")
        bdr = Border(left=thin, right=thin, top=thin, bottom=thin)

        if excel_path.exists():
            workbook = openpyxl.load_workbook(str(excel_path))
            sheet = workbook.active
        else:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Inspection Log"
            for column, header in enumerate(headers, 1):
                cell = sheet.cell(row=1, column=column, value=header)
                cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = bdr
            sheet.row_dimensions[1].height = 32
            sheet.freeze_panes = "A2"

        # Check if this filename already has valid entries in the sheet
        has_existing_rows = False
        for row in range(2, sheet.max_row + 1):
            if sheet.cell(row=row, column=4).value == filename:
                has_existing_rows = True
                break

        if has_existing_rows and only_if_missing:
            # Valid cycle row(s) already exist from the ML pipeline — do not overwrite them
            return

        fill_hex, text_hex = fills.get(verdict, ("F2F2F2", "000000"))

        if not has_existing_rows:
            sr_no = max(1, sheet.max_row)
            row_vals = [
                sr_no,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                0,
                filename,
                verdict,
                output_path or "N/A"
            ]
            sheet.append(row_vals)
            curr_row = sheet.max_row

            for col_idx in range(1, len(headers) + 1):
                c = sheet.cell(row=curr_row, column=col_idx)
                c.font = Font(name="Arial", size=10)
                c.border = bdr
                c.alignment = Alignment(
                    horizontal="center" if col_idx in (1, 2, 3, 5) else "left",
                    vertical="center"
                )
                if col_idx == 5:
                    c.fill = PatternFill(start_color=fill_hex, end_color=fill_hex, fill_type="solid")
                    c.font = Font(name="Arial", size=10, bold=True, color=text_hex)

        for column in range(1, len(headers) + 1):
            width = max(len(str(sheet.cell(row=row, column=column).value or ""))
                        for row in range(1, sheet.max_row + 1)) + 4
            sheet.column_dimensions[get_column_letter(column)].width = max(14, width)

        workbook.save(str(excel_path))
        print(f"📊 Excel log verified/updated: {excel_path}")
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
                try:
                    completed_normally = await asyncio.wait_for(
                        run_in_threadpool(
                            run_inference,
                            video_path=source_path,
                            out_base_dir=str(out_base_dir),
                            video_id=video_id,
                            model_settings=model_settings
                        ),
                        timeout=INFERENCE_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"⏱️ Inference for {video.filename} exceeded "
                        f"{INFERENCE_TIMEOUT_SECONDS}s — marking FAILED and continuing."
                    )
                    request_stop_inference(video_id)
                    timeout_output = ensure_unknown_output(source_path, out_base_dir, video.filename)
                    crud.update_video_result(db, video_id, "FAILED", verdict="TIMEOUT", output_path=timeout_output)
                    update_excel_log_verdict(out_base_dir, video.filename, "TIMEOUT", timeout_output)

                    # Wait for inference thread to actually release context.
                    # If it doesn't release within CONTEXT_RELEASE_TIMEOUT, we must NOT
                    # start the next video — two concurrent inference threads on the
                    # same GPU would violate the sequential queue guarantee.
                    from services.inference_service import get_context
                    CONTEXT_RELEASE_TIMEOUT = 15.0   # seconds
                    POLL_INTERVAL = 0.1
                    released = False
                    for _ in range(int(CONTEXT_RELEASE_TIMEOUT / POLL_INTERVAL)):
                        await asyncio.sleep(POLL_INTERVAL)
                        if get_context(video_id) is None:
                            released = True
                            break

                    if not released:
                        logger.critical(
                            "⛔ Inference thread for %s did not release context within %.0fs. "
                            "Worker will NOT start the next video to prevent GPU overlap. "
                            "Manual restart required.",
                            video.filename, CONTEXT_RELEASE_TIMEOUT,
                        )
                        # Mark all remaining QUEUED videos in this batch as FAILED
                        remaining = db.query(models.Video).filter(
                            models.Video.batch_id == video.batch_id,
                            models.Video.status == "QUEUED"
                        ).all()
                        for rv in remaining:
                            rv.status = "FAILED"
                            rv.verdict = "WORKER_STALLED"
                        db.commit()
                        # Break out of the worker loop — the worker is dead.
                        # The only recovery path is a server restart, which will
                        # trigger recover_interrupted_jobs().
                        return
                    continue

                # Check if stop was requested during execution
                if not completed_normally or video_id in cancelled_video_ids:
                    cancelled_video_ids.discard(video_id)
                    
                    # 1. Create ABORTED folder
                    aborted_dir = out_base_dir / "ABORTED"
                    aborted_dir.mkdir(parents=True, exist_ok=True)
                    
                    aborted_video_path = None
                    video_stem = Path(video.filename).stem
                    target_dest = aborted_dir / f"{video_stem}_ABORTED_cycle001.mp4"

                    # 2. Check if file was saved in UNKNOWN, NORMAL, or ANOMALY folder and relocate to ABORTED
                    for folder_name in ["UNKNOWN", "NORMAL", "ANOMALY"]:
                        sub_dir = out_base_dir / folder_name
                        if sub_dir.exists():
                            for mp4 in sub_dir.glob(f"{video_stem}_*.mp4"):
                                shutil.move(str(mp4), str(target_dest))
                                aborted_video_path = str(target_dest.absolute())
                                break
                        if aborted_video_path:
                            break

                    # 3. Check if temporary in-progress file (__processing__*.mp4) exists
                    if not aborted_video_path:
                        for temp_mp4 in out_base_dir.glob(f"*processing*{video_stem}*.mp4"):
                            try:
                                shutil.move(str(temp_mp4), str(target_dest))
                                aborted_video_path = str(target_dest.absolute())
                                break
                            except Exception:
                                pass

                    crud.update_video_result(db, video_id, "STOPPED", verdict="ABORTED", output_path=aborted_video_path)
                    update_excel_log_verdict(out_base_dir, video.filename, "ABORTED", aborted_video_path)
                    print(f"🛑 Inference stopped for {video.filename} -> saved to: {aborted_video_path}")
                    continue
                
                # Phase 2: Dynamically discover the ML result for THIS specific video
                video_stem = Path(video.filename).stem

                anomaly_files = list((out_base_dir / "ANOMALY").glob(f"{video_stem}_*.mp4")) if (out_base_dir / "ANOMALY").exists() else []
                normal_files  = list((out_base_dir / "NORMAL").glob(f"{video_stem}_*.mp4")) if (out_base_dir / "NORMAL").exists() else []
                unknown_files = list((out_base_dir / "UNKNOWN").glob(f"{video_stem}_*.mp4")) if (out_base_dir / "UNKNOWN").exists() else []

                if anomaly_files:
                    verdict = "ANOMALY"
                    output_path = str(anomaly_files[0].absolute())
                elif normal_files:
                    verdict = "NORMAL"
                    output_path = str(normal_files[0].absolute())
                elif unknown_files:
                    verdict = "UNKNOWN"
                    output_path = str(unknown_files[0].absolute())
                else:
                    verdict = "UNKNOWN"
                    output_path = ensure_unknown_output(source_path, out_base_dir, video.filename)
                    # When ML produced 0 cycles, record the fallback UNKNOWN row in Excel
                    update_excel_log_verdict(out_base_dir, video.filename, "UNKNOWN", output_path, only_if_missing=True)

                crud.update_video_result(db, video_id, "COMPLETED", verdict, output_path)
                print(f"✅ Inference finished for {video.filename} -> {verdict} ({output_path})")

            except Exception as e:
                print(f"❌ Inference failed for {video_id}: {e}")
                fallback_path = None
                if 'source_path' in locals() and 'out_base_dir' in locals() and source_path and out_base_dir and 'video' in locals() and video:
                    fallback_path = ensure_unknown_output(source_path, out_base_dir, video.filename)
                    update_excel_log_verdict(out_base_dir, video.filename, "UNKNOWN", fallback_path)
                crud.update_video_result(db, video_id, "FAILED", verdict="UNKNOWN", output_path=fallback_path)
            finally:
                db.close()
        finally:
            inference_queue.task_done()
