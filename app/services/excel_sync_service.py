import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import shutil
import zipfile

from app.services.log_service import LogService
from app.database.database import SessionLocal

_EXCEL_COLUMNS = [
    ("sr_no",             "Sr No."),
    ("timestamp",         "Timestamp"),
    ("cycle_no",          "Cycle No."),
    ("filename",          "Video File"),
    ("final_verdict",     "Status"),
    ("output_path",       "Video Folder Saving Path"),
]

_VERDICT_FILLS = {
    "NORMAL": "C6EFCE", 
    "ANOMALY": "FFC7CE",
    "PARTIAL": "FFEB9C", 
    "N/A": "F2F2F2", 
    "UNKNOWN": "FFEB9C"
}

class ExcelSyncService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExcelSyncService, cls).__new__(cls)
                cls._instance.running = False
                cls._instance.thread = None
                cls._instance.pending_dirs = set()
                cls._instance.dir_lock = threading.Lock()
                cls._instance._last_lock_logs = {}
        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def flush(self, excel_dir: str = None):
        """Force a synchronous flush of pending rows."""
        dirs_to_process = []
        if excel_dir:
            dirs_to_process = [excel_dir]
        else:
            with self.dir_lock:
                dirs_to_process = list(self.pending_dirs)
                
        for d in dirs_to_process:
            self._try_sync(d)

    def queue_row(self, batch_id: int, video_run_id: int, excel_dir: str, run_metrics: dict):
        """O(1) append to JSONL queue."""
        Path(excel_dir).mkdir(parents=True, exist_ok=True)
        jsonl_path = os.path.join(excel_dir, "inspection_log.jsonl")
        
        row_data = {
            "batch_id": batch_id,
            "video_run_id": video_run_id,
            "metrics": run_metrics
        }
        
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row_data) + "\n")
            
        with self.dir_lock:
            self.pending_dirs.add(excel_dir)

    def _sync_loop(self):
        # Crash recovery: scan outputs for existing jsonl files
        try:
            base_dir = Path(__file__).resolve().parents[3]
            outputs_dir = os.path.join(base_dir, "outputs")
            if os.path.exists(outputs_dir):
                for root, dirs, files in os.walk(outputs_dir):
                    if "inspection_log.jsonl" in files:
                        with self.dir_lock:
                            self.pending_dirs.add(root)
        except Exception as e:
            print(f"[WARN] Failed to scan for JSONL files: {e}")

        while self.running:
            dirs_to_process = []
            with self.dir_lock:
                dirs_to_process = list(self.pending_dirs)
            
            for excel_dir in dirs_to_process:
                self._try_sync(excel_dir)
            
            time.sleep(2.0)

    def _try_sync(self, excel_dir: str):
        jsonl_path = os.path.join(excel_dir, "inspection_log.jsonl")
        if not os.path.exists(jsonl_path):
            with self.dir_lock:
                self.pending_dirs.discard(excel_dir)
            return

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return
            
        if not lines:
            with self.dir_lock:
                self.pending_dirs.discard(excel_dir)
            return
            
        rows = []
        for line in lines:
            if not line.strip(): continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
                
        excel_path = os.path.join(excel_dir, "inspection_log.xlsx")
        
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        except ImportError:
            return
            
        def _create_new_workbook():
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Inspection Log"
            thin = Side(style="thin", color="BFBFBF")
            bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
            for ci, (_, header) in enumerate(_EXCEL_COLUMNS, 1):
                cell           = ws.cell(row=1, column=ci, value=header)
                cell.font      = Font(name="Arial", size=11, bold=True, color="FFFFFF")
                cell.fill      = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border    = bdr
            ws.row_dimensions[1].height = 32
            ws.freeze_panes = "A2"
            return wb, ws, 1

        try:
            if os.path.exists(excel_path):
                try:
                    wb = openpyxl.load_workbook(excel_path)
                    ws = wb.active
                    
                    last_valid_row = 1
                    for row in range(2, ws.max_row + 1):
                        val = ws.cell(row=row, column=1).value
                        if val is not None and str(val).strip():
                            last_valid_row = row
                            
                    data_rows_in_excel = last_valid_row - 1
                    next_sr = data_rows_in_excel + 1
                except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
                    if isinstance(exc, PermissionError):
                        raise
                    broken_path = os.path.join(excel_dir, f"inspection_log.broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                    try:
                        shutil.move(excel_path, broken_path)
                    except OSError:
                        pass
                    wb, ws, next_sr = _create_new_workbook()
            else:
                wb, ws, next_sr = _create_new_workbook()
                data_rows_in_excel = 0
                
            missing_rows = rows[data_rows_in_excel:]
            
            if not missing_rows:
                with self.dir_lock:
                    self.pending_dirs.discard(excel_dir)
                return
                
            start_row = data_rows_in_excel + 2
            current_row = start_row
            
            with SessionLocal() as db:
                batch_id = missing_rows[0].get("batch_id")
                video_run_id = missing_rows[0].get("video_run_id")
                # If we were previously locked, log that Excel is available now
                if batch_id in self._last_lock_logs:
                    LogService.info(db, batch_id, video_run_id, "Excel unlocked.")
                    LogService.info(db, batch_id, video_run_id, f"Flushing {len(missing_rows)} buffered rows.")
                    del self._last_lock_logs[batch_id]
                
            for row_data in missing_rows:
                metrics = row_data["metrics"]
                col_idx = 1
                for key, _ in _EXCEL_COLUMNS:
                    val = ""
                    if key == "sr_no":
                        val = next_sr
                    elif key == "cycle_no":
                        val = int(metrics.get(key, 0))
                    else:
                        val = metrics.get(key, "N/A")
                        
                    cell = ws.cell(row=current_row, column=col_idx, value=val)
                    if key == "final_verdict":
                        verdict = str(val).strip().upper()
                        if verdict in _VERDICT_FILLS:
                            cell.fill = PatternFill(start_color=_VERDICT_FILLS[verdict], end_color=_VERDICT_FILLS[verdict], fill_type="solid")
                            
                    col_idx += 1
                next_sr += 1
                current_row += 1
                
            thin = Side(style="thin", color="BFBFBF")
            bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
            for r in ws.iter_rows(min_row=start_row, max_row=current_row-1, min_col=1, max_col=len(_EXCEL_COLUMNS)):
                for cell in r:
                    cell.border = bdr
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    
            from openpyxl.utils import get_column_letter
            col_widths = {1: 10, 2: 25, 3: 12, 4: 50, 5: 20, 6: 80}
            for col_idx, width in col_widths.items():
                ws.column_dimensions[get_column_letter(col_idx)].width = width
                    
            wb.save(excel_path)
            
            with SessionLocal() as db:
                batch_id = missing_rows[0].get("batch_id")
                video_run_id = missing_rows[0].get("video_run_id")
                LogService.info(db, batch_id, video_run_id, "Synchronization complete.")
                
                # Verification routine
                from app.crud.cycle import get_cycles_by_batch
                db_cycles = get_cycles_by_batch(db, batch_id)
                db_count = len(db_cycles)
                json_count = len(rows)
                excel_count = current_row - 2  # start_row = 2, so row 2 is index 1
                
                if excel_count == json_count == db_count:
                    LogService.info(db, batch_id, video_run_id, f"Verification passed: Queued rows == 0. Excel ({excel_count}) == JSONL ({json_count}) == DB ({db_count}).")
                else:
                    LogService.info(db, batch_id, video_run_id, f"Verification warning: Excel ({excel_count}), JSONL ({json_count}), DB ({db_count}).")
            
            with self.dir_lock:
                self.pending_dirs.discard(excel_dir)
                
        except PermissionError:
            self._log_lock(rows)
        except Exception as e:
            if "WinError 32" in str(e) or "Permission denied" in str(e) or "used by another process" in str(e):
                self._log_lock(rows)
            else:
                import traceback
                print(f"[ERROR] Excel Sync failed: {e}\n{traceback.format_exc()}")

    def _log_lock(self, rows):
        """Prevent spamming the log by only logging once every ~10 seconds per locked file."""
        if not rows: return
        batch_id = rows[-1].get("batch_id")
        missing_count = len(rows)
        
        now = time.time()
        if now - self._last_lock_logs.get(batch_id, 0) > 10.0:
            self._last_lock_logs[batch_id] = now
            with SessionLocal() as db:
                video_run_id = rows[-1].get("video_run_id")
                LogService.info(db, batch_id, video_run_id, "Excel locked.")
                LogService.info(db, batch_id, video_run_id, f"Buffering row #{missing_count}.")
