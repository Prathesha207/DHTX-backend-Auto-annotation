import threading
import psutil
import gc
import logging
import os

logger = logging.getLogger("LeakDetector")

class LeakDetector:
    def __init__(self):
        self.baseline_threads = []
        self.baseline_files = []
        self.baseline_ram = 0
        self.peak_ram = 0

    def capture_baseline(self):
        self.baseline_threads = [t.name for t in threading.enumerate()]
        proc = psutil.Process(os.getpid())
        try:
            self.baseline_files = [f.path for f in proc.open_files()]
        except Exception:
            self.baseline_files = []
        self.baseline_ram = proc.memory_info().rss / (1024 * 1024)
        self.peak_ram = self.baseline_ram
        logger.info(f"Captured baseline: {len(self.baseline_threads)} threads, {len(self.baseline_files)} open files, {self.baseline_ram:.2f} MB RAM")

    def track_peak(self):
        proc = psutil.Process(os.getpid())
        ram = proc.memory_info().rss / (1024 * 1024)
        if ram > self.peak_ram:
            self.peak_ram = ram

    def check_leaks(self):
        success = True
        
        # Thread check
        current_threads = [t.name for t in threading.enumerate()]
        leaked_threads = set(current_threads) - set(self.baseline_threads)
        if leaked_threads:
            logger.error(f"Thread leak detected! New threads: {leaked_threads}")
            success = False
            
        # File handle check
        proc = psutil.Process(os.getpid())
        try:
            current_files = [f.path for f in proc.open_files()]
            leaked_files = set(current_files) - set(self.baseline_files)
            # Filter out known safe files like test.db
            leaked_files = {f for f in leaked_files if not f.endswith((".db", ".db-shm", ".db-wal"))}
            if leaked_files:
                logger.error(f"File handle leak detected! New files: {leaked_files}")
                success = False
        except Exception as e:
            logger.error(f"Could not check open files: {e}")
            
        # RAM check
        current_ram = proc.memory_info().rss / (1024 * 1024)
        logger.info(f"RAM Status: Baseline={self.baseline_ram:.2f}MB, Peak={self.peak_ram:.2f}MB, Current={current_ram:.2f}MB")
        if current_ram - self.baseline_ram > 50:
            logger.error(f"Significant RAM leak! Grew by {current_ram - self.baseline_ram:.2f} MB")
            success = False
            
        # GC objects
        gc.collect()
        
        # GPU check (if torch is used in the same process)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024*1024)
                logger.info(f"GPU Allocated: {allocated:.2f}MB")
                if allocated > 100: # Threshold for acceptable baseline
                    logger.error(f"GPU memory leak detected! {allocated:.2f}MB still allocated after empty_cache.")
                    success = False
        except ImportError:
            pass

        return success
