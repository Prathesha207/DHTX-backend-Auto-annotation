import threading
import time
import psutil
import os

class HealthMonitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _monitor_loop(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                cpu_percent = process.cpu_percent()
                mem_info = process.memory_info()
                ram_mb = mem_info.rss / (1024 * 1024)
                
                # Try getting GPU VRAM if torch is loaded
                vram_mb = "N/A"
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_mb = f"{torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB"
                except Exception:
                    pass
                
                open_handles = len(process.open_files())
                threads = process.num_threads()
                
                print(f"[HEALTH MONITOR] CPU: {cpu_percent}% | RAM: {ram_mb:.1f}MB | VRAM: {vram_mb} | Handles: {open_handles} | Threads: {threads}")
                
            except Exception as e:
                print(f"[HEALTH MONITOR ERROR] {e}")
            
            time.sleep(self.interval)

health_monitor = HealthMonitor(interval=60)
