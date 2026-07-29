"""
Lightweight per-frame performance monitor.

Usage:
    perf = PerfMonitor()
    perf.start_section("model1")
    ... run socket detection ...
    perf.end_section("model1")
    perf.start_section("model2")
    ... run tube detection ...
    perf.end_section("model2")
    breakdown = perf.frame_summary()  # {"model1": 12.3, "model2": 45.6, ...} ms
"""

import time
from collections import defaultdict


class PerfMonitor:

    def __init__(self):
        self._section_starts: dict[str, float] = {}
        self._section_totals: dict[str, float] = defaultdict(float)
        self._frame_start: float | None = None
        self._frame_times: list[float] = []

    # ── Frame-level bookkeeping ──────────────────────────────────

    def start_frame(self):
        self._section_totals.clear()
        self._frame_start = time.perf_counter()

    def end_frame(self) -> float:
        """Returns total frame time in milliseconds."""
        if self._frame_start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self._frame_times.append(elapsed_ms)
        self._frame_start = None
        return elapsed_ms

    # ── Section-level bookkeeping ────────────────────────────────

    def start_section(self, name: str):
        self._section_starts[name] = time.perf_counter()

    def end_section(self, name: str) -> float:
        """Returns section time in milliseconds."""
        start = self._section_starts.pop(name, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._section_totals[name] += elapsed_ms
        return elapsed_ms

    # ── Reporting ────────────────────────────────────────────────

    def frame_summary(self) -> dict:
        """
        Section breakdown for the current frame, in ms.
        Call after end_frame().
        """
        return {k: round(v, 2) for k, v in self._section_totals.items()}

    def fps_ema(self, alpha: float = 0.12) -> float:
        """Exponential moving average FPS from frame times."""
        if not self._frame_times:
            return 0.0
        # Use only recent frames for EMA
        fps = 1000.0 / max(self._frame_times[-1], 0.001)
        if len(self._frame_times) < 2:
            return fps
        prev_fps = 1000.0 / max(self._frame_times[-2], 0.001)
        return alpha * fps + (1.0 - alpha) * prev_fps

    _last_gpu_stats: dict | None = None
    _last_gpu_check_time: float = 0.0

    @classmethod
    def get_gpu_utilization(cls) -> dict | None:
        """Return GPU utilization and VRAM usage if CUDA is available, cached to 1 sec."""
        now = time.time()
        if now - cls._last_gpu_check_time < 1.0:
            return cls._last_gpu_stats

        try:
            import torch
            if not torch.cuda.is_available():
                return None
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_util = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                pass
            stats = {
                "gpu_utilization_pct": int(gpu_util),
                "gpu_allocated_gb": round(allocated, 2),
                "gpu_reserved_gb": round(reserved, 2),
                "gpu_total_gb": round(total, 2),
            }
            cls._last_gpu_stats = stats
            cls._last_gpu_check_time = now
            return stats
        except Exception:
            return None

    @staticmethod
    def get_cpu_utilization() -> float | None:
        """Return CPU utilization percentage. Requires psutil."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return None
