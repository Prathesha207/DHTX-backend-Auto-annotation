"""
inference_service.py — Stage A Production Refactor
===================================================
Phases 1–7 implemented:
  Phase 1 — VideoRuntimeContext + OverlayData (container only, no rendering yet)
  Phase 2 — AsyncVideoWriter (bounded-wait + consecutive-full failure threshold)
  Phase 3 — Hardened run_inference() lifecycle (finally captures writer_error)
  Phase 4 — Writer state machine: OPEN / CLOSING / CLOSED / FAILED / STALLED
  Phase 5 — Transactional install_hooks() / uninstall_hooks()  (idempotent)
  Phase 6 — Stop via threading.Event on VideoRuntimeContext (no more global set)
  Phase 7 — OverlayData snapshot set in hooked_draw_hud (Stage B renderer reads it)
  Phase 12 — All print() replaced with logger.*

Stage B (overlay rendering on writer thread) is NOT part of this commit.

Non-negotiable constraints:
  - inference_video_full_detection.py is NEVER modified
  - run_single_video() is NEVER modified
  - No model / forward-pass changes
  - No algorithm changes
  - Sequential batch queue unchanged
  - WebSocket / streaming behaviour unchanged
"""

import dataclasses
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from datetime import datetime as original_datetime_class

import cv2
import numpy as np

# Import the ML team's code strictly without modifying it
import ml.inference_video_full_detection as ml_pipeline
from ml.inference_video_full_detection import run_single_video, PipelineConfig
from services.frame_streamer import streamer
from services.video_overlay_renderer import render_overlay

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  FRAME VALIDATOR  (unchanged logic — only lookup mechanism changed)
# ══════════════════════════════════════════════════════════════════════


# ── Framevalidator ────────────────────────────────────
# This function/class is responsible for framevalidator operations.
class FrameValidator:
    def __init__(self, model_settings: dict):
        self.m1_total    = model_settings.get("m1_total", 30)
        self.m1_pass_req = model_settings.get("m1_pass",  18)
        self.m2_total    = model_settings.get("m2_total", 30)
        self.m2_pass_req = model_settings.get("m2_pass",  18)

        if not (1 <= self.m1_pass_req <= self.m1_total):
            raise ValueError("Model 1 pass_frames must be between 1 and total_frames")
        if not (1 <= self.m2_pass_req <= self.m2_total):
            raise ValueError("Model 2 pass_frames must be between 1 and total_frames")

        self.last_state = None
        self.reset_cycle()

    def reset_cycle(self):
        self.m1_frames      = 0
        self.m1_successes   = 0
        self.m1_passed      = False
        self.m2_frames      = 0
        self.m2_successes   = 0
        self.m2_passed      = False
        self.m2_locked      = True
        self.validation_complete = False

    def on_model1_frame(self, socket_detected, hand_in_roi):
        if self.m1_passed:
            return
        if hand_in_roi:
            self.m1_frames    = 0
            self.m1_successes = 0
            return
        self.m1_frames += 1
        if socket_detected:
            self.m1_successes += 1
        if self.m1_successes >= self.m1_pass_req:
            self.m1_passed  = True
            self.m2_locked  = False
        elif self.m1_frames >= self.m1_total:
            self.m1_frames    = 0
            self.m1_successes = 0

    def on_model2_frame(self, order_status, hand_in_roi):
        if self.m2_locked or self.validation_complete:
            return
        if hand_in_roi:
            return
        self.m2_frames += 1
        if order_status in ("OK", "ANOMALY"):
            self.m2_successes += 1
        if self.m2_successes >= self.m2_pass_req:
            self.m2_passed           = True
            self.validation_complete = True
        elif self.m2_frames >= self.m2_total:
            self.m2_frames    = 0
            self.m2_successes = 0


# ══════════════════════════════════════════════════════════════════════
#  OVERLAY DATA  
# ══════════════════════════════════════════════════════════════════════

@dataclass

# ── Overlaydata ───────────────────────────────────────
# This function/class is responsible for overlaydata operations.
class OverlayData:
    """Per-frame snapshot populated by hooked_draw_hud on the inference thread.
    The Stage B renderer will consume this on the writer thread to burn labels."""
    cycle_no:       int   = 0
    fps:            float = 0.0
    frame_idx:      int   = 0
    frame_ms:       float = 0.0
    verdict:        str   = ""
    state:          str   = "IDLE"
    m1_frames:      int   = 0
    m1_total:       int   = 0
    m2_frames:      int   = 0
    m2_total:       int   = 0
    socket_present: bool  = False
    tube_detected:  bool  = False
    hand_present:   bool  = False


# ══════════════════════════════════════════════════════════════════════
#  VIDEO RUNTIME CONTEXT  
# ══════════════════════════════════════════════════════════════════════

@dataclass

# ── Videoruntimecontext ───────────────────────────────
# This function/class is responsible for videoruntimecontext operations.
class VideoRuntimeContext:
    """Owns all per-video mutable state for the duration of one run_inference call."""
    video_id:       str
    validator:      Optional[FrameValidator] = None
    overlay_data:   OverlayData              = field(default_factory=OverlayData)
    # Per-video stop signal — set by request_stop_inference, read lock-free by hooked_read
    stop_requested: threading.Event          = field(default_factory=threading.Event)
    # Writer accounting (written by AsyncVideoWriter worker thread)
    dropped_frames: int                      = 0
    written_frames: int                      = 0
    # Writer error propagated to run_inference return value
    writer_error:   Optional[Exception]      = None


# ══════════════════════════════════════════════════════════════════════
#  CONTEXT REGISTRY  
# ══════════════════════════════════════════════════════════════════════

_contexts_lock: threading.Lock                    = threading.Lock()
_CONTEXTS:      dict[str, VideoRuntimeContext]    = {}
_local          = threading.local()   # _local.context → active ctx on this thread



# ── Create Context ────────────────────────────────────
# This function/class is responsible for create context operations.
def create_context(
    video_id: str,
    model_settings: dict | None = None,
) -> VideoRuntimeContext:
    """Create a new context for video_id and bind it to the current thread."""
    ctx = VideoRuntimeContext(
        video_id  = video_id,
        validator = FrameValidator(model_settings) if model_settings else None,
    )
    with _contexts_lock:
        if video_id in _CONTEXTS:
            raise RuntimeError(
                f"VideoRuntimeContext already exists for video_id={video_id!r}. "
                "Two workers cannot own the same video simultaneously."
            )
        _CONTEXTS[video_id] = ctx
    _local.context = ctx
    logger.info("Context created: video_id=%s", video_id)
    return ctx



# ── Get Active Context ────────────────────────────────
# This function/class is responsible for get active context operations.
def get_active_context() -> Optional[VideoRuntimeContext]:
    """Returns the context bound to the current thread — no locking needed."""
    return getattr(_local, "context", None)



# ── Get Context ───────────────────────────────────────
# This function/class is responsible for get context operations.
def get_context(video_id: str) -> Optional[VideoRuntimeContext]:
    with _contexts_lock:
        return _CONTEXTS.get(video_id)



# ── Cleanup Context ───────────────────────────────────
# This function/class is responsible for cleanup context operations.
def cleanup_context(video_id: str) -> None:
    with _contexts_lock:
        _CONTEXTS.pop(video_id, None)
    # Only clear thread-local if it belongs to this video_id (guards against wrong-ID clear)
    local_ctx = getattr(_local, "context", None)
    if local_ctx is not None and local_ctx.video_id == video_id:
        _local.context = None
    logger.info("Context cleaned: video_id=%s", video_id)




# ══════════════════════════════════════════════════════════════════════
#  ASYNC VIDEO WRITER  
# ══════════════════════════════════════════════════════════════════════


# ──  Writerstate ──────────────────────────────────────
# This function/class is responsible for  writerstate operations.
class _WriterState(Enum):
    OPEN    = auto()
    CLOSING = auto()
    CLOSED  = auto()
    FAILED  = auto()
    STALLED = auto()



# ── Asyncvideowriter ──────────────────────────────────
# This function/class is responsible for asyncvideowriter operations.
class AsyncVideoWriter:
    """Thread-safe wrapper around cv2.VideoWriter.

    Queue policy (Phase 2 decision):
        bounded wait (QUEUE_FULL_TIMEOUT) then count consecutive failures.
        If CONSECUTIVE_FULL_LIMIT consecutive queue-full events occur,
        writer_error is propagated to the context and stop_requested is triggered.
        This preserves recording integrity while bounding worst-case inference delay.

    Lifecycle (Phase 4 state machine):
        OPEN → write() accepted
        CLOSING → release() called, sentinel enqueued, waiting for worker
        CLOSED → worker exited cleanly, real_writer released
        FAILED → real_writer.write() raised or real_writer.release() raised
        STALLED → worker did not exit within drain_timeout

    Stage A: frames written as-is (no overlay).
    Stage B: render_overlay(frame, overlay) will be called inside _run_worker().
    """

    QUEUE_CAPACITY         = 300
    QUEUE_FULL_TIMEOUT     = 0.5    # seconds to wait before counting a consecutive-full
    CONSECUTIVE_FULL_LIMIT = 10     # consecutive queue-full events → writer_error + stop
    DRAIN_TIMEOUT          = 10.0   # seconds total for release() to drain + join

    def __init__(self, real_writer: cv2.VideoWriter, ctx: VideoRuntimeContext):
        self.real_writer       = real_writer
        self.ctx               = ctx
        self._q: queue.Queue   = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self._state            = _WriterState.OPEN
        self._state_lock       = threading.Lock()
        self._consecutive_full = 0
        self._worker           = threading.Thread(
            target = self._run_worker,
            name   = f"AsyncWriter-{ctx.video_id}",
            daemon = False,   # must not be daemon — must drain before process exit
        )
        self._worker.start()
        logger.info("AsyncVideoWriter OPEN: video_id=%s thread=%s",
                    ctx.video_id, self._worker.name)

    # ── Producer side (inference thread) ──────────────────────────────

    def write(self, frame: np.ndarray) -> None:
        with self._state_lock:
            if self._state != _WriterState.OPEN:
                self.ctx.dropped_frames += 1
                return  # silently discard after CLOSING/CLOSED/FAILED/STALLED

        # Snapshot overlay_data so the writer thread sees this exact frame's metadata
        overlay = dataclasses.replace(self.ctx.overlay_data)
        item    = (frame.copy(), overlay)

        try:
            self._q.put(item, timeout=self.QUEUE_FULL_TIMEOUT)
            self._consecutive_full = 0  # reset on success
        except queue.Full:
            self._consecutive_full  += 1
            self.ctx.dropped_frames += 1
            logger.warning(
                "Writer queue full: video_id=%s consecutive=%d dropped_total=%d",
                self.ctx.video_id,
                self._consecutive_full,
                self.ctx.dropped_frames,
            )
            if self._consecutive_full >= self.CONSECUTIVE_FULL_LIMIT:
                err = RuntimeError(
                    f"AsyncVideoWriter cannot keep up with inference "
                    f"(video_id={self.ctx.video_id}, "
                    f"{self._consecutive_full} consecutive queue-full events)"
                )
                logger.error("%s", err)
                self.ctx.writer_error = err
                self.ctx.stop_requested.set()

    def release(self) -> None:
        """Drain the queue, stop the worker thread, and release the underlying writer.

        Uses a shared deadline to bound total shutdown time to DRAIN_TIMEOUT seconds.
        real_writer.release() is only called when the worker thread has actually exited.
        Idempotent — safe to call more than once.
        """
        with self._state_lock:
            if self._state in (_WriterState.CLOSED, _WriterState.CLOSING, _WriterState.STALLED):
                return
            
            was_failed = (self._state == _WriterState.FAILED)
            self._state = _WriterState.CLOSING

        deadline = time.monotonic() + self.DRAIN_TIMEOUT

        # 1. Enqueue sentinel with bounded wait (only if not failed)
        if not was_failed:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                self._q.put(None, timeout=remaining)
            except queue.Full:
                logger.critical(
                    "AsyncVideoWriter sentinel enqueue timed out: "
                    "video_id=%s thread=%s drain_timeout=%.1fs",
                    self.ctx.video_id, self._worker.name, self.DRAIN_TIMEOUT,
                )
                self._mark_stalled()
                return

        # 2. Join worker thread with bounded wait
        remaining = max(0.0, deadline - time.monotonic())
        self._worker.join(timeout=remaining)

        if self._worker.is_alive():
            logger.critical(
                "AsyncVideoWriter worker did not exit within drain_timeout=%.1fs — "
                "video_id=%s thread=%s — writer STALLED, real_writer NOT released",
                self.DRAIN_TIMEOUT, self.ctx.video_id, self._worker.name,
            )
            self._mark_stalled()
            return

        # 3. Worker exited cleanly — safe to release the underlying writer
        try:
            self.real_writer.release()
        except Exception as exc:
            logger.error(
                "real_writer.release() failed: video_id=%s error=%s",
                self.ctx.video_id, exc,
            )
            self._mark_failed(exc)
            return

        with self._state_lock:
            self._state = _WriterState.CLOSED
        logger.info(
            "AsyncVideoWriter CLOSED: video_id=%s written=%d dropped=%d",
            self.ctx.video_id,
            self.ctx.written_frames,
            self.ctx.dropped_frames,
        )

    # ── Worker side (writer thread) ────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:           # sentinel → clean exit
                self._q.task_done()
                break

            frame, overlay = item

            # Stage B: render inspection HUD onto the saved frame (writer thread only)
            try:
                frame = render_overlay(frame, overlay)
            except Exception as exc:
                logger.warning("render_overlay failed (frame written without overlay): %s", exc)

            try:
                self.real_writer.write(frame)
                self.ctx.written_frames += 1
            except Exception as exc:
                logger.error(
                    "real_writer.write() raised: video_id=%s error=%s",
                    self.ctx.video_id, exc,
                )
                self._mark_failed(exc)
                self._q.task_done()
                break
            else:
                self._q.task_done()

    # ── State helpers ──────────────────────────────────────────────────

    def _mark_failed(self, exc: Exception) -> None:
        with self._state_lock:
            self._state = _WriterState.FAILED
        if self.ctx.writer_error is None:
            self.ctx.writer_error = exc

    def _mark_stalled(self) -> None:
        with self._state_lock:
            self._state = _WriterState.STALLED
        if self.ctx.writer_error is None:
            self.ctx.writer_error = RuntimeError(
                f"AsyncVideoWriter stalled: video_id={self.ctx.video_id} "
                f"thread={self._worker.name} drain_timeout={self.DRAIN_TIMEOUT}s"
            )


# ══════════════════════════════════════════════════════════════════════
#  DATETIME MOCK  
# ══════════════════════════════════════════════════════════════════════


# ── Mocknow ───────────────────────────────────────────
# This function/class is responsible for mocknow operations.
class MockNow:
    def strftime(self, fmt):
        if fmt == "%Y-%m-%d":
            return ""   # suppress extra date-folder creation
        return original_datetime_class.now().strftime(fmt)



# ── Mockdatetime ──────────────────────────────────────
# This function/class is responsible for mockdatetime operations.
class MockDatetime(original_datetime_class):
    @classmethod
    def now(cls, tz=None):
        return MockNow()


# ══════════════════════════════════════════════════════════════════════
#  HOOK FUNCTIONS  
# ══════════════════════════════════════════════════════════════════════

_global_shutdown = False



# ── Request Global Shutdown ───────────────────────────
# This function/class is responsible for Signals all active inference loops to terminate immediately during backend shutdown.
def request_global_shutdown() -> None:
    global _global_shutdown
    _global_shutdown = True
    logger.warning("Global shutdown requested — all inference reads will terminate.")



# ── Hooked Read ───────────────────────────────────────
# This function/class is responsible for hooked read operations.
def hooked_read(self_cap):
    """Intercepts cv2.VideoCapture.read.
    Uses thread-local context — NO registry locking in the hot read path."""
    ctx = get_active_context()
    if _global_shutdown:
        logger.info("hooked_read: global shutdown — terminating ML loop.")
        return False, None
    if ctx is not None and ctx.stop_requested.is_set():
        logger.info("hooked_read: stop requested — video_id=%s", ctx.video_id)
        return False, None
    return original_video_capture_read(self_cap)



# ── Hooked Draw Seg Overlay ───────────────────────────
# This function/class is responsible Signals all active inference loops to terminate immediately during backend shutdown.
def hooked_draw_seg_overlay(frame, pred_map, alpha=0.90):
    """Suppresses segmentation overlay until Model 1 validation has passed."""
    ctx       = get_active_context()
    validator = ctx.validator if ctx else None
    if validator and not validator.m1_passed:
        return frame
    return original_draw_seg_overlay(frame, pred_map, alpha)



# ── Hooked Draw Hud ───────────────────────────────────
# This function/class is responsible for hooked draw hud operations.
def hooked_draw_hud(*args, **kwargs):
    """Intercepts draw_hud.

    1. Updates validator state (Model 1 / Model 2 frame counting).
    2. Snapshots OverlayData into ctx.overlay_data (consumed by Stage B renderer).
    3. Pushes telemetry to the live WebSocket streamer.
    4. Returns the CLEAN frame — no text or HUD burned into it.
    """
    frame           = args[0]
    fps             = args[1] if len(args) > 1 else kwargs.get("fps", 0)
    frame_idx       = args[2] if len(args) > 2 else kwargs.get("frame_idx", 0)
    state           = args[3] if len(args) > 3 else kwargs.get("state", "IDLE")
    socket_detected = args[4] if len(args) > 4 else kwargs.get("sock_hit")
    order_status    = kwargs.get("order_status")
    hand_in_roi     = kwargs.get("hand_in_roi", False)
    cycle_no        = kwargs.get("cycle_no", 0)
    config          = kwargs.get("config")

    ctx       = get_active_context()
    validator = ctx.validator if ctx else None

    # ── Validator update (Model 1 / Model 2 frame counting) ───────────
    if validator and config:
        cls_socket      = config.socket.cls_socket
        is_valid_socket = (
            socket_detected is not None
            and socket_detected["class"] == cls_socket
        )
        if state == getattr(ml_pipeline, "STATE_IDLE", "IDLE"):
            if validator.last_state != getattr(ml_pipeline, "STATE_IDLE", "IDLE"):
                validator.reset_cycle()
        else:
            validator.on_model1_frame(is_valid_socket, hand_in_roi)
            if state != getattr(ml_pipeline, "STATE_WARMUP", "WARMUP"):
                validator.on_model2_frame(order_status, hand_in_roi)
        validator.last_state = state

    # ── Verdict from vote_counter ─────────────────────────────────────
    vote_counter     = kwargs.get("vote_counter")
    verdict          = "UNKNOWN"
    if vote_counter:
        try:
            verdict = vote_counter.final_verdict()
        except Exception:
            pass

    total_frames_val = kwargs.get("total_frames", 0)
    logger.debug(
        "draw_hud frame=%d/%d state=%s verdict=%s",
        frame_idx, total_frames_val, state, verdict,
    )

    # ──  \Snapshot OverlayData (Stage B renderer reads this) ──
    if ctx:
        ctx.overlay_data = OverlayData(
            cycle_no       = int(cycle_no),
            fps            = float(fps),
            frame_idx      = int(frame_idx),
            frame_ms       = float(kwargs.get("frame_ms", 0.0)),
            verdict        = verdict,
            state          = str(state),
            m1_frames      = int(validator.m1_frames    if validator else 0),
            m1_total       = int(validator.m1_total     if validator else 0),
            m2_frames      = int(validator.m2_frames    if validator else 0),
            m2_total       = int(validator.m2_total     if validator else 0),
            socket_present = bool(socket_detected is not None),
            tube_detected  = bool(order_status in ("OK", "ANOMALY")),
            hand_present   = bool(hand_in_roi),
        )

    # ── Telemetry for live WebSocket streamer  ─────────────
    telemetry = {
        "video_id": ctx.video_id if ctx else None,
        "video": {
            "frame":        int(frame_idx),
            "total_frames": int(total_frames_val),
            "fps":          float(round(fps, 1)),
            "frame_ms":     float(kwargs.get("frame_ms", 0.0)),
            "frame_ms_avg": float(kwargs.get("frame_ms_avg", 0.0)),
        },
        "cycle": {
            "number":  int(cycle_no),
            "state":   str(state),
            "verdict": verdict,
        },
        "socket": {"present": bool(socket_detected is not None)},
        "hand":   {"detected": bool(hand_in_roi), "in_roi": bool(hand_in_roi)},
        "tube":   {"order_status": str(order_status)},
        "models": {
            "model_1": {
                "purpose":          "SOCKET_DETECTION",
                "frames_processed": int(validator.m1_frames    if validator else 0),
                "success_frames":   int(validator.m1_successes if validator else 0),
                "passed":           bool(validator.m1_passed   if validator else False),
                "pass_req":         int(validator.m1_pass_req  if validator else 0),
                "total_req":        int(validator.m1_total     if validator else 0),
            },
            "model_2": {
                "purpose":          "INSPECTION",
                "frames_processed": int(validator.m2_frames    if validator else 0),
                "success_frames":   int(validator.m2_successes if validator else 0),
                "passed":           bool(validator.m2_passed   if validator else False),
                "pass_req":         int(validator.m2_pass_req  if validator else 0),
                "total_req":        int(validator.m2_total     if validator else 0),
            },
        },
    }

    streamer.push(frame, telemetry)
    return frame   # clean frame — no text burned in



# ── Hooked Draw Production Status Bar ─────────────────
# This function/class is responsible for hooked draw production status bar operations.
def hooked_draw_production_status_bar(*args, **kwargs):
    """Strips the production status bar from the frame (kept clean for the streamer)."""
    return args[0]



# ── Hooked End Cycle ──────────────────────────────────
# This function/class is responsible for hooked end cycle operations.
def hooked_end_cycle(self, final_verdict, extra_metrics=None):
    """Overrides verdict to UNKNOWN when validation was not completed."""
    ctx       = get_active_context()
    validator = ctx.validator if ctx else None

    if validator and not validator.validation_complete:
        logger.warning(
            "Cycle #%d ended with incomplete validation — overriding verdict to UNKNOWN. "
            "video_id=%s",
            self.cycle_no, ctx.video_id if ctx else "unknown",
        )
        return original_end_cycle(self, "UNKNOWN", extra_metrics)

    return original_end_cycle(self, final_verdict, extra_metrics)



# ── Hooked Start Cycle ────────────────────────────────
# This function/class is responsible for hooked start cycle operations.
def hooked_start_cycle(self):
    """Wraps the raw cv2.VideoWriter with AsyncVideoWriter immediately after cycle start."""
    original_start_cycle(self)
    ctx = get_active_context()
    if ctx is not None and getattr(self, "writer", None) is not None:
        self.writer = AsyncVideoWriter(self.writer, ctx)
        logger.info(
            "AsyncVideoWriter installed: cycle=#%d video_id=%s",
            self.cycle_no, ctx.video_id,
        )


# ══════════════════════════════════════════════════════════════════════
#  ORIGINALS CAPTURED BEFORE ANY PATCHING 
# ══════════════════════════════════════════════════════════════════════

original_datetime                   = ml_pipeline.datetime
original_video_capture_read         = cv2.VideoCapture.read
original_run_model_tube_segmentation = ml_pipeline.run_model_tube_segmentation  # preserved, not hooked
original_draw_seg_overlay           = ml_pipeline.draw_seg_overlay
original_draw_hud                   = ml_pipeline.draw_hud
original_draw_production_status_bar = ml_pipeline.draw_production_status_bar
original_end_cycle                  = ml_pipeline.CycleManager.end_cycle
original_start_cycle                = ml_pipeline.CycleManager.start_cycle


# ══════════════════════════════════════════════════════════════════════
#  HOOK LIFECYCLE 
# ══════════════════════════════════════════════════════════════════════

_PATCH_TARGETS = [
    # (owner,                          attr,                         replacement)
    (ml_pipeline,                      "datetime",                    MockDatetime),
    (cv2.VideoCapture,                 "read",                        hooked_read),
    (ml_pipeline,                      "draw_seg_overlay",            hooked_draw_seg_overlay),
    (ml_pipeline,                      "draw_hud",                    hooked_draw_hud),
    (ml_pipeline,                      "draw_production_status_bar",  hooked_draw_production_status_bar),
    (ml_pipeline.CycleManager,         "end_cycle",                   hooked_end_cycle),
    (ml_pipeline.CycleManager,         "start_cycle",                 hooked_start_cycle),
]

_ORIGINALS_MAP = {
    (id(ml_pipeline),              "datetime"):                   original_datetime,
    (id(cv2.VideoCapture),         "read"):                       original_video_capture_read,
    (id(ml_pipeline),              "draw_seg_overlay"):           original_draw_seg_overlay,
    (id(ml_pipeline),              "draw_hud"):                   original_draw_hud,
    (id(ml_pipeline),              "draw_production_status_bar"): original_draw_production_status_bar,
    (id(ml_pipeline.CycleManager), "end_cycle"):                  original_end_cycle,
    (id(ml_pipeline.CycleManager), "start_cycle"):                original_start_cycle,
}

_hooks_installed = False
_hooks_lock      = threading.Lock()



# ── Install Hooks ─────────────────────────────────────
# This function/class is responsible for install hooks operations.
def install_hooks() -> None:
    """Validate all patch targets, then apply atomically.

    Idempotent — safe to call multiple times.
    On any setattr failure, rolls back all previously applied patches.
    """
    global _hooks_installed
    with _hooks_lock:
        if _hooks_installed:
            logger.debug("install_hooks(): already installed — no-op.")
            return

        # 1. Validate all targets exist before touching anything
        for owner, attr, _ in _PATCH_TARGETS:
            if not hasattr(owner, attr):
                raise RuntimeError(
                    f"Cannot install hook: {owner!r}.{attr!r} does not exist. "
                    "The ML pipeline may have changed."
                )

        # 2. Apply patches transactionally — rollback on any failure
        applied: list[tuple] = []
        try:
            for owner, attr, replacement in _PATCH_TARGETS:
                setattr(owner, attr, replacement)
                applied.append((owner, attr))
        except Exception as exc:
            logger.critical(
                "Hook installation failed at %r.%r — rolling back %d patch(es)",
                owner, attr, len(applied),
            )
            _rollback_hooks(applied)
            raise RuntimeError(f"Hook installation aborted: {exc}") from exc

        _hooks_installed = True
        logger.info("All %d hooks installed successfully.", len(_PATCH_TARGETS))



# ── Uninstall Hooks ───────────────────────────────────
# This function/class is responsible for uninstall hooks operations.
def uninstall_hooks() -> None:
    """Restore all original ML pipeline attributes. Idempotent."""
    global _hooks_installed
    with _hooks_lock:
        if not _hooks_installed:
            logger.debug("uninstall_hooks(): not installed — no-op.")
            return

        _originals = [
            (ml_pipeline,              "datetime",                   original_datetime),
            (cv2.VideoCapture,         "read",                       original_video_capture_read),
            (ml_pipeline,              "draw_seg_overlay",           original_draw_seg_overlay),
            (ml_pipeline,              "draw_hud",                   original_draw_hud),
            (ml_pipeline,              "draw_production_status_bar", original_draw_production_status_bar),
            (ml_pipeline.CycleManager, "end_cycle",                  original_end_cycle),
            (ml_pipeline.CycleManager, "start_cycle",                original_start_cycle),
        ]
        for owner, attr, original in _originals:
            try:
                setattr(owner, attr, original)
            except Exception as exc:
                logger.error("uninstall_hooks: failed to restore %r.%r: %s", owner, attr, exc)

        _hooks_installed = False
        logger.info("All hooks uninstalled.")



# ──  Rollback Hooks ───────────────────────────────────
# This function/class is responsible for  rollback hooks operations.
def _rollback_hooks(applied: list[tuple]) -> None:
    """Restore only the patches that were already applied (transactional rollback)."""
    for owner, attr in reversed(applied):
        original = _ORIGINALS_MAP.get((id(owner), attr))
        if original is not None:
            try:
                setattr(owner, attr, original)
            except Exception as exc:
                logger.error("Rollback failed for %r.%r: %s", owner, attr, exc)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


# ── Get Active Video ──────────────────────────────────
# This function/class is responsible for get active video operations.
def get_active_video() -> str | None:
    """Returns the video_id of the currently running inference video.

    First checks the thread-local context (fast path, works on the inference worker thread).
    Falls back to scanning the global _CONTEXTS registry (works on any thread, e.g. the
    FastAPI request thread calling stop_inference).
    """
    # Fast path: thread-local (inference worker thread)
    ctx = get_active_context()
    if ctx is not None:
        return ctx.video_id

    # Cross-thread fallback: scan the global registry
    # Returns the first (and normally only) active video_id
    with _contexts_lock:
        if _CONTEXTS:
            return next(iter(_CONTEXTS))
    return None



# ── Request Stop Inference ────────────────────────────
# This function/class is responsible for request stop inference operations.
def request_stop_inference(video_id: str | None = None) -> bool:
    """Signal a stop for video_id via its context's threading.Event.

    Can be called from any thread (FastAPI request thread, worker thread, etc.).
    Uses the global _CONTEXTS registry — not thread-local — so it always reaches
    the running inference regardless of which thread calls this function.
    """
    target_id = video_id or get_active_video()
    if not target_id:
        logger.warning("request_stop_inference: no video_id and no active inference.")
        return False
    ctx = get_context(target_id)   # uses _CONTEXTS registry — thread-safe
    if ctx is None:
        logger.warning("request_stop_inference: no active context for video_id=%s", target_id)
        return False
    ctx.stop_requested.set()
    logger.info("Stop requested: video_id=%s", target_id)
    return True



# ── Is Stop Requested ─────────────────────────────────
# This function/class is responsible for is stop requested operations.
def is_stop_requested(video_id: str | None = None) -> bool:
    """Check if stop has been requested for the given video_id (or the active video).

    Safe to call from any thread.
    """
    target_id = video_id or get_active_video()
    if not target_id:
        return False
    ctx = get_context(target_id)
    return ctx.stop_requested.is_set() if ctx else False


# ══════════════════════════════════════════════════════════════════════
#  RUN INFERENCE 
# ══════════════════════════════════════════════════════════════════════


# ── Run Inference ─────────────────────────────────────
# This function/class is responsible for run inference operations.
def run_inference(
    video_path:     str,
    out_base_dir:   str,
    video_id:       str | None  = None,
    model_settings: dict | None = None,
) -> dict:
    """Run the ML inference pipeline for one video.

    Returns:
        dict with keys:
            - completed: bool (True if normal completion)
            - verdict: str | None (NORMAL, ANOMALY, UNKNOWN, ABORTED)
            - output_path: str | None (path to the saved video)
    """
    base_dir  = Path(__file__).resolve().parent.parent
    yaml_path = base_dir / "config" / "config.yaml"

    config = PipelineConfig.from_yaml(str(yaml_path))
    config.paths.video_path    = video_path
    config.paths.out_base      = out_base_dir
    config.ui.show_preview     = False
    
    config.device.require_gpu  = config.device.require_gpu if hasattr(config.device, 'require_gpu') else True
    config.device.use_half     = config.device.use_half if hasattr(config.device, 'use_half') else True

    def make_abs(p: str) -> str:
        if not p:
            return p
        path_obj = Path(p)
        return p if path_obj.is_absolute() else str((base_dir / path_obj).resolve())

    config.paths.seg_model_path    = make_abs(config.paths.seg_model_path)
    config.paths.socket_model_path = make_abs(config.paths.socket_model_path)
    config.paths.hand_pose_path    = make_abs(config.paths.hand_pose_path)

    ctx = create_context(video_id, model_settings)

    stopped      = False
    writer_error = None
    final_verdict = None
    final_output_path = None
    written_frames = 0
    dropped_frames = 0

    logger.info("Inference started: video_id=%s path=%s", video_id, video_path)

    try:
        cycles = run_single_video(
            config          = config,
            print_summary   = False,
            enable_debug    = False,
            forced_channels = None,
        )
        
        if cycles and hasattr(cycles, 'cycle_summaries') and cycles.cycle_summaries:
            latch_frames = getattr(config.inspection, 'latch_frames', 10)
            
            # Apply latch validation override across all cycles first
            for summary in cycles.cycle_summaries:
                infer_frames = summary.get("infer_frames", 0)
                verdict = summary.get("final_verdict")
                old_path = summary.get("output_path")
                
                if infer_frames < latch_frames and verdict != "UNKNOWN" and old_path and os.path.exists(old_path):
                    logger.warning("Video %s had only %d infer frames (min %d). Forcing UNKNOWN.", video_id, infer_frames, latch_frames)
                    
                    import os, shutil
                    unknown_dir = Path(config.paths.out_base) / "UNKNOWN"
                    unknown_dir.mkdir(parents=True, exist_ok=True)
                    
                    old_name = os.path.basename(old_path)
                    new_name = old_name.replace(f"_{verdict}_", "_UNKNOWN_")
                    new_path = unknown_dir / new_name
                    
                    try:
                        shutil.move(old_path, str(new_path))
                        # Update the summary inline so the final extraction picks it up
                        summary["final_verdict"] = "UNKNOWN"
                        summary["output_path"] = str(new_path)
                    except Exception as exc:
                        logger.error("Could not move latch-failed file to UNKNOWN: %s", exc)

            # Now strictly extract the final, authoritative result from the last cycle
            last_summary = cycles.cycle_summaries[-1]
            final_verdict = last_summary.get("final_verdict")
            final_output_path = last_summary.get("output_path")

    except Exception:
        logger.exception("Inference pipeline raised: video_id=%s", video_id)
        raise
    finally:
        stopped      = ctx.stop_requested.is_set()
        writer_error = ctx.writer_error
        written_frames = ctx.written_frames
        dropped_frames = ctx.dropped_frames
        cleanup_context(video_id)
        streamer.clear_snapshot(video_id)
        
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if writer_error is not None:
        logger.error(
            "Inference completed but writer failed: video_id=%s error=%s",
            video_id, writer_error,
        )
        return {"completed": False, "verdict": "UNKNOWN", "output_path": None, "reason": "WRITER_FAILED"}

    if stopped:
        logger.info("Inference was stopped by user request: video_id=%s", video_id)
        return {"completed": False, "verdict": "ABORTED", "output_path": final_output_path, "reason": "STOP_REQUESTED"}

    logger.info(
        "Inference completed successfully: video_id=%s written=%d dropped=%d",
        video_id,
        written_frames,
        dropped_frames,
    )
    
    if final_verdict == "UNKNOWN":
        reason = "VALIDATION_INCOMPLETE"
    else:
        reason = None
        
    return {"completed": True, "verdict": final_verdict, "output_path": final_output_path, "reason": reason}


# ══════════════════════════════════════════════════════════════════════
#  MODULE INIT — install all hooks once at import time
# ══════════════════════════════════════════════════════════════════════
install_hooks()
