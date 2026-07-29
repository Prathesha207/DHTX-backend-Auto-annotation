"""
Inference State Machine
========================
Controls the inference flow for a single video. Reads config from DB.
Calls into inference_video_full_detection.py for actual model execution.
Logs everything via LogService.

States:
    WAIT_FOR_SOCKET     — run Model1 only
    MODEL1_VALIDATION   — run Model1 + Hand check
    MODEL2_SKIP         — run Model1 only (fixed time delay)
    MODEL2_VALIDATION   — run Model1 + Hand check + Model2
    WAIT_SOCKET_REMOVAL — run Model1 only
    CYCLE_FINISHED      — transient, reset and loop

The existing inference_video_full_detection.py does the actual model
inference. This module only controls WHAT to run and WHEN.
"""

import os
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np

from sqlalchemy.orm import Session

# ── Inference engine imports (functions, NOT the main loop) ──────
from app.services import ml_adapter as inf_mod
from app.services.ml_adapter import (
    detect_socket,
    detect_hand_in_roi,
    build_roi,
    SegmentationEngine,
    VoteCounter,
    SequenceStabilityGate,
    AnomalyConfirmGate,
    CycleManager,
    append_to_excel,
    draw_raw_argmax_fallback,
    draw_socket_box,
    draw_production_status_bar,
    draw_hud,
    draw_debug_overlay,
    draw_seg_overlay,
    draw_final_verdict_overlay,
    YOLO_SOCKET_CONF,
    YOLO_POSE_CONF,
    CLS_SOCKET,
    CLS_NO_SOCKET,
    NUM_CLASSES,
    WARMUP_FRAMES,
    VERDICT_THR,
    VERDICT_HOLD_SEC,
    N_ANOMALY_CONFIRM,
    LATCH_FRAMES,
    MIN_SEQ_STABLE,
    TUBE_SHORT,
    STATE_IDLE,
    STATE_WARMUP,
    STATE_HAND,
    STATE_INSPECT,
    STATE_NORMAL,
    STATE_ANOMALY,
    STATE_PARTIAL,
    _ORDER_DISPLAY,
    _prev_gray_ref,
)

from app.services.log_service import LogService, flush_pending_logs
from app.services.cycle_service import CycleService
from app.services.video_run_service import VideoRunService
from app.services.websocket_manager import manager
from app.services.perf_monitor import PerfMonitor
from app.video_metadata import probe_video


# ══════════════════════════════════════════════════════════════════
#  States
# ══════════════════════════════════════════════════════════════════
WAIT_FOR_SOCKET = "WAIT_FOR_SOCKET"
MODEL1_VALIDATION = "MODEL1_VALIDATION"
MODEL2_SKIP = "MODEL2_SKIP"
MODEL2_VALIDATION = "MODEL2_VALIDATION"
WAIT_SOCKET_REMOVAL = "WAIT_SOCKET_REMOVAL"
CYCLE_FINISHED = "CYCLE_FINISHED"

# Map new states to the old HUD states for rendering compatibility
_HUD_STATE_MAP = {
    WAIT_FOR_SOCKET: STATE_IDLE,
    MODEL1_VALIDATION: STATE_WARMUP,
    MODEL2_SKIP: STATE_WARMUP,
    MODEL2_VALIDATION: STATE_INSPECT,
    WAIT_SOCKET_REMOVAL: STATE_IDLE,
    CYCLE_FINISHED: STATE_IDLE,
}


@dataclass
class InferenceConfigSnapshot:
    """Frozen copy of inference_config row for one video run."""
    model1_frame_count: int = 30
    model1_pass_frames: int = 28
    model2_start_skip_frame: int = 10
    model2_frame_count: int = 20
    model2_pass_frames: int = 18
    socket_absent_frames: int = 10
    socket_loss_abort_frames: int = 15
    enable_debug_logging: bool = False
    enable_perf_logging: bool = True


class InferenceStateMachine:
    """
    Runs the complete inference pipeline for one video.
    Called by MLRunner.run_video() instead of inf_mod.run_single_video().
    """

    def __init__(
        self,
        *,
        db: Session,
        batch_id: int,
        video_run_id: int,
        video_path: str,
        output_dir: str,
        original_name: str,
        models: tuple,
        config: InferenceConfigSnapshot,
        enable_debug: bool = False,
        stream_hud: bool = False,
    ):
        self.db = db
        self.batch_id = batch_id
        self.video_run_id = video_run_id
        self.video_path = video_path
        self.output_dir = output_dir
        self.original_name = original_name
        self.config = config
        self.enable_debug = enable_debug

        self.seg_net, self.yolo_socket, self.yolo_pose = models

        # ── State ────────────────────────────────────────────────
        self.state = WAIT_FOR_SOCKET
        self.frame_idx = 0

        # Model1 counters
        self.m1_valid_total = 0
        self.m1_valid_pass = 0

        # Model2 counters
        self.m2_skip_count = 0
        self.m2_valid_count = 0

        # Socket removal counter
        self.socket_absent_count = 0
        self.m1_socket_absent_streak = 0  # for socket_loss_abort_frames

        # Telemetry stats
        self.hand_ignored_ct = 0
        self.socket_lost_ct = 0
        self.m2_skipped_ct = 0
        self._last_status_time = 0.0
        
        # Verdict
        self.verdict: str | None = None

        # ROI / socket state
        self.invisible_roi = None
        self.last_socket_centre = None
        self.last_sock_hit = None

        # Frame context for rendering
        self.hand_in_roi = False

        # Perf
        self.perf = PerfMonitor()

        # Components (initialized in run())
        self.seg_engine: SegmentationEngine | None = None
        self.vote_counter: VoteCounter | None = None
        self.seq_gate: SequenceStabilityGate | None = None
        self.anomaly_gate: AnomalyConfirmGate | None = None
        self.cycle_mgr: CycleManager | None = None

        # Rendering state
        self.pred_map = None
        self.status_dict = {2: "Absent", 3: "Absent", 4: "Absent"}
        self.order_status = "N/A"
        self.detected_seq = []
        self.current_dbg = {}
        self.cycle_start_frame = 0
        self.cycle_total_frames = 0
        self.infer_frames = 0

        # Last emitted status for throttling
        self._last_emitted_state = None
        self._fps_ema = 30.0

        self.last_vis = None

        # Peak state for cycle summary
        self.peak_status = {2: "Absent", 3: "Absent", 4: "Absent"}
        self.peak_seq = []
        self.peak_order = "N/A"

    # ── Logging helpers ──────────────────────────────────────────

    def _log_info(self, msg, **kw):
        LogService.info(
            self.db, self.batch_id, msg,
            video_run_id=self.video_run_id,
            frame_number=kw.get("frame", self.frame_idx),
            state=self.state,
            cycle_id=kw.get("cycle_id"),
        )

    def _log_warning(self, msg, **kw):
        LogService.warning(
            self.db, self.batch_id, msg,
            video_run_id=self.video_run_id,
            frame_number=kw.get("frame", self.frame_idx),
            state=self.state,
            cycle_id=kw.get("cycle_id"),
        )

    def _log_debug(self, msg, **kw):
        if not self.config.enable_debug_logging:
            return
        LogService.debug(
            self.db, self.batch_id, msg,
            video_run_id=self.video_run_id,
            frame_number=kw.get("frame", self.frame_idx),
            state=self.state,
            cycle_id=kw.get("cycle_id"),
        )

    def _log_perf(self, msg, **kw):
        if not self.config.enable_perf_logging:
            return
        LogService.perf(
            self.db, self.batch_id, msg,
            video_run_id=self.video_run_id,
            frame_number=kw.get("frame", self.frame_idx),
            state=self.state,
        )

    def _log_error(self, msg, **kw):
        LogService.error(
            self.db, self.batch_id, msg,
            video_run_id=self.video_run_id,
            frame_number=kw.get("frame", self.frame_idx),
            state=self.state,
        )

    # ── State transition ─────────────────────────────────────────

    def _transition(self, new_state: str):
        old = self.state
        self.state = new_state
        self._log_info(f"[STATE] {old} → {new_state}")

    # ── Reset helpers ────────────────────────────────────────────

    def _reset_m1(self):
        self.m1_valid_total = 0
        self.m1_valid_pass = 0
        self.m1_socket_absent_streak = 0

    def _reset_m2(self):
        self.m2_skip_count = 0
        self.m2_valid_count = 0
        self.verdict = None

    def _reset_all(self):
        self._reset_m1()
        self._reset_m2()
        self.socket_absent_count = 0
        self.invisible_roi = None
        self.last_socket_centre = None
        self.hand_in_roi = False
        self.pred_map = None
        self.status_dict = {2: "Absent", 3: "Absent", 4: "Absent"}
        self.order_status = "N/A"
        self.detected_seq = []
        self.current_dbg = {}
        self.cycle_total_frames = 0
        self.infer_frames = 0
        self.peak_status = {2: "Absent", 3: "Absent", 4: "Absent"}
        self.peak_seq = []
        self.peak_order = "N/A"
        if self.seg_engine:
            self.seg_engine.reset()
        if self.vote_counter:
            self.vote_counter.reset()
        if self.seq_gate:
            self.seq_gate.reset()
        if self.anomaly_gate:
            self.anomaly_gate.reset()
        _prev_gray_ref[0] = None

    # ══════════════════════════════════════════════════════════════
    #  Main entry point
    # ══════════════════════════════════════════════════════════════

    def run(self, cancel_event=None):
        """
        Process the video. Returns True on success, False on error.
        """
        self._log_info(f"Starting inference: {self.original_name}")
        self._log_info(f"Config: M1 count={self.config.model1_frame_count} "
                       f"pass={self.config.model1_pass_frames} | "
                       f"M2 skip={self.config.model2_start_skip_frame} "
                       f"count={self.config.model2_frame_count} "
                       f"pass={self.config.model2_pass_frames} | "
                       f"absent={self.config.socket_absent_frames}")

        if not os.path.exists(self.video_path):
            self._log_error(f"Video file not found: {self.video_path}")
            return False

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._log_error(f"Failed to open video: {self.video_path}")
            return False

        # Attempt to gather total frames for progress bar
        total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_est <= 0:
            total_frames_est = 3000

        fps_src = cap.get(cv2.CAP_PROP_FPS)
        if fps_src <= 0 or fps_src > 120:
            fps_src = 30.0

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w == 0 or src_h == 0:
            src_w, src_h = 1920, 1080

        video_stem = Path(self.video_path).stem

        # Create output subdirectories
        for sub in ("NORMAL", "ANOMALY", "UNKNOWN"):
            Path(os.path.join(self.output_dir, sub)).mkdir(
                parents=True, exist_ok=True)

        # Initialize components
        self.seg_engine = SegmentationEngine(self.seg_net)
        self.vote_counter = VoteCounter(VERDICT_THR)
        self.seq_gate = SequenceStabilityGate()
        self.anomaly_gate = AnomalyConfirmGate(N_ANOMALY_CONFIRM)
        self.cycle_mgr = CycleManager(
            self.output_dir, video_stem, fps_src, (src_w, src_h))

        ZERO_PRED = np.zeros((src_h, src_w), dtype=np.int32)
        
        self._emit_capabilities()
        self._emit_event("inference_started", {"total_frames": total_frames_est})

        try:
    # ── Frame loop ───────────────────────────────────────────
            while True:
                if cancel_event and cancel_event.is_set():
                    self._log_warning("Inference cancelled via cancel_event")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_idx += 1
                self.perf.start_frame()

                vis = frame.copy()

                # ── Dispatch to current state handler ────────────────
                vis = self._dispatch_frame(frame, vis, ZERO_PRED)
                
                # Copy for frontend stream before HUD text is applied
                frontend_vis = vis.copy()

                # ── Rendering (HUD + production bar) ─────────────────
                hud_state = self._get_hud_state()

                # Production status bar
                vis = draw_production_status_bar(
                    vis, hud_state,
                    self.cycle_mgr.cycle_no,
                    self.cycle_mgr.passed,
                    self.cycle_mgr.failed,
                    self.cycle_mgr.unknown,
                )

                # HUD
                vis = draw_hud(
                    vis, self._fps_ema, self.frame_idx, hud_state,
                    self.last_sock_hit,
                    self.status_dict, self.order_status, self.detected_seq,
                    anomaly_counter=self.anomaly_gate._count if self.anomaly_gate else 0,
                    hand_in_roi=self.hand_in_roi,
                    warmup_frame=self.m1_valid_total,
                    warmup_retry=0,
                    vote_counter=self.vote_counter,
                    infer_frames=self.infer_frames,
                    seq_stable_ctr=self.seq_gate._stable_ct if self.seq_gate else 0,
                    cycle_no=self.cycle_mgr.cycle_no,
                )

                if self.enable_debug and self.current_dbg:
                    vis = draw_debug_overlay(vis, self.current_dbg)

                # ── FPS ──────────────────────────────────────────────
                frame_ms = self.perf.end_frame()
                if frame_ms > 0:
                    cur_fps = 1000.0 / frame_ms
                    self._fps_ema = 0.88 * self._fps_ema + 0.12 * cur_fps

                # ── Emit [STATUS] for frontend ───────────────────────
                if self.frame_idx % 5 == 0 or self.state != self._last_emitted_state:
                    self._emit_status()
                    self._last_emitted_state = self.state

                # ── Emit [FRAME] for live preview ────────────────────
                if manager.has_clients(self.batch_id):
                    self._emit_frame(frontend_vis, src_w)

                # ── Write to cycle video ─────────────────────────────
                self.cycle_mgr.write(vis)
                self.last_vis = vis

                # ── Progress report ──────────────────────────────────
                if self.frame_idx % 15 == 0:
                    VideoRunService.update_progress(
                        db=self.db,
                        video=self._get_video_run(),
                        current_frame=self.frame_idx,
                        total_frames=total_frames_est,
                    )

                # ── Performance logging ──────────────────────────────
                if self.frame_idx % 30 == 0:
                    breakdown = self.perf.frame_summary()
                    self._log_perf(
                        f"frame_time={frame_ms:.1f}ms "
                        f"sections={breakdown} "
                        f"FPS={self._fps_ema:.1f}"
                    )
                    gpu = PerfMonitor.get_gpu_utilization()
                    if gpu:
                        self._log_perf(
                            f"GPU alloc={gpu['gpu_allocated_gb']}GB "
                            f"reserved={gpu['gpu_reserved_gb']}GB "
                            f"total={gpu['gpu_total_gb']}GB"
                        )
                    cpu = PerfMonitor.get_cpu_utilization()
                    if cpu is not None:
                        self._log_perf(f"CPU={cpu:.1f}%")

            


        finally:
    # ── End of video ─────────────────────────────────────────
            # If a cycle is still active (video ended mid-cycle), finalize it
            if self.cycle_mgr and self.cycle_mgr.active:
                self._finalize_current_cycle(abort=True)

            if cap:
                cap.release()

        
        manager.send_threadsafe(
            self.batch_id,
            {
                "type": "video_finished",
                "batch_id": self.batch_id,
                "video_run_id": self.video_run_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Final report
        self.cycle_mgr.final_report()
        self._log_info(
            f"FINAL REPORT: total={self.cycle_mgr.total_cycles} "
            f"passed={self.cycle_mgr.passed} "
            f"failed={self.cycle_mgr.failed} "
            f"unknown={self.cycle_mgr.unknown}"
        )

        flush_pending_logs()
        return True

    # ══════════════════════════════════════════════════════════════
    #  Per-frame dispatch
    # ══════════════════════════════════════════════════════════════

    def _dispatch_frame(self, frame, vis, ZERO_PRED):
        """Route the frame to the current state handler."""

        if self.state == WAIT_FOR_SOCKET:
            return self._handle_wait_for_socket(frame, vis, ZERO_PRED)

        elif self.state == MODEL1_VALIDATION:
            return self._handle_model1_validation(frame, vis, ZERO_PRED)

        elif self.state == MODEL2_SKIP:
            return self._handle_model2_skip(frame, vis, ZERO_PRED)

        elif self.state == MODEL2_VALIDATION:
            return self._handle_model2_validation(frame, vis, ZERO_PRED)

        elif self.state == WAIT_SOCKET_REMOVAL:
            return self._handle_wait_socket_removal(frame, vis, ZERO_PRED)

        elif self.state == CYCLE_FINISHED:
            self._handle_cycle_finished()
            return vis

        return vis

    # ── WAIT_FOR_SOCKET ──────────────────────────────────────────

    def _handle_wait_for_socket(self, frame, vis, ZERO_PRED):
        self.perf.start_section("model1")
        sock_hit = detect_socket(self.yolo_socket, frame, YOLO_SOCKET_CONF)
        self.perf.end_section("model1")
        self.last_sock_hit = sock_hit

        socket_now = sock_hit is not None and sock_hit["class"] == CLS_SOCKET

        self.pred_map = ZERO_PRED
        self.status_dict = {2: "Absent", 3: "Absent", 4: "Absent"}
        self.order_status = "N/A"
        self.detected_seq = []
        self.hand_in_roi = False

        if socket_now:
            x1, y1, x2, y2 = sock_hit["bbox"]
            self.last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.invisible_roi = build_roi(frame.shape, sock_hit["bbox"])
            self._log_info(f"SOCKET_DETECTED conf={sock_hit['conf']:.2f}")
            self._transition(MODEL1_VALIDATION)

        vis = draw_socket_box(vis, sock_hit)
        return vis

    # ── MODEL1_VALIDATION ────────────────────────────────────────

    def _handle_model1_validation(self, frame, vis, ZERO_PRED):
        # Socket detection
        self.perf.start_section("model1")
        sock_hit = detect_socket(self.yolo_socket, frame, YOLO_SOCKET_CONF)
        self.perf.end_section("model1")
        self.last_sock_hit = sock_hit

        socket_now = sock_hit is not None and sock_hit["class"] == CLS_SOCKET

        # Cache socket centre
        if socket_now:
            x1, y1, x2, y2 = sock_hit["bbox"]
            self.last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.invisible_roi = build_roi(frame.shape, sock_hit["bbox"])
            self.m1_socket_absent_streak = 0
        else:
            self.m1_socket_absent_streak += 1

        # Safety abort: if socket is clearly gone
        if self.m1_socket_absent_streak >= self.config.socket_loss_abort_frames:
            self.socket_lost_ct += 1
            self._log_warning(
                f"SOCKET_LOST during MODEL1_VALIDATION "
                f"(absent {self.m1_socket_absent_streak} frames) — aborting")
            self._emit_event("abort", {"reason": "socket_lost", "state": MODEL1_VALIDATION})
            self._reset_all()
            self._transition(WAIT_FOR_SOCKET)
            vis = draw_socket_box(vis, sock_hit)
            return vis

        # Hand check
        self.perf.start_section("hand")
        self.hand_in_roi = (
            detect_hand_in_roi(
                self.yolo_pose, frame, self.invisible_roi, YOLO_POSE_CONF)
            if self.invisible_roi is not None else False
        )
        self.perf.end_section("hand")

        if self.hand_in_roi:
            self.hand_ignored_ct += 1
            self._log_debug(
                f"HAND_IGNORED (model1, valid={self.m1_valid_total}"
                f"/{self.config.model1_frame_count})")
            vis = draw_socket_box(vis, sock_hit)
            return vis

        # Valid frame (not hand)
        self.m1_valid_total += 1
        if socket_now:
            self.m1_valid_pass += 1

        self._log_debug(
            f"[M1] valid={self.m1_valid_total}"
            f"/{self.config.model1_frame_count}, "
            f"pass={self.m1_valid_pass}"
            f"/{self.config.model1_pass_frames}")

        # Check if M1 window is complete
        if self.m1_valid_total >= self.config.model1_frame_count:
            if self.m1_valid_pass >= self.config.model1_pass_frames:
                self._log_info(
                    f"[M1] PASSED ({self.m1_valid_pass}"
                    f"/{self.config.model1_frame_count} valid frames had socket)")
                self.cycle_start_frame = self.frame_idx

                # Start cycle + create placeholder DB row
                self.cycle_mgr.start_cycle()
                self._log_info(
                    f"[CYCLE] #{self.cycle_mgr.cycle_no:03d} "
                    f"STARTED at frame {self.frame_idx}")

                CycleService.create_placeholder(
                    db=self.db,
                    video_run_id=self.video_run_id,
                    cycle_number=self.cycle_mgr.cycle_no,
                    start_frame=self.frame_idx,
                )
                self._log_debug("[DB] Cycle placeholder row created")

                self.vote_counter.reset()
                self._reset_m1()
                self._transition(MODEL2_SKIP)
            else:
                self._log_warning(
                    f"[M1] FAILED (only {self.m1_valid_pass}"
                    f"/{self.config.model1_frame_count} valid frames had "
                    f"socket, needed {self.config.model1_pass_frames})")
                self._reset_all()
                self._transition(WAIT_FOR_SOCKET)

        vis = draw_socket_box(vis, sock_hit)
        return vis

    # ── MODEL2_SKIP ──────────────────────────────────────────────

    def _handle_model2_skip(self, frame, vis, ZERO_PRED):
        # Socket detection only (cheapest check)
        self.perf.start_section("model1")
        sock_hit = detect_socket(self.yolo_socket, frame, YOLO_SOCKET_CONF)
        self.perf.end_section("model1")
        self.last_sock_hit = sock_hit

        socket_now = sock_hit is not None and sock_hit["class"] == CLS_SOCKET

        if socket_now:
            x1, y1, x2, y2 = sock_hit["bbox"]
            self.last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.invisible_roi = build_roi(frame.shape, sock_hit["bbox"])

        # Abort if socket lost mid-skip
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET
        if no_socket_now:
            self.socket_absent_count += 1
        else:
            self.socket_absent_count = 0

        if self.socket_absent_count >= self.config.socket_loss_abort_frames:
            self._log_warning(
                f"[M2_SKIP] socket lost mid-skip "
                f"(absent {self.socket_absent_count} frames), "
                f"aborting cycle #{self.cycle_mgr.cycle_no:03d}")
            CycleService.discard_placeholder(
                db=self.db,
                video_run_id=self.video_run_id,
                cycle_number=self.cycle_mgr.cycle_no,
            )
            if self.cycle_mgr.active:
                if self.cycle_mgr.writer:
                    self.cycle_mgr.writer.release()
                    self.cycle_mgr.writer = None
                self.cycle_mgr.active = False
                # Clean up temp file
                if self.cycle_mgr.temp_path and os.path.exists(self.cycle_mgr.temp_path):
                    os.remove(self.cycle_mgr.temp_path)
            self._reset_all()
            self._transition(WAIT_FOR_SOCKET)
            vis = draw_socket_box(vis, sock_hit)
            return vis

        # Skip counter (raw frames, not hand-filtered)
        self.m2_skipped_ct += 1
        self.m2_skip_count += 1
        self._log_debug(
            f"[M2_SKIP] frame {self.m2_skip_count}"
            f"/{self.config.model2_start_skip_frame}")

        if self.m2_skip_count >= self.config.model2_start_skip_frame:
            self.m2_skip_count = 0
            self.socket_absent_count = 0
            self.seg_engine.reset()  # fresh start for tube detection
            self._transition(MODEL2_VALIDATION)

        vis = draw_socket_box(vis, sock_hit)
        return vis

    # ── MODEL2_VALIDATION ────────────────────────────────────────

    def _handle_model2_validation(self, frame, vis, ZERO_PRED):
        # Socket detection
        self.perf.start_section("model1")
        sock_hit = detect_socket(self.yolo_socket, frame, YOLO_SOCKET_CONF)
        self.perf.end_section("model1")
        self.last_sock_hit = sock_hit

        socket_now = sock_hit is not None and sock_hit["class"] == CLS_SOCKET
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET

        if socket_now:
            x1, y1, x2, y2 = sock_hit["bbox"]
            self.last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.invisible_roi = build_roi(frame.shape, sock_hit["bbox"])
            self.socket_absent_count = 0
        elif no_socket_now:
            self.socket_absent_count += 1

        # Abort if socket lost
        if self.socket_absent_count >= self.config.socket_loss_abort_frames:
            self.socket_lost_ct += 1
            self._log_warning(
                f"[M2] socket lost during validation "
                f"(absent {self.socket_absent_count} frames), "
                f"aborting cycle #{self.cycle_mgr.cycle_no:03d}")
            self._emit_event("abort", {"reason": "socket_lost", "state": MODEL2_VALIDATION})
            CycleService.discard_placeholder(
                db=self.db,
                video_run_id=self.video_run_id,
                cycle_number=self.cycle_mgr.cycle_no,
            )
            if self.cycle_mgr.active:
                if self.cycle_mgr.writer:
                    self.cycle_mgr.writer.release()
                    self.cycle_mgr.writer = None
                self.cycle_mgr.active = False
                if self.cycle_mgr.temp_path and os.path.exists(self.cycle_mgr.temp_path):
                    os.remove(self.cycle_mgr.temp_path)
            self._reset_all()
            self._transition(WAIT_FOR_SOCKET)
            vis = draw_socket_box(vis, sock_hit)
            return vis

        # Hand check
        self.perf.start_section("hand")
        self.hand_in_roi = (
            detect_hand_in_roi(
                self.yolo_pose, frame, self.invisible_roi, YOLO_POSE_CONF)
            if self.invisible_roi is not None else False
        )
        self.perf.end_section("hand")

        if self.hand_in_roi:
            self.hand_ignored_ct += 1
            self._log_debug(
                f"HAND_IGNORED (model2, valid={self.m2_valid_count}"
                f"/{self.config.model2_frame_count})")
            vis = draw_socket_box(vis, sock_hit)
            return vis

        if not socket_now:
            # Socket not detected but not enough consecutive absences to abort
            # Still a valid frame but socket absent — don't run tube detection
            vis = draw_socket_box(vis, sock_hit)
            return True

        # ── Valid frame: run Model2 (tube detection) ─────────────
        self.cycle_total_frames += 1

        self.perf.start_section("model2")
        self.pred_map = self.seg_engine.infer(
            frame,
            socket_centre=self.last_socket_centre,
            apply_identity_lock=True
        )

        from app.services.ml_adapter import (
            restrict_mask_to_socket_roi, evaluate_tube_order,
            MASK_ROI_CLASSES, MASK_ROI_SHAPE, MASK_ROI_AUTO_SCALE,
            MASK_ROI_RADIUS, MASK_ROI_RADIUS_X, MASK_ROI_RADIUS_Y,
            MASK_ROI_RADIUS_UP, MASK_ROI_RADIUS_DOWN,
            MASK_ROI_RADIUS_LEFT, MASK_ROI_RADIUS_RIGHT,
            MASK_ROI_OFFSET_X, MASK_ROI_OFFSET_Y, MASK_ROI_POLYGON
        )

        sock_bbox = sock_hit["bbox"] if sock_hit else None
        sock_w = (sock_bbox[2] - sock_bbox[0]) if sock_bbox else None
        sock_h = (sock_bbox[3] - sock_bbox[1]) if sock_bbox else None
        bbox_size = (sock_w, sock_h) if sock_w and sock_h else None

        self.pred_map = restrict_mask_to_socket_roi(
            self.pred_map, center=self.last_socket_centre,
            bbox_size=bbox_size,
            classes=MASK_ROI_CLASSES,
            shape=MASK_ROI_SHAPE,
            auto_scale=MASK_ROI_AUTO_SCALE,
            radius=MASK_ROI_RADIUS,
            radius_x=MASK_ROI_RADIUS_X,
            radius_y=MASK_ROI_RADIUS_Y,
            radius_up=MASK_ROI_RADIUS_UP,
            radius_down=MASK_ROI_RADIUS_DOWN,
            radius_left=MASK_ROI_RADIUS_LEFT,
            radius_right=MASK_ROI_RADIUS_RIGHT,
            offset_x=MASK_ROI_OFFSET_X,
            offset_y=MASK_ROI_OFFSET_Y,
            polygon=MASK_ROI_POLYGON
        )

        self.status_dict, raw_order, self.detected_seq, self.current_dbg = evaluate_tube_order(
            self.pred_map, sock_bbox,
            debug=self.enable_debug
        )
        self.perf.end_section("model2")

        # Gate chain
        stable_order = self.seq_gate.update(raw_order, self.detected_seq)
        gate_result = self.anomaly_gate.update(stable_order)
        self.order_status = gate_result

        # Vote
        self.vote_counter.record(gate_result)
        self.m2_valid_count += 1
        self.infer_frames += 1

        self._log_debug(
            f"[M2] valid={self.m2_valid_count}"
            f"/{self.config.model2_frame_count}, "
            f"pred={gate_result}, "
            f"votes(ok={self.vote_counter.normal_votes}, "
            f"anom={self.vote_counter.anomaly_votes})")

        # Track peak status
        if any(v == "Present" for v in self.status_dict.values()):
            self.peak_status = dict(self.status_dict)
            self.peak_seq = list(self.detected_seq)
            self.peak_order = self.order_status

        # Draw segmentation overlay
        has_tubes = any(
            (self.pred_map == ci).any() for ci in (2, 3, 4))
        if has_tubes:
            vis = draw_seg_overlay(vis, self.pred_map)
        elif self.seg_engine.last_raw_pred is not None:
            vis = draw_raw_argmax_fallback(vis, self.seg_engine.last_raw_pred)

        vis = draw_socket_box(vis, sock_hit)

        # ── Early exit check ─────────────────────────────────────
        if self.vote_counter.normal_votes >= self.config.model2_pass_frames:
            self.verdict = "NORMAL"
            self._log_info(
                f"[M2] EARLY VERDICT: NORMAL "
                f"({self.vote_counter.normal_votes}"
                f"/{self.config.model2_frame_count} OK votes)")
            self._transition(WAIT_SOCKET_REMOVAL)
            return vis

        if self.vote_counter.anomaly_votes >= self.config.model2_pass_frames:
            self.verdict = "ANOMALY"
            self._log_info(
                f"[M2] EARLY VERDICT: ANOMALY "
                f"({self.vote_counter.anomaly_votes}"
                f"/{self.config.model2_frame_count} anomaly votes)")
            self._transition(WAIT_SOCKET_REMOVAL)
            return vis

        # Window exhausted without threshold met
        if self.m2_valid_count >= self.config.model2_frame_count:
            if self.vote_counter.normal_votes > self.vote_counter.anomaly_votes:
                self.verdict = "NORMAL"
            elif self.vote_counter.anomaly_votes > self.vote_counter.normal_votes:
                self.verdict = "ANOMALY"
            else:
                self.verdict = "UNKNOWN"
            self._log_info(
                f"[M2] INCONCLUSIVE after {self.config.model2_frame_count} "
                f"valid frames, majority → {self.verdict} "
                f"(ok={self.vote_counter.normal_votes}, "
                f"anom={self.vote_counter.anomaly_votes})")
            self._transition(WAIT_SOCKET_REMOVAL)

        return vis

    # ── WAIT_SOCKET_REMOVAL ──────────────────────────────────────

    def _handle_wait_socket_removal(self, frame, vis, ZERO_PRED):
        # Model1 ONLY — cheapest possible state
        self.perf.start_section("model1")
        sock_hit = detect_socket(self.yolo_socket, frame, YOLO_SOCKET_CONF)
        self.perf.end_section("model1")
        self.last_sock_hit = sock_hit

        socket_now = sock_hit is not None and sock_hit["class"] == CLS_SOCKET
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET

        self.hand_in_roi = False  # Hand check OFF in this state

        if socket_now:
            if self.socket_absent_count > 0:
                self._log_debug(
                    f"SOCKET_REGAINED (flicker absorbed, "
                    f"{self.socket_absent_count} frames)")
            self.socket_absent_count = 0
        elif no_socket_now:
            self.socket_absent_count += 1
            if self.socket_absent_count == 1:
                self._log_debug("SOCKET_ABSENT (streak: 1)")

        if self.socket_absent_count >= self.config.socket_absent_frames:
            self._log_info(
                f"Socket absent for {self.socket_absent_count} "
                f"consecutive frames — ending cycle")
            self._transition(CYCLE_FINISHED)

        vis = draw_socket_box(vis, sock_hit)
        return vis

    # ── CYCLE_FINISHED ───────────────────────────────────────────

    def _handle_cycle_finished(self):
        self._finalize_current_cycle()
        self._reset_all()
        self._transition(WAIT_FOR_SOCKET)

    def _finalize_current_cycle(self, abort=False):
        """End the current cycle, write DB + Excel."""
        if not self.cycle_mgr.active:
            return

        final_verdict = self.verdict or self.vote_counter.final_verdict()
        anomaly_ratio = (
            self.vote_counter.anomaly_votes / max(self.vote_counter.total, 1))

        # Show verdict card on the last frame
        if self.last_vis is not None:
            stats_card = {
                "total_frames": self.cycle_total_frames,
                "warmup_frames": 0,
                "infer_frames": self.infer_frames,
                "normal_votes": self.vote_counter.normal_votes,
                "anomaly_votes": self.vote_counter.anomaly_votes,
                "anomaly_ratio": anomaly_ratio,
            }
            card = draw_final_verdict_overlay(
                self.last_vis, final_verdict,
                cycle_no=self.cycle_mgr.cycle_no, stats=stats_card)
            self.cycle_mgr.hold_final_frame(card)

        seq_display = (
            " > ".join(TUBE_SHORT[c] for c in self.peak_seq)
            if self.peak_seq else "-")
        display_verdict = _ORDER_DISPLAY.get(
            self.peak_order if self.peak_order != "N/A" else final_verdict,
            "N/A")

        extra = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(self.video_path),
            "socket": "Present",
            "tube_blue": self.peak_status.get(2, "Absent"),
            "trans_mid_tube": self.peak_status.get(3, "Absent"),
            "trans_end_tube": self.peak_status.get(4, "Absent"),
            "detected_sequence": seq_display,
            "tube_order_result": display_verdict,
            "warmup_frames": 0,
            "infer_frames": self.infer_frames,
            "ok_votes": self.vote_counter.normal_votes,
            "anomaly_votes": self.vote_counter.anomaly_votes,
            "anomaly_ratio": round(anomaly_ratio, 4),
            "total_frames": self.cycle_total_frames,
            "avg_fps": float(self._fps_ema),
        }

        summary = self.cycle_mgr.end_cycle(final_verdict, extra_metrics=extra)

        duration_s = time.time() - (self.cycle_mgr.start_time or time.time())
        self._log_info(
            f"[CYCLE] #{self.cycle_mgr.cycle_no:03d} ENDED: "
            f"{final_verdict} (duration={duration_s:.1f}s)")
        self._log_info(
            f"[RUNNING TOTAL] PASSED={self.cycle_mgr.passed} "
            f"FAILED={self.cycle_mgr.failed} "
            f"UNKNOWN={self.cycle_mgr.unknown}")

        # Update the placeholder DB row with final values
        if summary:
            CycleService.finalize_cycle(
                db=self.db,
                video_run_id=self.video_run_id,
                cycle_number=summary["cycle_no"],
                start_frame=self.cycle_start_frame,
                end_frame=self.frame_idx,
                duration_seconds=duration_s,
                final_verdict=final_verdict,
                output_video_path=summary.get("output_path", ""),
                tube_blue=extra.get("tube_blue"),
                transition_middle=extra.get("trans_mid_tube"),
                transition_end=extra.get("trans_end_tube"),
                detected_sequence=extra.get("detected_sequence"),
                tube_order_result=extra.get("tube_order_result"),
                anomaly_ratio=anomaly_ratio,
                ok_votes=self.vote_counter.normal_votes,
                anomaly_votes=self.vote_counter.anomaly_votes,
                total_frames=self.cycle_total_frames,
                warmup_frames=0,
                inference_frames=self.infer_frames,
                average_fps=self._fps_ema,
            )
            self._log_debug("[DB] Cycle row finalized in SQLite")

            # Excel — secondary output, not the source of truth
            try:
                result = append_to_excel(summary, self.output_dir)
                if result:
                    self._log_info(
                        f"[EXCEL] Row #{result.get('row', '?')} "
                        f"(cycle #{summary['cycle_no']}) → "
                        f"{result.get('path', 'unknown')}")
            except Exception as ex:
                self._log_warning(f"[EXCEL] Write failed: {ex}")

    # ══════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════

    def _get_hud_state(self) -> str:
        """Map state machine state to old HUD state for rendering."""
        if self.state == MODEL2_VALIDATION:
            # Use the order_status to pick the correct HUD state
            return {
                "OK": STATE_NORMAL,
                "ANOMALY": STATE_ANOMALY,
                "PARTIAL": STATE_PARTIAL,
            }.get(self.order_status, STATE_INSPECT)
        if self.state == MODEL1_VALIDATION and self.hand_in_roi:
            return STATE_HAND
        if self.state == MODEL2_VALIDATION and self.hand_in_roi:
            return STATE_HAND
        return _HUD_STATE_MAP.get(self.state, STATE_IDLE)

    def _emit_event(self, event_name: str, payload: dict = None):
        """Emit a discrete lifecycle event for UI syncing."""
        msg = {
            "type": "event",
            "batch_id": self.batch_id,
            "video_run_id": self.video_run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_name
        }
        if payload:
            msg.update(payload)
        manager.send_threadsafe(self.batch_id, msg)

    def _emit_capabilities(self):
        """Send static capabilities of this backend engine."""
        manager.send_threadsafe(
            self.batch_id,
            {
                "type": "capabilities",
                "batch_id": self.batch_id,
                "video_run_id": self.video_run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "models": ["Socket Detector", "Hand Detector", "Tube Detector"]
            }
        )

    def _emit_status(self):
        """Emit structured, throttled telemetry for the dashboard."""
        now = time.time()
        # Throttle to ~10Hz (every 100ms)
        if now - self._last_status_time < 0.1:
            return
        self._last_status_time = now
        
        # Calculate state progress
        prog_curr = 0
        prog_tgt = 0
        prog_lbl = ""
        if self.state == MODEL1_VALIDATION:
            prog_curr = self.m1_valid_total
            prog_tgt = self.config.model1_frame_count
            prog_lbl = "valid frames"
        elif self.state == MODEL2_SKIP:
            prog_curr = self.m2_skip_count
            prog_tgt = self.config.model2_start_skip_frame
            prog_lbl = "skipped"
        elif self.state == MODEL2_VALIDATION:
            prog_curr = self.m2_valid_count
            prog_tgt = self.config.model2_frame_count
            prog_lbl = "inspected"

        socket_running = self.state in [WAIT_FOR_SOCKET, MODEL1_VALIDATION, MODEL2_SKIP, MODEL2_VALIDATION, WAIT_SOCKET_REMOVAL]
        hand_running = self.state in [MODEL1_VALIDATION, MODEL2_VALIDATION]
        tube_running = self.state == MODEL2_VALIDATION

        status_payload = {
            "state": self.state,
            "progress": {
                "current": prog_curr,
                "target": prog_tgt,
                "label": prog_lbl
            },
            "models": {
                "Socket Detector": {
                    "running": socket_running,
                    "elapsed_ms": int(self.perf._section_totals.get("model1", 0.0)) if socket_running else 0,
                    "confidence": round(float(self.last_sock_hit["conf"]), 2) if self.last_sock_hit else 0.0
                },
                "Hand Detector": {
                    "running": hand_running,
                    "elapsed_ms": int(self.perf._section_totals.get("hand", 0.0)) if hand_running else 0,
                    "result": "PRESENT" if self.hand_in_roi else "ABSENT"
                },
                "Tube Detector": {
                    "running": tube_running,
                    "elapsed_ms": int(self.perf._section_totals.get("model2", 0.0)) if tube_running else 0
                }
            },
            "decision": {
                "socket": "Present" if (self.last_sock_hit and self.last_sock_hit["class"] == CLS_SOCKET) else "Absent",
                "hand": "Present" if self.hand_in_roi else "Absent",
                "tubes": {TUBE_SHORT[ci]: self.status_dict.get(ci, "Absent") for ci in (2, 3, 4)},
                "sequence": [TUBE_SHORT[c] for c in self.detected_seq] if self.detected_seq else [],
                "vote": self.order_status
            },
            "recording": {
                "active": self.state != WAIT_FOR_SOCKET,
                "output": f"cycle_{self.cycle_mgr.cycle_no}.mp4" if self.cycle_mgr else "---",
                "frames_written": self.cycle_total_frames
            },
            "perf": {
                "gpu": self.perf.get_gpu_utilization(),
                "cpu": self.perf.get_cpu_utilization(),
                "fps": round(float(self._fps_ema), 1),
                "frame_time_ms": int(self.perf._section_totals.get("total", 0.0)),
                "encode_ms": int(self.perf._section_totals.get("encode", 0.0))
            },
            "frame_stats": {
                "frames_read": self.frame_idx,
                "hand_ignored": self.hand_ignored_ct,
                "socket_lost": self.socket_lost_ct,
                "m2_skipped": self.m2_skipped_ct
            },
            "cycle_stats": {
                "cycle": self.cycle_mgr.cycle_no if self.cycle_mgr else 0,
                "current_verdict": self.order_status,
                "ok_votes": self.vote_counter.normal_votes if self.vote_counter else 0,
                "anomaly_votes": self.vote_counter.anomaly_votes if self.vote_counter else 0,
                "elapsed_s": round(time.time() - (self.cycle_mgr.start_time or time.time()), 1) if self.cycle_mgr else 0.0
            }
        }
        manager.send_threadsafe(
            self.batch_id,
            {
                "type": "status",
                "batch_id": self.batch_id,
                "video_run_id": self.video_run_id,
                "timestamp": datetime.utcnow().isoformat(),
                **status_payload,
            },
        )

    def _emit_frame(self, vis, src_w):
        """Emit base64 frame for live preview."""
        self.perf.start_section("encode")
        vis_out = vis
        h, w = vis.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            vis_out = cv2.resize(vis, (1280, int(h * scale)))
        success, buffer = cv2.imencode(
            '.jpg', vis_out, [cv2.IMWRITE_JPEG_QUALITY, 65])
        self.perf.end_section("encode")
        if success:
            b64 = base64.b64encode(buffer).decode('utf-8')
            manager.send_threadsafe(
                self.batch_id,
                {
                    "type": "frame",
                    "batch_id": self.batch_id,
                    "video_run_id": self.video_run_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "base64": b64,
                },
            )

    def _get_video_run(self):
        """Get the current video run ORM object."""
        from app.crud.video_run import get_video_run
        return get_video_run(self.db, self.video_run_id)
