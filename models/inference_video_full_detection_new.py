#!/usr/bin/env python3
"""
Diagnostic Inference Engine — v50_MergedSingleVideo
=====================================================
Base inference engine : v45 (live inference every frame, no mask freeze,
                         FIX-I1..FIX-I8 boundary/threshold/morphology fixes)
Tube-order logic       : v49 FIX-I18 — Nearest-Pixel Angular Gate
Cycle/production logic : v49 — CycleManager, production status bar,
                         cycle-based Excel logging, single-video / multi-cycle
                         processing loop

This file processes ONE video and treats every socket-attach → socket-removal
span within that video as its own inspection "cycle", exactly like v49, but
using v45's simpler (non-identity-hysteresis, non-solid-fill, non-dilate)
segmentation pipeline.

NEW in this revision (v50d)  —  GPU UTILIZATION FIX (merged from v51)
----------------------------
  [FIX-I38] Hard GPU enforcement + diagnostics.
  [FIX-I39] Full GPU utilization pass — TF32, FP16 (half precision) for
            segmentation UNet++ AND both YOLO models, GPU-side frame
            normalization (moved off the CPU), realistic-size YOLO warmup,
            non_blocking tensor transfers, GPU sanity benchmark.

NEW in this revision (v50e)  —  MASK WIDTH INCREASE
----------------------------
  [FIX-I41] MASK_DILATE_SZ raised from 6 -> 16 (default) so the tube-class
            overlays (2/3/4) render as thick, filled bands instead of thin
            hairline outlines. Purely a rendering/dilation-kernel change —
            no effect on the underlying segmentation, identity lock, or
            tube-order evaluation logic. Tune further with --dilate_sz.

NEW in this revision (v50f)  —  SOLID / CONTINUOUS THICK-BAND OVERLAY
----------------------------
  [FIX-I42] MASK_DILATE_SZ raised again, 16 -> 28 (default), so the tube
            overlays read as thick painted bands (matching the reference
            "thick overlay" look) instead of merely "wider hairlines".
            Tune further with --dilate_sz.
  [FIX-I43] NEW morphological CLOSE pass, run AFTER dilation, with its own
            (larger) kernel MASK_CLOSE_SZ. Dilation alone only thickens
            pixels that are already there — if the raw segmentation mask
            for a tube is patchy/broken into several small blobs along its
            length (common right where CLASS_CONF_THR trims low-confidence
            pixels), the dilated result is still several separate thick
            blobs with visible gaps between them. A CLOSE with a bigger
            kernel bridges small gaps between nearby same-class blobs so
            the band reads as one continuous, unbroken stroke along the
            tube — exactly like the reference image. Same
            protect-from-other-classes rule as dilation applies: a class
            is never allowed to close/grow into pixels already claimed by
            a DIFFERENT non-background class. Tune with --close_sz.
  [FIX-I44] OVERLAY_ALPHA raised 0.65 -> 0.90 (default) so the painted
            bands render as flat, saturated color (matching the reference)
            instead of a semi-transparent tint over the video. Tune with
            --overlay_alpha.
"""

import os, sys, time, platform, argparse, math, shutil, zipfile, json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# [FIX] Windows cp1252 consoles cannot encode box-drawing characters (U+2550 ═,
# U+2588 █, etc.) that appear in several print() calls below. safe_print()
# transparently falls back to ASCII replacement when the current stdout cannot
# encode the text, so the application never crashes on a UnicodeEncodeError
# regardless of the console encoding.
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace unencodable chars with '?' and retry
        safe_args = [
            a.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')
            if isinstance(a, str) else a
            for a in args
        ]
        print(*safe_args, **kwargs)

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
if platform.system() == "Windows":
    _BASE     = r"C:\Users\sonar\Desktop\autodistill-grounded-sam-2"
    _OUT_BASE = r"C:\Users\sonar\Desktop\UNet++_results"
else:
    _BASE     = "/mnt/c/Users/sonar/Desktop/autodistill-grounded-sam-2"
    _OUT_BASE = "/mnt/c/Users/sonar/Desktop/UNet++_results"

DEFAULT_VIDEO = os.path.join(_BASE, "input_videos", "shift_recording.mp4")
DEFAULT_MODEL = os.path.join(_BASE, "outputs", "finetune_augmentation_v2",
                             "run_20260623_140144", "best_model_finetuned_v2.pth")
DEFAULT_YOLO  = ""
DEFAULT_POSE  = "yolov8n-pose.pt"

SHOW_PREVIEW      = True
MAX_EXCEL_RETRIES = 5

# ══════════════════════════════════════════════════════════════════════════════
#  WARMUP & VERDICT CONFIG
# ══════════════════════════════════════════════════════════════════════════════
WARMUP_FRAMES      = 20
MAX_WARMUP_RETRIES = 3
VERDICT_THR        = 0.50
VERDICT_HOLD_SEC   = 2.0

# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION CONFIG  (v45 values)
# ══════════════════════════════════════════════════════════════════════════════
NUM_CLASSES    = 5
IMG_SIZE       = (512, 512)
# [FIX-I44] Raised 0.65 -> 0.90 (default) — a higher alpha means the overlay
# layer dominates the blend, so the tube colors render as flat, saturated
# paint instead of a translucent tint you can see the underlying video
# through. Tune with --overlay_alpha.
OVERLAY_ALPHA  = 0.90

# [FIX-I38] DEVICE is now resolved (with full diagnostics + GPU sanity
# benchmark) by resolve_device() further down, right before it's used in
# run_single_video(). Kept as a module-level global placeholder so every
# function below that references DEVICE keeps working unchanged; only the
# *value* it ends up holding is set for real at startup.
DEVICE = "cpu"

# [FIX-I39] Whether the segmentation model + both YOLO models run in FP16.
# Resolved for real in run_single_video() (default True whenever DEVICE is
# a CUDA device).
USE_HALF = False

# [FIX-I39] Realistic (H, W) used to build the YOLO warmup dummy frame so
# cudnn.benchmark autotunes the kernel shape that will actually be used by
# real video frames, instead of a throwaway 64x64 shape.
GPU_WARMUP_HW = (720, 1280)

# [FIX-I39] GPU-resident ImageNet mean/std used to normalize frames on the
# GPU instead of on the CPU (see raw_infer). Populated once DEVICE/USE_HALF
# are resolved, in resolve_device().
_NORM_MEAN = None
_NORM_STD  = None

# [FIX-I1] Reduced EMA alpha — less temporal blur at Mid/End boundary
EMA_ALPHA = 0.20

# [FIX-I2] Tube boundary sharpening — temperature rescaling for classes 3 & 4
TUBE_BOUNDARY_SHARPENING = True
# [FIX-I8] Softened from 0.5 -> 0.85 (was too aggressive, erased class-3 at
# the connector junction where classes 3/4 are genuinely close together).
TUBE_SHARPNESS_TEMP      = 0.85

# [FIX-I3 / FIX-I8] Loosened confidence thresholds for classes 3 & 4
CLASS_CONF_THR = {
    1: 0.25,   # device
    2: 0.20,   # tube_blue
    3: 0.35,   # trans_mid_tube  [FIX-I8] was 0.35
    4: 0.22,   # trans_end_tube  [FIX-I8] was 0.35
}

# [FIX-I6] Sequence stability gate
MIN_SEQ_STABLE = 3

# [FIX-I8] Mask/pipeline debug instrumentation — per-stage pixel-count trace.
# Toggle live with the 'm' key, or pass --mask_debug on the CLI.
MASK_DEBUG = False

CLASS_INFO = {
    0: ("background",     (0,   0,   0),   False),
    1: ("device",         (0,   128, 0),   False),
    2: ("tube_blue",      (0,   255, 255), True),
    3: ("trans_mid_tube", (255, 0,   0),   True),
    4: ("trans_end_tube", (255, 0,   255), True),
}
TUBE_LABELS = {2: "Yellow  tube_blue", 3: "Blue    mid-tube", 4: "Pink    end-tube"}
TUBE_SHORT  = {2: "Yel", 3: "Blu", 4: "Pnk"}

EXPECTED_SEQ    = [2, 3, 4]

# [FIX-I18, from v49] Search radius (px) around the socket centre used to
# find each tube's nearest-to-socket pixel for angular order detection.
# [FIX-I27] This now defines the half-width/half-height of a RECTANGULAR
# search region (a square of side 2*NEAREST_SEARCH_RADIUS centred on the
# socket) instead of a circle — replaces the old circular search region.
NEAREST_SEARCH_RADIUS = 350

# [FIX-I30] Radius (px, half-side of a square) around the socket centre
# OUTSIDE of which tube-class pixels (2/3/4) are suppressed entirely, right
# after segmentation inference. This keeps the visible/tracked mask
# confined to the socket-adjacent stretch of each tube and discards
# detections out along the coiled/looped sections further from the socket.
# Kept independent from NEAREST_SEARCH_RADIUS (used later, only for angle
# evaluation) so the two can be tuned separately if needed.
MASK_ROI_RADIUS  = 160
MASK_ROI_CLASSES = (2, 3, 4)
# [FIX-I31] Shape of the socket-adjacent keep-region. "circle" hugs the
# socket much more tightly than "square" at the same radius (a square keeps
# its full corner-to-corner reach on the diagonals, which is exactly where
# a nearby coiled tube loop tends to sit) — use "circle" unless you have a
# specific reason to keep the old square envelope.
MASK_ROI_SHAPE   = "quad_ellipse"   # "circle" | "square" | "ellipse" | "quad_ellipse" | "polygon"
# [FIX-I32] Used only when MASK_ROI_SHAPE == "ellipse". Independent
# horizontal/vertical radii let the keep-region be taller than it is wide
# (or vice versa) instead of a single uniform radius in every direction —
# useful when the tube run you want to keep extends much further in one
# direction from the socket than the other.
MASK_ROI_RADIUS_X = 160
MASK_ROI_RADIUS_Y = 330
# [FIX-I32] Shifts the ROI centre away from the socket bbox centre before
# applying the radius/ellipse test. Negative Y moves the kept region
# UP (toward smaller pixel-row values) relative to the socket, positive Y
# moves it down; same convention for X (negative = left, positive = right).
# Use this when the part of the tube you want to keep sits mostly on one
# side of the socket rather than symmetrically around it.
MASK_ROI_OFFSET_X = 0
MASK_ROI_OFFSET_Y = -120

# [FIX-I34] "quad_ellipse" — an ellipse whose radius is allowed to differ
# independently in each of the 4 directions from the (offset) centre, i.e.
# a separate radius for "up", "down", "left", "right". This is the
# simplest way to match an irregular, non-symmetric keep-region (extends
# far up-right but not much down-left, etc.) without needing to click a
# full custom polygon — just 4 numbers to tune instead of clicking points.
# For a candidate pixel at offset (dx, dy) from the centre:
#   rx = MASK_ROI_RADIUS_RIGHT if dx >= 0 else MASK_ROI_RADIUS_LEFT
#   ry = MASK_ROI_RADIUS_DOWN  if dy >= 0 else MASK_ROI_RADIUS_UP
#   keep if (dx/rx)^2 + (dy/ry)^2 <= 1
MASK_ROI_RADIUS_UP    = 330   # reach above the centre (toward smaller y)
MASK_ROI_RADIUS_DOWN  = 120   # reach below the centre (toward larger y)
MASK_ROI_RADIUS_LEFT  = 160   # reach to the left of the centre
MASK_ROI_RADIUS_RIGHT = 220   # reach to the right of the centre

# [FIX-I35] Auto-scale the quad_ellipse radii from the socket's own YOLO
# bbox size instead of fixed pixel counts. A socket bbox is a stable,
# per-frame available reference for "how zoomed-in/close this particular
# camera setup is" — a socket that fills more of the frame (closer camera,
# tighter crop, different resolution, etc.) means the whole tube layout
# will also be proportionally bigger in pixels, so scaling the ROI off the
# bbox's own width/height generalizes across videos/setups without
# re-tuning MASK_ROI_RADIUS_* by hand each time.
#
# When enabled, radius_up/down are computed as MULT * bbox_height, and
# radius_left/right as MULT * bbox_width, OVERRIDING the fixed
# MASK_ROI_RADIUS_UP/DOWN/LEFT/RIGHT constants above for shape=
# "quad_ellipse" (those constants remain the fallback whenever no socket
# bbox size is available yet, e.g. before the socket has ever been seen).
# The MULT_* defaults below were derived from the fixed defaults above
# against a representative bbox size from the calibration footage, so
# switching this on should reproduce very similar framing to what you've
# already tuned, but expressed as a resolution/zoom-independent ratio.
MASK_ROI_AUTO_SCALE = True
MASK_ROI_MULT_UP    = 1.3   # x bbox HEIGHT
MASK_ROI_MULT_DOWN   = 0.65  # x bbox HEIGHT
MASK_ROI_MULT_LEFT   = 0.75  # x bbox WIDTH
MASK_ROI_MULT_RIGHT  = 1.05  # x bbox WIDTH

# [FIX-I36] Hard EXCLUSION zone — a separate keep-out ellipse, independent
# of (and applied AFTER) the inclusion ROI above. Use this for a known
# distractor object near the socket (e.g. a different tube/cable that
# happens to sit close enough that it could occasionally be picked up as
# one of the tube classes 2/3/4 due to similar colour/lighting) that you
# always want zeroed out regardless of how the inclusion radii are tuned.
# Unlike the inclusion ROI (which defines what to KEEP), this defines a
# region to always DROP.
#
# The exclusion ellipse is centred at
#   (socket_cx + exclude_offset_x, socket_cy + exclude_offset_y)
# with radii exclude_radius_x / exclude_radius_y. Like the main ROI, it
# auto-scales off the socket bbox size by default so the same relative
# position/size (e.g. "the other cable sits up-and-to-the-left of the
# socket, about this big relative to the socket") generalizes across
# videos shot at different zoom/distance.
MASK_EXCLUDE_ENABLED    = True
MASK_EXCLUDE_CLASSES    = (2, 3, 4)
MASK_EXCLUDE_AUTO_SCALE = True
# Auto-scale multipliers (used when MASK_EXCLUDE_AUTO_SCALE is True)
MASK_EXCLUDE_OFFSET_MULT_X = -1.3   # x bbox WIDTH  (negative = left of socket)
MASK_EXCLUDE_OFFSET_MULT_Y = -0.9   # x bbox HEIGHT (negative = above socket)
MASK_EXCLUDE_RADIUS_MULT_X = 1.1    # x bbox WIDTH
MASK_EXCLUDE_RADIUS_MULT_Y = 1.1    # x bbox HEIGHT
# Fixed-pixel fallback (used when auto-scale is off, or no bbox size yet)
MASK_EXCLUDE_OFFSET_X = -250
MASK_EXCLUDE_OFFSET_Y = -150
MASK_EXCLUDE_RADIUS_X = 200
MASK_EXCLUDE_RADIUS_Y = 200

# [FIX-I37] Motion-gated mask freeze. By default (v45 base engine) live
# inference reruns EVERY frame, so even a perfectly static tube gets a
# slightly different pred_map each frame (fresh model noise, EMA settling,
# etc.) — one source of residual flicker even with the identity lock in
# place. When enabled, once warmup is done the engine instead:
#   1. Runs inference once and "freezes" that pred_map + a reference
#      grayscale crop of the ROI.
#   2. On every subsequent frame, compares the current ROI crop against
#      the frozen reference using dense optical flow. If the fraction of
#      pixels moving more than MASK_FREEZE_MOTION_THR px/frame stays below
#      MASK_FREEZE_MOTION_FRAC, the tube is considered NOT to have moved,
#      and the exact same frozen pred_map object is reused untouched — not
#      just similar, IDENTICAL — so there is zero flicker while static.
#   3. Once real motion is detected (tube nudged, repositioned, etc.), a
#      fresh inference is run and a new frozen snapshot replaces the old
#      one.
# The comparison is always against the ORIGINAL frozen reference frame
# (not just the previous frame), so slow cumulative drift across many
# frames still triggers a refresh once total displacement crosses the
# threshold, rather than only reacting to large frame-to-frame jumps.
MASK_FREEZE_ENABLED     = True
MASK_FREEZE_MOTION_THR  = 2.5    # px/frame — per-pixel optical-flow magnitude considered "moved"
MASK_FREEZE_MOTION_FRAC = 0.04   # fraction of ROI pixels that must exceed the threshold to trigger a refresh

# [FIX-I33] Custom polygon keep-region (kept available, but "quad_ellipse"
# above is the recommended/default method — simpler to tune, no clicking
# required). Used only when MASK_ROI_SHAPE == "polygon". Each entry is a
# (dx, dy) offset FROM THE SOCKET CENTRE (not absolute pixel coordinates),
# so the traced shape stays anchored to the socket even if it shifts
# slightly frame-to-frame. Populate this by running
# tools/pick_roi_polygon.py on a representative frame if you ever do want
# an exact custom boundary.
MASK_ROI_POLYGON = None   # list[(dx, dy), ...] or None (unset)

# [FIX-I8] Lowered 80 -> 40. A real class-3 blob squeezed against the
# connector is often thin/short; requiring 80px in the ROI was causing
# evaluate_tube_order() to report "Absent" even when the segmentation mask
# itself had (a smaller number of) class-3 pixels.
MIN_TUBE_PX     = 40
N_ANOMALY_CONFIRM = 4
LATCH_FRAMES    = 10
# [FIX-I8] Lowered 80 -> 40 in tandem with MIN_TUBE_PX.
MIN_AREA_PX     = 40
MORPH_KERNEL_SZ = 5
PERSIST_FRAMES  = 1

# [FIX-I21 — Mask Width] Dilate the thin tube-class masks (2/3/4) so they
# render with visible width on screen instead of a hairline. Dilation is
# "protected": a class is never allowed to grow into pixels already claimed
# by a DIFFERENT non-background class, so classes still can't bleed into
# each other — only into background/unclaimed pixels.
# [FIX-I41] Raised default 6 -> 16 so the on-screen tube overlays render as
# thick, filled bands (matching a "thick overlay" reference look) instead
# of thin hairline outlines.
# [FIX-I42] Raised again, 16 -> 28, plus a second, larger CLOSE pass
# (MASK_CLOSE_SZ, see FIX-I43) so nearby broken-up blobs along one tube
# fuse into a single continuous painted band instead of just "fatter
# separate blobs". Tune further at runtime with --dilate_sz / --close_sz
# if this is too much/little for your camera resolution.
MASK_DILATE_ENABLED = True
MASK_DILATE_SZ      = 18
MASK_DILATE_CLASSES = (2, 3, 4)

# [FIX-I43] Second-stage morphological CLOSE, run right after the dilation
# above, with its OWN (larger) kernel. Dilation alone only fattens pixels
# that already exist in the mask — if the raw per-frame mask for a tube is
# broken into several separate small blobs along its length (very common
# right at the CLASS_CONF_THR cutoff), dilating each blob just gives you
# several fatter separate blobs, still with visible gaps between them. A
# CLOSE (dilate-then-erode) with a bigger kernel bridges those small gaps
# between nearby same-class blobs, so the whole tube reads as ONE
# continuous solid stroke — this is what actually produces the smooth,
# unbroken "thick painted band" look. Same class-protection rule as the
# dilation step: never allowed to close into pixels already claimed by a
# DIFFERENT non-background class. Tune with --close_sz; set to 0 (or pass
# --no_dilate) to disable.
MASK_CLOSE_ENABLED = True
MASK_CLOSE_SZ       = 40

# [FIX-I22 / FIX-I23 — Tube Identity Lock, now 3-way]
# ---------------------------------------------------------------------------
# SYMPTOM: classes 2 (tube_blue / Yellow), 3 (trans_mid_tube / Blue) and 4
# (trans_end_tube / Pink) are visually/probabilistically close right at the
# connector junction, so a single continuous physical tube can flicker
# between any of these three classes near the socket even though it's the
# same tube all the way in.
#
# FIX: maintain a per-pixel EMA of each of the three tube-class probabilities
# and a per-pixel identity LOCK (2, 3, or 4) that is only allowed to flip
# once the winning class's EMA exceeds the currently-locked class's EMA by
# more than TUBE_IDENTITY_MARGIN — i.e. only on a real, sustained
# probability shift (the object/tube actually changed), not on ordinary
# frame-to-frame noise. Once a pixel's identity is locked post-warmup,
# `pred` is overwritten with the locked class instead of the raw per-frame
# argmax for classes 2/3/4.
#
# [FIX-I24] Because the lock is per-pixel, it would normally stay pinned to
# a fixed screen location even if the physical tube moved. To keep the lock
# attached to the object instead of the background, the lock/EMA arrays are
# warped every frame with dense optical flow computed between consecutive
# frames (see SegmentationEngine._track_lock_with_flow), before this
# frame's segmentation evidence is merged in.
#
# The EMA/lock state is updated EVERY frame from frame 1 (including during
# warmup) so it has already converged by the time enforcement turns on —
# this avoids a visible jump right at the warmup->live transition.
IDENTITY_HYSTERESIS_ENABLED = True
IDENTITY_EMA_ALPHA          = 0.05   # slow EMA — resistant to single-frame noise
TUBE_IDENTITY_MARGIN        = 0.12   # EMA must cross this margin to flip identity
TUBE_PRESENT_THR            = 0.15   # min total tube prob to consider "tube present" here

# [FIX-I26] Controls WHEN the identity lock is allowed to start overwriting
# `pred` (as opposed to just learning in the background).
#   "post_warmup" (default, matches the original design intent) — the lock
#       only enforces once warmup_done is True. The EMA/lock state is still
#       being LEARNED every frame from frame 1, but nothing gets overwritten
#       until warmup has finished, so a noisy/occluded first few frames of a
#       cycle can't bake in a wrong lock before the model has stabilized.
#   "immediate" — the lock enforces starting from frame 1 of the cycle.
#       Only use this if you've confirmed WARMUP_FRAMES reliably clears any
#       initial occlusion/partial view before this matters.
LOCK_ENGAGE_MODE = "post_warmup"   # "post_warmup" | "immediate"

# [FIX-I26] Once a pixel's tube identity is locked, its confidence is no
# longer re-checked against the full per-class threshold (CLASS_CONF_THR) —
# that check is calibrated for raw per-frame argmax winners, and a locked
# pixel is often NOT the raw argmax winner in a given frame (that's the
# whole point of the lock). Re-applying the full threshold to locked pixels
# just fights the lock's own decision every frame and produces flicker.
# Instead, locked pixels get this much lower floor threshold — enough to
# still drop pixels that are genuinely background, not enough to undo a
# lock that the EMA has already confirmed with sustained evidence.
LOCKED_PIXEL_MIN_CONF = 0.08

# [FIX-I22b] Stricter confidence thresholds used ONLY before the identity
# lock has engaged (i.e. during warmup / re-acquisition after a cycle
# reset), so weak/noisy early activations don't get baked into the identity
# EMA before the model has stabilized on this cycle's footage.
WARMUP_CLASS_CONF_THR = {
    1: 0.35,
    2: 0.32,
    3: 0.30,
    4: 0.22,
}

USE_HSV_GATE  = False
HSV_GATES     = {
    2: (80,  105, 55,  False),
    3: (100, 135, 45,  False),
    4: (140, 180, 45,  True),
}
HSV_EMA_DECAY = 0.40

# ══════════════════════════════════════════════════════════════════════════════
#  YOLO CONFIG
# ══════════════════════════════════════════════════════════════════════════════
YOLO_SOCKET_CONF   = 0.35
YOLO_POSE_CONF     = 0.40
CLS_NO_SOCKET      = 0
CLS_SOCKET         = 1
ROI_PAD_X          = 180
ROI_PAD_Y          = 160
OF_MOTION_THR      = 3.5
OF_MOTION_FRAC     = 0.06
# [from v49] Larger grace window so brief socket-detection flicker inside a
# single physical cycle doesn't get counted as a cycle boundary.
SOCKET_RESET_GRACE = 45
_prev_gray_ref     = [None]
_ROI_COL_AMBER     = (0, 165, 255)

# [FIX-I28] Per-frame latency (ms) instrumentation. Any frame whose total
# processing time exceeds this threshold gets logged to stdout so slow
# frames (model stalls, disk I/O hiccups, etc.) are visible; the running
# average is also shown on the HUD next to FPS.
FRAME_MS_WARN_THRESHOLD = 150.0
FRAME_MS_EMA_ALPHA      = 0.12

# ══════════════════════════════════════════════════════════════════════════════
#  STATES
# ══════════════════════════════════════════════════════════════════════════════
STATE_IDLE    = "IDLE"
STATE_WARMUP  = "WARMUP"
STATE_HAND    = "HAND"
STATE_INSPECT = "INSPECT"
STATE_NORMAL  = "NORMAL"
STATE_ANOMALY = "ANOMALY"
STATE_PARTIAL = "PARTIAL"

_CHIP_LABEL = {
    STATE_IDLE:    "NO SOCKET",
    STATE_WARMUP:  "WARMING UP...",
    STATE_HAND:    "HAND IN ROI",
    STATE_INSPECT: "INSPECTING TUBES",
    STATE_NORMAL:  "NORMAL",
    STATE_ANOMALY: "ANOMALY DETECTED",
    STATE_PARTIAL: "PARTIAL VIEW",
}
_ORDER_DISPLAY = {"OK": "NORMAL", "ANOMALY": "ANOMALY",
                  "PARTIAL": "PARTIAL", "N/A": "N/A"}

_PROD_STATUS = {
    STATE_IDLE:    "WAITING FOR PART",
    STATE_WARMUP:  "STARTING INSPECTION",
    STATE_HAND:    "OPERATOR HANDLING PART",
    STATE_INSPECT: "INSPECTING",
    STATE_NORMAL:  "PASS",
    STATE_ANOMALY: "FAIL",
    STATE_PARTIAL: "PARTIAL VIEW",
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_roi(frame_shape, bbox):
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - ROI_PAD_X), max(0, y1 - ROI_PAD_Y),
            min(W - 1, x2 + ROI_PAD_X), min(H - 1, y2 + ROI_PAD_Y))


def is_cyclic_match(detected, expected):
    if len(detected) != len(expected):
        return False
    n = len(expected)
    for i in range(n):
        if detected == expected[i:] + expected[:i]:
            return True
    return False


def make_radial_channel_np(h, w, cx=None, cy=None):
    cx = cx if cx is not None else w / 2.0
    cy = cy if cy is not None else h / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
    return np.clip(
        dist / max(math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2), 1.0),
        0.0, 1.0
    )


def detect_in_channels_from_ckpt(ckpt_path: str) -> int:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt.get("model_state_dict", ckpt)
        for key, tensor in sd.items():
            if tensor.ndim == 4 and key.endswith(".weight"):
                ic = tensor.shape[1]
                if ic in (3, 4):
                    print(f"[AUTO] Detected in_channels={ic} from key: {key}")
                    return ic
        print("[AUTO] Could not detect in_channels — defaulting to 3")
        return 3
    except Exception as e:
        print(f"[AUTO] Channel detection failed ({e}) — defaulting to 3")
        return 3


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I40] GPU SANITY BENCHMARK  (merged from v51)
# ══════════════════════════════════════════════════════════════════════════════
def _gpu_sanity_benchmark(device: str):
    """
    Times a plain FP16 matmul on `device` and flags it if the measured
    throughput is far below what a modern GPU should manage. Low
    utilization alone can just mean the surrounding CPU pipeline is the
    bottleneck — but it can ALSO mean PyTorch doesn't have compiled CUDA
    kernels for this specific GPU's compute capability and is silently
    falling back to a slow generic path. This benchmark tells the two
    apart: if this number is healthy, the model itself is fast; if it's
    low, upgrading the torch build is the actual fix.
    """
    try:
        torch.cuda.synchronize()
        n = 4096
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        dt_per_iter = (time.perf_counter() - t0) / iters
        tflops = (2 * n ** 3) / dt_per_iter / 1e12
        print(f"  [FIX-I40] FP16 matmul sanity check : {n}x{n} in "
              f"{dt_per_iter * 1000:.2f} ms/iter  (~{tflops:.1f} TFLOP/s)")
        if tflops < 2.0:
            print("  [FIX-I40] [WARN] That throughput is well below what a modern")
            print("            GPU should manage in FP16. This can happen when the")
            print("            installed torch build has no compiled kernels for this")
            print("            GPU's compute capability (common right after a new GPU")
            print("            generation launches, e.g. Blackwell / RTX 50-series) and")
            print("            PyTorch silently falls back to a slow generic path. If")
            print("            so, the fix is upgrading torch to a build that matches")
            print("            this GPU/CUDA version, not further pipeline tuning.")
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FIX-I40] [WARN] GPU sanity benchmark failed (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I38/I39] GPU / DEVICE RESOLUTION + DIAGNOSTICS  (merged from v51)
# ══════════════════════════════════════════════════════════════════════════════
def resolve_device(require_gpu: bool = True) -> str:
    """
    Decide which device ("cuda:0" or "cpu") every model in this script will
    be loaded/run on, and print a full diagnostic block explaining WHY, so
    a silent fallback to CPU (the symptom that caused GPU 0 to sit at
    ~0-2% utilization while CPU/Memory did the work) is immediately visible
    in the console log instead of only showing up as "why is this so slow".

    If require_gpu is True (the default) and CUDA is not usable, this
    raises a RuntimeError instead of quietly continuing on CPU — the old
    silent-fallback behaviour is exactly what let this go unnoticed.

    [FIX-I39] Also enables TF32 matmul/cudnn kernels and cudnn.benchmark,
    and pre-builds GPU-resident ImageNet mean/std normalization tensors
    (_NORM_MEAN/_NORM_STD) so raw_infer() never has to build/move them
    per-frame. [FIX-I40] Also runs a GPU sanity benchmark to distinguish
    "GPU works but the surrounding pipeline is CPU-bound" from "torch has
    no compiled kernels for this GPU and is silently slow".
    """
    global _NORM_MEAN, _NORM_STD

    safe_print("\n" + "═" * 78)
    print("  [FIX-I38] GPU / DEVICE DIAGNOSTICS")
    safe_print("═" * 78)
    print(f"  torch version         : {torch.__version__}")
    print(f"  torch.version.cuda     : {torch.version.cuda}")
    print(f"  cudnn version          : {torch.backends.cudnn.version()}")
    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_ok}")

    if cuda_ok:
        try:
            idx  = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap  = torch.cuda.get_device_capability(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            print(f"  Selected GPU index      : {idx}")
            print(f"  Selected GPU name       : {name}")
            print(f"  Compute capability      : sm_{cap[0]}{cap[1]}")
            print(f"  GPU memory free/total   : {free_b/1e9:.2f} GB / {total_b/1e9:.2f} GB")
        except Exception as e:
            print(f"  [WARN] Could not read full CUDA device info: {e}")
            cuda_ok = False

    if not cuda_ok:
        print("  !! torch reports NO usable CUDA device. Common causes:")
        print("     - A CPU-only build of torch is installed (most common).")
        print("       Fix: pip uninstall torch torchvision torchaudio, then")
        print("       reinstall the CUDA build from https://pytorch.org/get-started/locally/")
        print("       matching your installed NVIDIA driver / CUDA version.")
        print("     - Your GPU (e.g. newer RTX 50-series/Blackwell cards) needs a")
        print("       torch build new enough to include its compute-capability")
        print("       (sm_) kernels — an older torch build will simply not see it.")
        print("     - NVIDIA driver too old for the installed CUDA toolkit.")
        safe_print("═" * 78 + "\n")
        if require_gpu:
            raise RuntimeError(
                "CUDA is not available to torch, so all three models "
                "(segmentation UNet++, YOLO socket, YOLO pose) would silently "
                "run on CPU. Refusing to continue with --require_gpu set "
                "(default). Fix the torch/CUDA install, or pass "
                "--no_require_gpu to explicitly allow CPU execution."
            )
        print("  [WARN] Continuing on CPU because --no_require_gpu was passed.")
        _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return "cpu"

    device = "cuda:0"
    torch.backends.cudnn.benchmark = True
    # [FIX-I39] TF32 matmul/conv kernels — free throughput on Ampere+ GPUs
    # for both the UNet++ segmentation model and both YOLO models. Safe
    # default: TF32 trades a small amount of mantissa precision for a
    # large speedup and is what NVIDIA recommends leaving on for inference
    # workloads like this one.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    print(f"  [FIX-I38] cudnn.benchmark      : True")
    print(f"  [FIX-I39] TF32 matmul/cudnn    : True")
    print(f"  RESOLVED DEVICE       : {device}")

    # [FIX-I39] Pre-build GPU-resident normalization tensors once, so
    # raw_infer() never has to build/move them per-frame.
    _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _NORM_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # [FIX-I40] Confirm the GPU is actually fast, not just "available".
    _gpu_sanity_benchmark(device)
    safe_print("═" * 78 + "\n")

    return device


def _log_model_device(label: str, device_str: str):
    print(f"[GPU-CHECK] {label:<22} -> device = {device_str}")


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I30] SOCKET-ADJACENT MASK ROI GATE
# ══════════════════════════════════════════════════════════════════════════════
def restrict_mask_to_socket_roi(pred_map, center, radius=None, classes=None,
                                 shape=None, radius_x=None, radius_y=None,
                                 offset_x=None, offset_y=None, polygon=None,
                                 radius_up=None, radius_down=None,
                                 radius_left=None, radius_right=None,
                                 bbox_size=None, auto_scale=None):
    """
    Suppress tube-class pixels that fall OUTSIDE a keep-region anchored to
    the socket. `center` is (cx, cy) in original-frame pixel coordinates —
    normally the socket bbox centre from the current frame's YOLO hit, or
    the cached last-known socket centre when YOLO missed this frame.

    [FIX-I31] `shape` controls the keep-region's geometry:
      "circle" — pixels farther than `radius` (Euclidean) from the
          (possibly offset) centre are dropped.
      "square" — axis-aligned square envelope of half-side `radius`.
      "ellipse" [FIX-I32] — independent horizontal/vertical radii
          (`radius_x`, `radius_y`), so the keep-region can be taller than
          it is wide (or vice versa) instead of uniform in every direction.
      "quad_ellipse" [FIX-I34, default] — like "ellipse" but with a
          SEPARATE radius for each of the 4 directions from the centre
          (up/down/left/right: `radius_up`, `radius_down`, `radius_left`,
          `radius_right`). This is the simplest way to match an irregular,
          non-symmetric keep-region (e.g. reaches much further up-right
          than down-left) without needing to trace a full custom polygon —
          just 4 numbers to tune.
      "polygon" [FIX-I33] — an exact custom boundary, given as a list of
          (dx, dy) offsets FROM THE SOCKET CENTRE in `polygon`. Available
          if quad_ellipse still isn't flexible enough; generate offsets
          with tools/pick_roi_polygon.py.

    [FIX-I35] `bbox_size` = (bbox_w, bbox_h) of the current socket YOLO
    detection (or last-known bbox if this frame's YOLO missed). When
    `auto_scale` (defaults to MASK_ROI_AUTO_SCALE) is True and
    shape == "quad_ellipse", the 4 directional radii are computed as
    MASK_ROI_MULT_{UP,DOWN} * bbox_h and MASK_ROI_MULT_{LEFT,RIGHT} *
    bbox_w instead of using the fixed radius_up/down/left/right constants —
    this makes the ROI size track the socket's own on-screen scale, so the
    same multipliers generalize across videos shot at different zoom
    levels/distances/resolutions without re-tuning pixel radii by hand.

    [FIX-I32] `offset_x` / `offset_y` shift the keep-region's centre (and,
    for "polygon", the whole traced shape) away from the raw socket centre
    before the shape test is applied. Use this when the tube segment you
    want to keep sits mostly on one side of the socket (e.g. mostly above
    it) rather than symmetrically around it. Negative offset_y moves the
    kept region UP; negative offset_x moves it LEFT (standard image
    pixel-row/column convention).

    Any pixel in `pred_map` whose class is in `classes` (default: the tube
    classes 2/3/4) and whose location is outside the keep-region is reset
    to background (0).

    If `center` is None (socket never seen this cycle), the map is
    returned unchanged — there is nothing to anchor the ROI to.
    """
    if center is None:
        return pred_map


    radius       = MASK_ROI_RADIUS if radius is None else radius
    classes      = MASK_ROI_CLASSES if classes is None else classes
    shape        = MASK_ROI_SHAPE if shape is None else shape
    radius_x     = MASK_ROI_RADIUS_X if radius_x is None else radius_x
    radius_y     = MASK_ROI_RADIUS_Y if radius_y is None else radius_y
    offset_x     = MASK_ROI_OFFSET_X if offset_x is None else offset_x
    offset_y     = MASK_ROI_OFFSET_Y if offset_y is None else offset_y
    polygon      = MASK_ROI_POLYGON if polygon is None else polygon
    radius_up    = MASK_ROI_RADIUS_UP if radius_up is None else radius_up
    radius_down  = MASK_ROI_RADIUS_DOWN if radius_down is None else radius_down
    radius_left  = MASK_ROI_RADIUS_LEFT if radius_left is None else radius_left
    radius_right = MASK_ROI_RADIUS_RIGHT if radius_right is None else radius_right
    auto_scale   = MASK_ROI_AUTO_SCALE if auto_scale is None else auto_scale

    # [FIX-I35] Derive the 4 directional radii from the socket's own bbox
    # size instead of the fixed constants, so the ROI scales automatically
    # with camera zoom/distance/resolution across different videos. Falls
    # back to the fixed radius_up/down/left/right values above if no bbox
    # size is available yet (e.g. socket never seen this cycle).
    if shape == "quad_ellipse" and auto_scale and bbox_size is not None:
        bbox_w, bbox_h = bbox_size
        if bbox_w > 0 and bbox_h > 0:
            radius_up    = MASK_ROI_MULT_UP    * bbox_h
            radius_down  = MASK_ROI_MULT_DOWN  * bbox_h
            radius_left  = MASK_ROI_MULT_LEFT  * bbox_w
            radius_right = MASK_ROI_MULT_RIGHT * bbox_w

    h, w = pred_map.shape[:2]
    cx, cy = center
    cx += offset_x
    cy += offset_y

    if shape == "quad_ellipse":
        # Bounding box wide/tall enough to cover the largest reach in any
        # direction; the per-pixel test below picks the correct
        # direction-specific radius for each quadrant.
        rx_max = max(radius_left, radius_right)
        ry_max = max(radius_up, radius_down)
        x1 = max(0, int(cx - rx_max)); x2 = min(w, int(cx + rx_max) + 1)
        y1 = max(0, int(cy - ry_max)); y2 = min(h, int(cy + ry_max) + 1)
        roi_mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx - cx
        dy = yy - cy
        rx = np.where(dx >= 0, radius_right, radius_left).astype(np.float32)
        ry = np.where(dy >= 0, radius_down, radius_up).astype(np.float32)
        norm2 = (dx / np.maximum(rx, 1e-6)) ** 2 + (dy / np.maximum(ry, 1e-6)) ** 2
        roi_mask[y1:y2, x1:x2] = norm2 <= 1.0
    elif shape == "polygon":
        if not polygon:
            # No polygon configured — nothing to gate on, leave unchanged
            # rather than silently dropping every tube pixel.
            return pred_map
        pts = np.array([[cx + dx, cy + dy] for dx, dy in polygon],
                        dtype=np.int32)
        roi_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_u8, [pts], 1)
        roi_mask = roi_mask_u8.astype(bool)
    elif shape == "ellipse":
        x1 = max(0, int(cx - radius_x)); x2 = min(w, int(cx + radius_x) + 1)
        y1 = max(0, int(cy - radius_y)); y2 = min(h, int(cy + radius_y) + 1)
        roi_mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        norm2  = ((xx - cx) / max(radius_x, 1e-6)) ** 2 + \
                 ((yy - cy) / max(radius_y, 1e-6)) ** 2
        roi_mask[y1:y2, x1:x2] = norm2 <= 1.0
    elif shape == "circle":
        x1 = max(0, int(cx - radius)); x2 = min(w, int(cx + radius) + 1)
        y1 = max(0, int(cy - radius)); y2 = min(h, int(cy + radius) + 1)
        roi_mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dist2  = (xx - cx) ** 2 + (yy - cy) ** 2
        roi_mask[y1:y2, x1:x2] = dist2 <= (radius ** 2)
    else:  # "square"
        x1 = max(0, int(cx - radius)); x2 = min(w, int(cx + radius))
        y1 = max(0, int(cy - radius)); y2 = min(h, int(cy + radius))
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y1:y2, x1:x2] = True

    for ci in classes:
        outside = (pred_map == ci) & (~roi_mask)
        if outside.any():
            pred_map[outside] = 0

    return pred_map


def apply_exclusion_zone(pred_map, center, classes=None, offset_x=None,
                          offset_y=None, radius_x=None, radius_y=None,
                          bbox_size=None, auto_scale=None, enabled=None):
    """
    [FIX-I36] Hard keep-out zone — the inverse of restrict_mask_to_socket_roi.
    Any tube-class pixel that falls INSIDE this ellipse is zeroed out,
    regardless of the main inclusion ROI. Use this for a known distractor
    object near the socket (a different tube/cable that could occasionally
    get misclassified as one of the tube classes due to similar colour or
    lighting) that should always be suppressed.

    The ellipse is centred at (center[0] + offset_x, center[1] + offset_y)
    with radii (radius_x, radius_y). When `auto_scale` (defaults to
    MASK_EXCLUDE_AUTO_SCALE) is True and `bbox_size` (bbox_w, bbox_h) is
    available, offset_x/offset_y/radius_x/radius_y are instead computed as
    MASK_EXCLUDE_*_MULT_X * bbox_w / MASK_EXCLUDE_*_MULT_Y * bbox_h, so the
    same relative position/size (e.g. "the other cable sits up-and-left of
    the socket, about this big relative to it") generalizes across videos
    shot at different zoom/distance — same reasoning as the main ROI's
    auto-scaling.

    If `center` is None or the zone is disabled, the map is returned
    unchanged.
    """
    enabled = MASK_EXCLUDE_ENABLED if enabled is None else enabled
    if not enabled or center is None:
        return pred_map

    classes    = MASK_EXCLUDE_CLASSES if classes is None else classes
    offset_x   = MASK_EXCLUDE_OFFSET_X if offset_x is None else offset_x
    offset_y   = MASK_EXCLUDE_OFFSET_Y if offset_y is None else offset_y
    radius_x   = MASK_EXCLUDE_RADIUS_X if radius_x is None else radius_x
    radius_y   = MASK_EXCLUDE_RADIUS_Y if radius_y is None else radius_y
    auto_scale = MASK_EXCLUDE_AUTO_SCALE if auto_scale is None else auto_scale

    if auto_scale and bbox_size is not None:
        bbox_w, bbox_h = bbox_size
        if bbox_w > 0 and bbox_h > 0:
            offset_x = MASK_EXCLUDE_OFFSET_MULT_X * bbox_w
            offset_y = MASK_EXCLUDE_OFFSET_MULT_Y * bbox_h
            radius_x = MASK_EXCLUDE_RADIUS_MULT_X * bbox_w
            radius_y = MASK_EXCLUDE_RADIUS_MULT_Y * bbox_h

    h, w = pred_map.shape[:2]
    cx = center[0] + offset_x
    cy = center[1] + offset_y

    x1 = max(0, int(cx - radius_x)); x2 = min(w, int(cx + radius_x) + 1)
    y1 = max(0, int(cy - radius_y)); y2 = min(h, int(cy + radius_y) + 1)
    if x2 <= x1 or y2 <= y1:
        return pred_map

    yy, xx = np.mgrid[y1:y2, x1:x2]
    norm2  = ((xx - cx) / max(radius_x, 1e-6)) ** 2 + \
             ((yy - cy) / max(radius_y, 1e-6)) ** 2
    exclude_mask = np.zeros((h, w), dtype=bool)
    exclude_mask[y1:y2, x1:x2] = norm2 <= 1.0

    for ci in classes:
        inside = (pred_map == ci) & exclude_mask
        if inside.any():
            pred_map[inside] = 0

    return pred_map


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_yolo(path, label, device=None):
    """
    [FIX-I38] Explicitly moves the loaded YOLO model onto `device` (via
    ultralytics' own `.to(device)`) right after loading.

    [FIX-I39] The warmup frame is now built at a realistic size
    (GPU_WARMUP_HW) instead of 64x64. torch/cudnn's autotuned-kernel cache
    (cudnn.benchmark) is keyed on input shape, so warming up at a tiny,
    never-seen-again shape didn't actually pre-warm anything useful — the
    first real video frame was still paying the full one-time autotune
    cost. Warmup also runs in FP16 (half=USE_HALF) when enabled, so the
    FP16 kernel variant is the one that gets autotuned/cached. Logs the
    actual device the model reports afterwards so it's immediately visible
    in the console whether this model really is on the GPU.
    """
    if not path:
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        if device is not None:
            m.to(device)
            # [FIX-I39] Realistic-size warmup so cudnn.benchmark autotunes
            # the actual shape used by real video frames, and so the FP16
            # kernel path (when USE_HALF) is exercised before the video
            # loop starts.
            try:
                wh, ww = GPU_WARMUP_HW
                dummy = np.zeros((wh, ww, 3), dtype=np.uint8)
                _q_kw = {"quantize": "half"} if USE_HALF else {}
                m.predict(dummy, device=device, verbose=False, **_q_kw)
            except Exception as warm_e:
                print(f"[WARN] YOLO {label} warmup inference failed: {warm_e}")
        print(f"[OK ] YOLO {label}: {path}  half={USE_HALF}")
        try:
            actual_device = next(m.model.parameters()).device
        except Exception:
            actual_device = "unknown"
        _log_model_device(f"YOLO-{label}", str(actual_device))
        return m
    except Exception as e:
        print(f"[WARN] YOLO {label} failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SOCKET DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
def detect_socket(model, frame, conf_thr):
    if model is None:
        return None
    # [FIX-I38] Explicitly pass device=DEVICE instead of relying on
    # ultralytics' own auto-detection, which could silently pick CPU.
    # [FIX-I39] quantize="half" runs this on the FP16 kernel path when
    # enabled (replaces deprecated half= parameter).
    _q_kw = {"quantize": "half"} if USE_HALF else {}
    res = model(frame, verbose=False, device=DEVICE, **_q_kw)[0]
    best, best_c = None, -1.0
    for box in res.boxes:
        c = float(box.conf[0])
        if c >= conf_thr and c > best_c:
            best_c = c
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            best = {"bbox": (x1, y1, x2, y2),
                    "class": int(box.cls[0]), "conf": c}
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I37] MASK-FREEZE MOTION CHECK
# ══════════════════════════════════════════════════════════════════════════════
def mask_has_moved(gray_now, gray_ref, thr=MASK_FREEZE_MOTION_THR,
                    frac=MASK_FREEZE_MOTION_FRAC):
    """
    Dense-optical-flow comparison between the current ROI crop and the
    reference crop taken when the mask was last frozen. Returns True if
    enough of the region shows real displacement to justify re-running
    inference; False if the tube looks statically unchanged, in which
    case the caller should keep reusing the frozen pred_map untouched.

    Comparing against the ORIGINAL frozen reference (rather than just the
    previous frame) means slow cumulative drift across many frames still
    triggers a refresh once total displacement crosses the threshold,
    instead of only reacting to a single large frame-to-frame jump.
    """
    if gray_now.shape != gray_ref.shape or gray_now.size == 0:
        # Shape mismatch (e.g. ROI size changed) — safest is to treat as
        # "moved" so a fresh inference re-establishes a valid frozen state.
        return True
    flow = cv2.calcOpticalFlowFarneback(
        gray_ref, gray_now, None, pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moved_frac = float((mag > thr).sum()) / mag.size
    return moved_frac >= frac


# ══════════════════════════════════════════════════════════════════════════════
#  HAND DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_hand_in_roi(pose_model, frame_bgr, roi, pose_conf_thr):
    rx1, ry1, rx2, ry2 = roi
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    motion   = False
    if _prev_gray_ref[0] is not None:
        rc = gray[ry1:ry2, rx1:rx2]
        rp = _prev_gray_ref[0][ry1:ry2, rx1:rx2]
        if rc.shape == rp.shape and rc.size > 0:
            flow = cv2.calcOpticalFlowFarneback(
                rc, rp, None, pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if float((mag > OF_MOTION_THR).sum()) / roi_area >= OF_MOTION_FRAC:
                motion = True
    _prev_gray_ref[0] = gray
    if motion:
        return True
    if pose_model is None:
        return False
    # [FIX-I38] Explicitly pass device=DEVICE instead of relying on
    # ultralytics' own auto-detection, which could silently pick CPU.
    # [FIX-I39] quantize="half" runs this on the FP16 kernel path when
    # enabled (replaces deprecated half= parameter).
    _q_kw = {"quantize": "half"} if USE_HALF else {}
    res = pose_model(frame_bgr, verbose=False, device=DEVICE, **_q_kw)[0]
    if hasattr(res, "keypoints") and res.keypoints is not None:
        for kpts in res.keypoints.data:
            if kpts.shape[0] < 11:
                continue
            for idx in (5, 6, 7, 8, 9, 10):
                kx, ky, kc = kpts[idx].tolist()
                if (kc >= pose_conf_thr and
                        rx1 <= int(kx) <= rx2 and ry1 <= int(ky) <= ry2):
                    return True
    for box in res.boxes:
        bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
        ix1 = max(bx1, rx1); iy1 = max(by1, ry1)
        ix2 = min(bx2, rx2); iy2 = min(by2, ry2)
        if ix2 > ix1 and iy2 > iy1 and (ix2 - ix1) * (iy2 - iy1) / roi_area >= 0.10:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  HSV GATE (off by default)
# ══════════════════════════════════════════════════════════════════════════════
def build_hsv_valid_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H   = hsv[:, :, 0].astype(np.int32)
    S   = hsv[:, :, 1].astype(np.int32)
    valid = {}
    for cls_id, (h_lo, h_hi, s_min, wrap) in HSV_GATES.items():
        hue_ok       = (H >= h_lo) | (H <= 12) if wrap else (H >= h_lo) & (H <= h_hi)
        valid[cls_id] = hue_ok & (S >= s_min)
    return valid


def apply_hsv_gate(pred, hsv_valid):
    result = pred.copy()
    for cls_id, vm in hsv_valid.items():
        result[(result == cls_id) & (~vm)] = 0
    return result


def apply_hsv_ema_decay(ema_probs, hsv_valid, decay=HSV_EMA_DECAY):
    if ema_probs is None:
        return ema_probs
    out = ema_probs.copy()
    for cls_id, vm in hsv_valid.items():
        out[cls_id][~vm] *= decay
    return out / out.sum(axis=0, keepdims=True).clip(min=1e-7)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
# [FIX-I39] A.Normalize() removed from this pipeline — normalization
# (uint8 -> [0,1] -> mean/std) now happens on the GPU inside raw_infer(),
# not on the CPU per-frame. This pipeline now only does the CPU-bound
# geometric work (resize + pad) that OpenCV/albumentations must do anyway,
# then hands off a raw uint8 CHW tensor for a single GPU upload.
_tf_pipeline = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE[0], interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(min_height=IMG_SIZE[0], min_width=IMG_SIZE[1],
                  border_mode=cv2.BORDER_REFLECT_101),
    ToTensorV2(),
])

USE_RADIAL_CHANNEL = False
IN_CHANNELS        = 3


def _dilate_class_protected(bm, kernel, protect_mask=None):
    """
    [FIX-I21] Dilate a single-class binary mask `bm` by `kernel`, but never
    let it grow into pixels covered by `protect_mask` (i.e. pixels already
    claimed by a different, already-assigned class). This gives a thin
    tube-class mask visible on-screen width without letting it eat into a
    neighbouring tube class's pixels.
    """
    dilated = cv2.dilate(bm, kernel)
    if protect_mask is not None:
        dilated[protect_mask == 1] = 0
    return dilated


def _close_class_protected(bm, kernel, protect_mask=None):
    """
    [FIX-I43] Morphological CLOSE (dilate then erode) on a single-class
    binary mask `bm`, using a kernel that is typically LARGER than the
    dilation kernel used in _dilate_class_protected. Purpose: bridge small
    gaps BETWEEN nearby same-class blobs so a tube that segmented as
    several separate pieces along its length reads as one continuous
    stroke, instead of several separate (even if individually fattened)
    blobs. Uses the same protect_mask convention as dilation — the CLOSE's
    intermediate dilation step is never allowed to grow into pixels
    already claimed by a DIFFERENT non-background class, so bridging one
    tube's gaps can never bleed into a neighbouring tube class.
    """
    dilated = cv2.dilate(bm, kernel)
    if protect_mask is not None:
        dilated[protect_mask == 1] = 0
    closed = cv2.erode(dilated, kernel)
    # Re-OR the original mask back in — erode can occasionally eat into
    # small isolated original blobs faster than the bridging dilate step
    # re-grew them; keeping every originally-present pixel guarantees the
    # CLOSE never shrinks the mask below what dilation already produced.
    closed = np.maximum(closed, bm)
    if protect_mask is not None:
        closed[protect_mask == 1] = 0
    return closed


@torch.no_grad()
def raw_infer(seg_model, frame_bgr, socket_centre=None):
    """
    Single forward pass. Appends a distance-from-socket radial channel when
    the model was trained with 4 input channels.

    [FIX-I5] socket_centre is cached by the caller so a YOLO miss on one
    frame does not silently fall back to image centre mid-sequence.

    [FIX-I39] Normalization moved to the GPU: _tf_pipeline only does the
    CPU-bound resize/pad and hands back a raw uint8 CHW tensor. That
    tensor is uploaded once (non_blocking=True) and the
    uint8 -> float/half -> [0,1] -> mean/std normalize steps all run as
    GPU tensor ops using the pre-built _NORM_MEAN/_NORM_STD tensors,
    instead of doing the same math per-pixel on the CPU every frame. When
    USE_HALF is enabled the whole forward pass (normalize + autocast
    matmul/conv) runs in FP16.
    """
    import time
    t_start = time.time()
    
    oh, ow = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t_tf = time.time()
    t_u8   = _tf_pipeline(image=rgb)["image"]  # uint8 CHW tensor, still on CPU
    t_tf_end = time.time()

    global _NORM_MEAN, _NORM_STD
    print(f"[DEBUG raw_infer] module DEVICE={DEVICE}, compute_dtype={torch.float16 if (USE_HALF and DEVICE.startswith('cuda')) else torch.float32}")
    if _NORM_MEAN is None:
        _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    if _NORM_STD is None:
        _NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    compute_dtype = torch.float16 if (USE_HALF and DEVICE.startswith("cuda")) else torch.float32

    # Single CPU->GPU upload of the raw uint8 tensor.
    t = t_u8.unsqueeze(0).to(DEVICE, non_blocking=True).to(compute_dtype)
    t = t.div_(255.0).sub_(_NORM_MEAN.to(compute_dtype)).div_(_NORM_STD.to(compute_dtype))

    if USE_RADIAL_CHANNEL:
        scale = IMG_SIZE[0] / max(oh, ow)
        new_h = int(oh * scale)
        new_w = int(ow * scale)
        pad_y = (IMG_SIZE[0] - new_h) // 2
        pad_x = (IMG_SIZE[1] - new_w) // 2
        scx   = socket_centre[0] * scale + pad_x if socket_centre else IMG_SIZE[1] / 2.0
        scy   = socket_centre[1] * scale + pad_y if socket_centre else IMG_SIZE[0] / 2.0
        rad   = make_radial_channel_np(IMG_SIZE[0], IMG_SIZE[1], cx=scx, cy=scy)
        rad_t = torch.from_numpy(rad).to(DEVICE, non_blocking=True).to(compute_dtype)
        t     = torch.cat([t, rad_t.unsqueeze(0).unsqueeze(0)], dim=1)

    with torch.amp.autocast("cuda", enabled=(DEVICE.startswith("cuda")),
                             dtype=torch.float16 if USE_HALF else torch.float32):
        print(f"[DEBUG raw_infer] tensor t device={t.device}, seg_model param device={next(seg_model.parameters()).device}")
        
        import time
        t0 = time.time()
        logits = seg_model(t)
        torch.cuda.synchronize()
        print(f"[DEBUG raw_infer] model forward took {(time.time()-t0)*1000:.1f} ms")

    t_fwd_end = time.time()

    probs  = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
    scale  = IMG_SIZE[0] / max(oh, ow)
    new_h  = int(oh * scale)
    new_w  = int(ow * scale)
    pad_y  = (IMG_SIZE[0] - new_h) // 2
    pad_x  = (IMG_SIZE[1] - new_w) // 2
    crop   = probs[:, pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    t_post = time.time()
    
    res = np.stack([
        cv2.resize(crop[c].astype(np.float32), (ow, oh),
                   interpolation=cv2.INTER_LINEAR)
        for c in range(NUM_CLASSES)
    ])
    t_end = time.time()
    
    print(f"[TIMING] raw_infer total: {(t_end-t_start)*1000:.1f}ms | cvtColor: {(t_tf-t_start)*1000:.1f}ms | Albu: {(t_tf_end-t_tf)*1000:.1f}ms | Fwd: {(t_fwd_end-t_tf_end)*1000:.1f}ms | Post: {(t_post-t_fwd_end)*1000:.1f}ms | Resize: {(t_end-t_post)*1000:.1f}ms")
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION ENGINE  (v45 — live inference, no solid-fill / nearest-
#  component pruning; those v49 additions are intentionally NOT carried
#  over — only the tube-order logic and cycle logic were requested from v49.
#  This revision generalizes the FIX-I22 identity lock to all three tube
#  classes [FIX-I23] and makes the lock track the object via optical flow
#  [FIX-I24].)
# ══════════════════════════════════════════════════════════════════════════════
class SegmentationEngine:
    """
    Runs live inference every frame. EMA is maintained continuously so
    temporal smoothing still applies, but the mask is never frozen — it
    follows the object as it moves.
    """

    # Order used everywhere the identity lock stacks its 3 tube classes.
    _TUBE_CLASSES = (2, 3, 4)

    def __init__(self, model):
        self.model         = model
        self.ema_probs     = None
        self._persist      = {c: 0 for c in range(1, NUM_CLASSES)}
        self._morph_k      = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SZ, MORPH_KERNEL_SZ))
        # [FIX-I21] dilation kernel used to give thin tube masks visible width
        self._dilate_k     = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_DILATE_SZ, MASK_DILATE_SZ))
        # [FIX-I43] second, larger kernel used to CLOSE (bridge) gaps
        # between nearby same-class blobs after dilation, so one tube
        # reads as a single continuous band.
        self._close_k       = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_CLOSE_SZ, MASK_CLOSE_SZ))
        self.last_raw_pred = None
        # [FIX-I8] frame counter purely for readable debug output
        self._dbg_frame_ct = 0
        # [FIX-I22/I23] per-pixel identity EMA/lock state for the tube
        # classes (2/3/4) — lazily allocated on first frame once H,W known.
        # identity_ema shape: (3, H, W) for classes (2, 3, 4) respectively.
        self.identity_ema  = None
        self.identity_lock = None
        # [FIX-I24] previous grayscale frame used to compute the optical
        # flow that the identity lock/EMA is warped along every frame.
        self._prev_gray_track = None

    def reset_dilate_kernel(self):
        """Call after changing MASK_DILATE_SZ / MASK_CLOSE_SZ at runtime/CLI
        to rebuild the kernels."""
        self._dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_DILATE_SZ, MASK_DILATE_SZ))
        self._close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_CLOSE_SZ, MASK_CLOSE_SZ))

    def reset(self):
        self.ema_probs         = None
        self.last_raw_pred     = None
        self._persist          = {c: 0 for c in range(1, NUM_CLASSES)}
        self.identity_ema      = None
        self.identity_lock     = None
        self._prev_gray_track  = None

    # ------------------------------------------------------------------
    # [FIX-I24] Optical-flow lock tracking
    # ------------------------------------------------------------------
    def _track_lock_with_flow(self, frame_bgr):
        """
        Warp the existing identity lock/EMA state forward with dense
        optical flow BEFORE this frame's segmentation evidence is merged
        in, so the lock physically follows the tube as it moves instead of
        staying pinned to fixed pixel coordinates. No-op on the very first
        frame of a cycle (nothing to warp yet) or once the lock has not
        been allocated.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Downsample for faster optical flow (dense flow on 720p is extremely slow on CPU)
        h, w = gray.shape
        scale = 320.0 / w
        small_w, small_h = 320, int(h * scale)
        gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        if self._prev_gray_track is None or self.identity_lock is None:
            self._prev_gray_track = gray_small
            return
        if gray_small.shape != self._prev_gray_track.shape:
            # Defensive: frame size changed mid-run — drop tracking state
            # rather than remap onto mismatched dimensions.
            self._prev_gray_track = gray_small
            return

        flow_small = cv2.calcOpticalFlowFarneback(
            self._prev_gray_track, gray_small, None, pyr_scale=0.5, levels=2,
            winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            
        # Upscale flow field and scale displacements back to original resolution
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] /= scale
        flow[..., 1] /= scale

        h, w = gray.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)

        self.identity_lock = cv2.remap(
            self.identity_lock, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Optimize by remapping as a single 3-channel image (moves loop to C++)
        ema_hwc = self.identity_ema.transpose(1, 2, 0)
        ema_mapped = cv2.remap(
            ema_hwc, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # Handle cases where cv2.remap returns 2D array if channels=1 (though it shouldn't for 3)
        if ema_mapped.ndim == 2:
            ema_mapped = ema_mapped[..., np.newaxis]
        self.identity_ema = ema_mapped.transpose(2, 0, 1).copy()

        self._prev_gray_track = gray_small

    # ------------------------------------------------------------------
    # [FIX-I22/I23] 3-way identity hysteresis (classes 2, 3, 4)
    # ------------------------------------------------------------------
    def _apply_identity_hysteresis(self, probs, pred, enforce=True):
        """
        Lock each pixel's tube identity (class 2, 3, or 4) once established,
        so a continuous physical tube can't flicker to a different tube
        class near the connector junction.

        - self.identity_ema[k] tracks a slow EMA of prob[cls] for
          cls = self._TUBE_CLASSES[k], at every pixel where a tube is
          present (combined tube mass >= TUBE_PRESENT_THR).
        - self.identity_lock holds the CURRENT locked class (2, 3, 4, or
          0 = unset) for that pixel.
        - A pixel's lock is only allowed to flip to a different class once
          that class's EMA exceeds the currently-locked class's EMA by more
          than TUBE_IDENTITY_MARGIN — i.e. only on a real, sustained shift,
          not per-frame noise.
        - The EMA/lock is updated EVERY frame (even during warmup) so it
          has already converged by the time `enforce` (== warmup_done, or
          always-True under LOCK_ENGAGE_MODE="immediate") turns on; only
          the actual overwrite of `pred` is gated by `enforce`.

        Returns (pred, override_mask) — override_mask marks exactly the
        pixels this call overwrote, so the caller can apply a relaxed
        confidence floor to them instead of the full per-class threshold
        (see LOCKED_PIXEL_MIN_CONF / FIX-I26).
        """
        h, w = probs.shape[1], probs.shape[2]
        if self.identity_ema is None:
            self.identity_ema  = np.zeros((3, h, w), dtype=np.float32)
            self.identity_lock = np.zeros((h, w), dtype=np.int8)

        # class_probs[0]=cls2, [1]=cls3, [2]=cls4
        class_probs = np.stack([probs[c] for c in self._TUBE_CLASSES], axis=0)
        tube_mass   = class_probs.sum(axis=0)
        present     = tube_mass >= TUBE_PRESENT_THR

        # Optimize EMA update using np.where to avoid fancy-indexing overhead on huge arrays
        new_ema = IDENTITY_EMA_ALPHA * class_probs + (1.0 - IDENTITY_EMA_ALPHA) * self.identity_ema
        self.identity_ema = np.where(present[None, :, :], new_ema, self.identity_ema)

        # tube no longer present here at all -> forget the lock
        self.identity_lock[~present] = 0

        # index (0/1/2) of the class currently winning the EMA, and its value
        best_idx = np.argmax(self.identity_ema, axis=0)
        best_val = np.max(self.identity_ema, axis=0)  # np.max is vastly faster than take_along_axis

        # Fast advanced indexing to get cur_val using np.choose
        # We only care about cur_val where self.identity_lock != 0
        lock_idx_for_cur = np.clip(self.identity_lock.astype(np.int32) - 2, 0, 2)
        cur_val = np.choose(lock_idx_for_cur, self.identity_ema)
        
        lock_idx = np.clip(self.identity_lock.astype(np.int32) - 2, 0, 2)

        # first time this pixel has a tube -> seed the lock from current winner
        unset = present & (self.identity_lock == 0)
        self.identity_lock[unset] = (best_idx[unset] + 2).astype(np.int8)

        # only flip an existing lock on a sustained, margin-crossing shift
        locked = present & (self.identity_lock != 0) & (~unset)
        flip   = locked & (best_idx != lock_idx) & (best_val > cur_val + TUBE_IDENTITY_MARGIN)
        self.identity_lock[flip] = (best_idx[flip] + 2).astype(np.int8)

        override = present & np.isin(pred, self._TUBE_CLASSES) & (self.identity_lock != 0)
        if enforce:
            pred[override] = self.identity_lock[override]
        else:
            # Not enforcing yet (e.g. still in warmup under
            # LOCK_ENGAGE_MODE="post_warmup") — nothing was actually
            # overwritten, so nothing should be excluded from the normal
            # confidence-threshold pass downstream.
            override = np.zeros_like(override)

        if MASK_DEBUG:
            n_flip = int(flip.sum())
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=identity_lock   "
                  f"present_px={int(present.sum())}  flips_this_frame={n_flip}  "
                  f"enforce={enforce}  overridden_px={int(override.sum())}")

        # [FIX-I26] Return the override mask alongside pred so the caller
        # can relax (not skip) the confidence check specifically for
        # pixels the lock just assigned — see LOCKED_PIXEL_MIN_CONF.
        return pred, override

    def infer(self, frame_bgr, socket_centre=None, apply_identity_lock=True):
        """
        Full inference pipeline — called every non-hand frame.

        [FIX-I1] Lower EMA alpha keeps temporal blur minimal.
        [FIX-I2] Temperature rescaling sharpens the Mid/End boundary.
        [FIX-I3] Tighter conf thresholds force ambiguous pixels to background.
        [FIX-I8] Optional per-stage pixel-count debug trace for classes 2/3/4,
                 and morphological OPEN is skipped for classes 3/4 (thin tube
                 classes) so short/thin real regions aren't erased outright.
        [FIX-I22/I23] Once `apply_identity_lock` is True (post-warmup), each
                 tube pixel's class is locked to whichever of 2/3/4 it was
                 first established as, and only flips on a sustained
                 probability-margin crossing — this stops a single physical
                 tube from flickering to a different tube class right at
                 the socket.
        [FIX-I24] Before merging this frame's evidence, the lock/EMA state
                 is warped forward with optical flow so it tracks the
                 physical tube instead of staying at a fixed pixel location.
        [FIX-I42/I43] After the [FIX-I21] dilation pass, a second, larger
                 morphological CLOSE bridges gaps between nearby same-class
                 blobs so the tube renders as one continuous thick band
                 (see _close_class_protected) instead of several separate
                 fattened blobs.

        Note: the socket-adjacent ROI gate [FIX-I30] is deliberately NOT
        applied here — this method has no notion of the current frame's
        detected socket bbox center (only the possibly-stale cached
        `socket_centre`). It is applied by the caller in
        process_video_cycles(), right after this method returns, using the
        freshest socket-centre information available for that frame.
        """
        self._dbg_frame_ct += 1

        # [FIX-I24] Move the existing lock state to where the tube has
        # moved to, BEFORE folding in this frame's raw evidence.
        self._track_lock_with_flow(frame_bgr)

        probs_raw = raw_infer(self.model, frame_bgr, socket_centre=socket_centre)

        if MASK_DEBUG:
            raw_argmax = probs_raw.argmax(axis=0)
            counts_raw = {ci: int((raw_argmax == ci).sum()) for ci in (2, 3, 4)}
            maxp_raw   = {ci: float(probs_raw[ci].max()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=raw_argmax      "
                  f"px={counts_raw}  maxprob={ {k: round(v,3) for k,v in maxp_raw.items()} }")

        # [FIX-I1] Temporal EMA — low alpha means current frame dominates
        # Use inplace math to avoid allocating massive temporary arrays
        if self.ema_probs is None:
            self.ema_probs = probs_raw.copy()
        else:
            np.multiply(self.ema_probs, (1.0 - EMA_ALPHA), out=self.ema_probs)
            np.add(self.ema_probs, probs_raw * EMA_ALPHA, out=self.ema_probs)

        probs = self.ema_probs.copy()

        if MASK_DEBUG:
            ema_argmax = probs.argmax(axis=0)
            counts_ema = {ci: int((ema_argmax == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_ema       px={counts_ema}")

        # [FIX-I2] Sharpen Mid/End boundary via temperature rescaling.
        # Total (3+4) probability mass is conserved; only the 3-vs-4 ratio changes.
        if TUBE_BOUNDARY_SHARPENING:
            # Optimize memory allocations with inplace ops
            tube_stack = np.stack([probs[3], probs[4]], axis=0)
            t_max = tube_stack.max(axis=0, keepdims=True)
            np.subtract(tube_stack, t_max, out=tube_stack)
            np.divide(tube_stack, TUBE_SHARPNESS_TEMP, out=tube_stack)
            np.exp(tube_stack, out=tube_stack)
            t_sum = tube_stack.sum(axis=0, keepdims=True)
            t_sum += 1e-7
            np.divide(tube_stack, t_sum, out=tube_stack)
            
            tube_mass = probs[3] + probs[4]
            probs[3] = tube_stack[0] * tube_mass
            probs[4] = tube_stack[1] * tube_mass

        if MASK_DEBUG and TUBE_BOUNDARY_SHARPENING:
            sharp_argmax = probs.argmax(axis=0)
            counts_sharp = {ci: int((sharp_argmax == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_sharpen   "
                  f"px={counts_sharp}  temp={TUBE_SHARPNESS_TEMP}")

        if USE_HSV_GATE:
            hsv_valid = build_hsv_valid_mask(frame_bgr)
            probs     = apply_hsv_ema_decay(probs, hsv_valid)

        pred = probs.argmax(axis=0).astype(np.uint8)
        self.last_raw_pred = pred.copy().astype(np.int32)

        # [FIX-I22/I23] Always learn the identity EMA/lock (even during
        # warmup); only overwrite `pred` with the locked class once
        # apply_identity_lock (i.e. warmup_done, under the default
        # LOCK_ENGAGE_MODE="post_warmup") is True.
        lock_override_mask = None
        if IDENTITY_HYSTERESIS_ENABLED:
            pred, lock_override_mask = self._apply_identity_hysteresis(
                probs, pred, enforce=apply_identity_lock)

        # [FIX-I22b] Use stricter thresholds pre-lock so weak/noisy early
        # activations don't get baked into the identity EMA before the
        # model has stabilized on this cycle's footage.
        active_conf_thr = CLASS_CONF_THR if apply_identity_lock else WARMUP_CLASS_CONF_THR

        # [FIX-I3/FIX-I8/FIX-I26] Force ambiguous boundary pixels to
        # background. Pixels the identity lock just overrode are checked
        # against LOCKED_PIXEL_MIN_CONF instead of the full per-class
        # threshold — re-applying the full threshold to a class the lock
        # deliberately assigned (which is often NOT the raw argmax winner
        # there) would erase it again immediately and produce flicker.
        for ci, thr in active_conf_thr.items():
            mask = pred == ci
            if not mask.any():
                continue
            if lock_override_mask is not None:
                locked_here = mask & lock_override_mask
                normal_here = mask & (~lock_override_mask)
            else:
                locked_here = np.zeros_like(mask)
                normal_here = mask

            if normal_here.any():
                low = probs[ci][normal_here] < thr
                if low.any():
                    yx = np.where(normal_here)
                    pred[yx[0][low], yx[1][low]] = 0

            if locked_here.any():
                low_locked = probs[ci][locked_here] < LOCKED_PIXEL_MIN_CONF
                if low_locked.any():
                    yx = np.where(locked_here)
                    pred[yx[0][low_locked], yx[1][low_locked]] = 0

        if MASK_DEBUG:
            counts_conf = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_confthr   "
                  f"px={counts_conf}  thr={ {k: active_conf_thr[k] for k in (2,3,4)} }  "
                  f"locked={apply_identity_lock}")

        if USE_HSV_GATE:
            pred = apply_hsv_gate(pred, build_hsv_valid_mask(frame_bgr))

        # [FIX-I21] Give the thin tube-class masks (2/3/4) visible width by
        # dilating them. Each class is dilated independently and is blocked
        # from growing into pixels already claimed by a DIFFERENT class, so
        # widening a mask never lets it bleed into a neighbouring tube.
        # [FIX-I41/I42] Kernel size (MASK_DILATE_SZ) raised to 28 by default
        # so the on-screen tube overlays render as thick, filled bands.
        if MASK_DILATE_ENABLED:
            for ci in MASK_DILATE_CLASSES:
                bm = (pred == ci).astype(np.uint8)
                if not bm.any():
                    continue
                other_classes_mask = ((pred != 0) & (pred != ci)).astype(np.uint8)
                dilated = _dilate_class_protected(bm, self._dilate_k,
                                                   protect_mask=other_classes_mask)
                pred[pred == ci]   = 0
                pred[dilated == 1] = ci

            if MASK_DEBUG:
                counts_dilate = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
                print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_dilate    "
                      f"px={counts_dilate}  dilate_sz={MASK_DILATE_SZ} "
                      f"classes={MASK_DILATE_CLASSES}")

            # [FIX-I43] Second-stage CLOSE (bigger kernel) to bridge gaps
            # between nearby same-class blobs, right after dilation and
            # BEFORE the morphology/CC-filter passes below — closing first
            # means the small-component filter afterwards sees the already
            # -bridged, larger connected shape rather than several small
            # pieces that might individually fall under MIN_AREA_PX.
            if MASK_CLOSE_ENABLED and MASK_CLOSE_SZ > 0:
                for ci in MASK_DILATE_CLASSES:
                    bm = (pred == ci).astype(np.uint8)
                    if not bm.any():
                        continue
                    other_classes_mask = ((pred != 0) & (pred != ci)).astype(np.uint8)
                    closed = _close_class_protected(bm, self._close_k,
                                                     protect_mask=other_classes_mask)
                    pred[pred == ci]  = 0
                    pred[closed == 1] = ci

                if MASK_DEBUG:
                    counts_close = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
                    print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_close     "
                          f"px={counts_close}  close_sz={MASK_CLOSE_SZ} "
                          f"classes={MASK_DILATE_CLASSES}")

        # [FIX-I8] Morphological cleanup — CLOSE always applied; OPEN is
        # skipped for classes 3 & 4 (thin tube classes) because OPEN erases
        # thin structures outright, which was the main cause of the mid-tube
        # (class 3 / "Blue") mask disappearing at the connector junction.
        # Classes 1/2 keep the original CLOSE+OPEN behaviour.
        for ci in range(1, NUM_CLASSES):
            bm = (pred == ci).astype(np.uint8)
            bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, self._morph_k)
            if ci not in (3, 4):
                bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, self._morph_k)
            pred[pred == ci] = 0
            pred[bm == 1]    = ci

        if MASK_DEBUG:
            counts_morph = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_morph     px={counts_morph}")

        # Remove small connected components
        for ci in range(1, NUM_CLASSES):
            bm = (pred == ci).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bm)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PX:
                    pred[labels == i] = 0

        if MASK_DEBUG:
            counts_cc = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_ccfilter  "
                  f"px={counts_cc}  min_area={MIN_AREA_PX}")

        # Persistence filter
        for ci in range(1, NUM_CLASSES):
            self._persist[ci] = self._persist[ci] + 1 if (pred == ci).any() else 0
            if self._persist[ci] < PERSIST_FRAMES:
                pred[pred == ci] = 0

        if MASK_DEBUG:
            counts_final = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=final           px={counts_final}\n")

        return pred.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
#  VOTE COUNTER  (v49 — adds reset() so each cycle starts at 0/0)
# ══════════════════════════════════════════════════════════════════════════════
class VoteCounter:
    def __init__(self, threshold=VERDICT_THR):
        self.threshold     = threshold
        self.anomaly_votes = 0
        self.normal_votes  = 0

    def reset(self):
        self.anomaly_votes = 0
        self.normal_votes  = 0

    def record(self, gate_result):
        if gate_result == "ANOMALY":
            self.anomaly_votes += 1
        elif gate_result == "OK":
            self.normal_votes  += 1

    @property
    def total(self):
        return self.anomaly_votes + self.normal_votes

    def final_verdict(self):
        if self.total == 0:
            return "UNKNOWN"
        return "ANOMALY" if (self.anomaly_votes / self.total) > self.threshold else "NORMAL"

    def summary(self):
        r = self.anomaly_votes / max(self.total, 1)
        return (f"normal={self.normal_votes}  anomaly={self.anomaly_votes}  "
                f"total={self.total}  ratio={r:.3f}  verdict={self.final_verdict()}")


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I6] SEQUENCE STABILITY GATE
# ══════════════════════════════════════════════════════════════════════════════
class SequenceStabilityGate:
    """
    Suppresses the angular verdict until the detected sequence has been
    identical for MIN_SEQ_STABLE consecutive frames. A single bleed frame
    that flips to ANOMALY is treated as PARTIAL and does not reach
    AnomalyConfirmGate.
    """

    def __init__(self):
        self._prev_seq  = None
        self._stable_ct = 0

    def reset(self):
        self._prev_seq  = None
        self._stable_ct = 0

    def update(self, raw_order: str, detected_seq: list) -> str:
        seq_key = tuple(detected_seq)
        if seq_key == self._prev_seq:
            self._stable_ct += 1
        else:
            self._stable_ct = 1
            self._prev_seq  = seq_key
        if self._stable_ct >= MIN_SEQ_STABLE:
            return raw_order
        return "PARTIAL"


# ══════════════════════════════════════════════════════════════════════════════
#  TUBE ORDER ANALYSIS  [FIX-I18, from v49 — NEAREST-PIXEL ANGULAR GATE]
#  [FIX-I27] rectangular search ROI replaces the old circular region.
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_tube_order(pred_map, socket_bbox=None, debug=False):
    """
    Determine cyclic angular order of tube classes around the socket centre.

    [FIX-I18] Nearest-Pixel Angular Gate:
    For each tube class, search a region centred on the socket, and take the
    SINGLE pixel of that class closest to the socket centre (i.e. the point
    where the tube first meets/enters the socket housing). The bearing of
    that nearest pixel from the socket centre is the tube's angle. Order is
    then just the cyclic sort of those three bearings.

    [FIX-I27] The search region is a RECTANGLE (axis-aligned square of side
    2*NEAREST_SEARCH_RADIUS, centred on the socket) instead of a circle —
    this replaces the earlier circular search mask.
    """
    if socket_bbox is not None:
        x1, y1, x2, y2 = socket_bbox
        scx, scy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    else:
        ys, xs = np.where(pred_map == 1)
        scx    = float(xs.mean()) if len(xs) else pred_map.shape[1] / 2.0
        scy    = float(ys.mean()) if len(ys) else pred_map.shape[0] / 2.0

    H, W = pred_map.shape[:2]
    R    = NEAREST_SEARCH_RADIUS

    # ── RECTANGULAR SEARCH REGION AROUND THE SOCKET CENTRE [FIX-I27] ─────────
    rx1 = max(0, int(scx - R))
    ry1 = max(0, int(scy - R))
    rx2 = min(W - 1, int(scx + R))
    ry2 = min(H - 1, int(scy + R))

    rect_mask = np.zeros((H, W), dtype=np.uint8)
    rect_mask[ry1:ry2 + 1, rx1:rx2 + 1] = 1
    pred_roi = pred_map * rect_mask

    status    = {2: "Absent", 3: "Absent", 4: "Absent"}
    angles    = {}
    anchors   = {}
    nearest_d = {}

    if MASK_DEBUG:
        tube_px_counts = {ci: int((pred_roi == ci).sum()) for ci in (2, 3, 4)}
        print(f"[MASKDBG] evaluate_tube_order (nearest-pixel, rect ROI) px counts={tube_px_counts}  "
              f"MIN_TUBE_PX={MIN_TUBE_PX}  half_side={R}")

    for ci in (2, 3, 4):
        ty, tx = np.where(pred_roi == ci)
        if len(tx) < MIN_TUBE_PX:
            if MASK_DEBUG and len(tx) > 0:
                print(f"[MASKDBG] class {ci} has {len(tx)}px in search rect but "
                      f"< MIN_TUBE_PX={MIN_TUBE_PX} -> reported Absent")
            continue

        # [FIX-I18] find the single pixel of this class nearest the socket
        # centre — that is where the tube enters/meets the socket.
        dists     = np.sqrt((tx - scx) ** 2 + (ty - scy) ** 2)
        nearest_i = int(np.argmin(dists))
        nx, ny    = float(tx[nearest_i]), float(ty[nearest_i])

        anchors[ci]   = (nx, ny)
        angles[ci]    = math.atan2(ny - scy, nx - scx)
        nearest_d[ci] = float(dists[nearest_i])
        status[ci]    = "Present"

    visible = [c for c in (2, 3, 4) if c in angles]
    dbg     = {}

    if debug:
        dbg = {"scx": scx, "scy": scy, "rx1": rx1, "ry1": ry1,
               "rx2": rx2, "ry2": ry2, "radius": R, "anchors": anchors,
               "nearest_dist": nearest_d,
               "angles_deg": {ci: math.degrees(a) for ci, a in angles.items()},
               "gaps_deg": [], "max_gap_idx": -1, "start_ci": -1,
               "seq": [], "result": "PARTIAL"}

    if len(visible) < len(EXPECTED_SEQ):
        if debug:
            dbg["result"] = "PARTIAL"
        return status, "PARTIAL", visible, dbg

    sorted_by_angle = sorted(visible, key=lambda c: angles[c])
    TWO_PI = 2.0 * math.pi
    n      = len(sorted_by_angle)
    raw_a  = [angles[c] for c in sorted_by_angle]
    gaps   = [(raw_a[(i + 1) % n] - raw_a[i]) % TWO_PI for i in range(n)]
    start  = (int(np.argmax(gaps)) + 1) % n
    seq    = sorted_by_angle[start:] + sorted_by_angle[:start]
    order_result = "OK" if is_cyclic_match(seq, EXPECTED_SEQ) else "ANOMALY"

    if debug:
        dbg["gaps_deg"]    = [(sorted_by_angle[i], sorted_by_angle[(i + 1) % n],
                               math.degrees(gaps[i])) for i in range(n)]
        dbg["max_gap_idx"] = int(np.argmax(gaps))
        dbg["start_ci"]    = sorted_by_angle[start]
        dbg["seq"]         = seq
        dbg["result"]      = order_result

    return status, order_result, seq, dbg


# ══════════════════════════════════════════════════════════════════════════════
#  GATES
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyConfirmGate:
    def __init__(self, n=N_ANOMALY_CONFIRM):
        self.n        = n
        self._count   = 0
        self._latched = False

    def reset(self):
        self._count   = 0
        self._latched = False

    def update(self, order_result):
        if order_result == "ANOMALY":
            self._count += 1
            if self._count >= self.n:
                self._latched = True
        else:
            self._count   = 0
            self._latched = False
        if self._latched:
            return "ANOMALY"
        if order_result == "PARTIAL":
            return "PARTIAL"
        return "OK"


class ResultLatchGate:
    def __init__(self, n=LATCH_FRAMES):
        self.n       = n
        self._last   = None
        self._count  = 0
        self.locked  = False
        self.verdict = None

    def reset(self):
        self._last   = None
        self._count  = 0
        self.locked  = False
        self.verdict = None

    def update(self, gate_result):
        if self.locked:
            return True
        if gate_result in ("OK", "ANOMALY"):
            self._count = self._count + 1 if gate_result == self._last else 1
            self._last  = gate_result
            if self._count >= self.n:
                self.locked  = True
                self.verdict = gate_result
                return True
        else:
            self._last  = None
            self._count = 0
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CYCLE MANAGER  (from v49 — cycle status logic)
# ══════════════════════════════════════════════════════════════════════════════
class CycleManager:
    """
    Owns per-cycle output-video writing and PASS/FAIL/UNKNOWN bookkeeping
    for a single input video that may contain many socket-attach /
    socket-remove inspection cycles.
    """

    def __init__(self, resolved_output_dir, video_stem, fps_src, frame_size):
        self.resolved_output_dir = resolved_output_dir
        self.video_stem          = video_stem
        self.fps_src             = fps_src
        self.frame_size          = frame_size

        self.cycle_no   = 0
        self.active     = False
        self.writer     = None
        self.temp_path  = None
        self.start_time = None

        self.passed  = 0  # NORMAL
        self.failed  = 0  # ANOMALY
        self.unknown = 0  # UNKNOWN

        self.cycle_summaries = []

    @property
    def total_cycles(self):
        return self.passed + self.failed + self.unknown

    def start_cycle(self):
        self.cycle_no  += 1
        self.active     = True
        self.start_time = time.time()
        Path(self.resolved_output_dir).mkdir(parents=True, exist_ok=True)
        self.temp_path = os.path.join(
            self.resolved_output_dir,
            f"__processing__{self.video_stem}_cycle{self.cycle_no:03d}.mp4")
        self.writer = cv2.VideoWriter(
            self.temp_path, cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps_src, self.frame_size)
        print(f"\n[CYCLE START] #{self.cycle_no:03d}  ({self.video_stem})")

    def write(self, frame):
        if self.active and self.writer is not None:
            self.writer.write(frame)

    def hold_final_frame(self, frame, seconds=VERDICT_HOLD_SEC):
        if not self.active or self.writer is None:
            return
        hold = max(1, int(self.fps_src * seconds))
        for _ in range(hold):
            self.writer.write(frame)

    def end_cycle(self, verdict, extra_metrics=None):
        if not self.active:
            return None

        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.active = False

        dest_dir, folder_name = get_verdict_dir(self.resolved_output_dir, verdict)
        final_name = f"{self.video_stem}_cycle{self.cycle_no:03d}.mp4"
        final_path = os.path.join(dest_dir, final_name)
        if Path(final_path).exists():
            Path(final_path).unlink()
            
        # Retry loop for WinError 32 (file in use by VideoWriter async release)
        import time
        max_retries = 10
        for attempt in range(max_retries):
            try:
                shutil.move(self.temp_path, final_path)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    print(f"[WARN] Could not move {self.temp_path} after {max_retries} retries.")
                time.sleep(0.1)
            except OSError:
                break

        if verdict == "NORMAL":
            self.passed += 1
        elif verdict == "ANOMALY":
            self.failed += 1
        else:
            self.unknown += 1

        duration_s = time.time() - (self.start_time or time.time())
        safe_print(f"[CYCLE END]    #{self.cycle_no:03d}  →  {verdict:<8}  "
              f"({duration_s:.1f}s)  →  {final_path}")
        print(f"[RUNNING TOTAL] PASSED={self.passed}  FAILED={self.failed}  "
              f"UNKNOWN={self.unknown}  (of {self.total_cycles} cycles)")

        summary = {
            "cycle_no":      self.cycle_no,
            "final_verdict": verdict,
            "output_folder": folder_name,
            "output_path":   final_path,
        }
        if extra_metrics:
            summary.update(extra_metrics)
        self.cycle_summaries.append(summary)
        return summary

    def final_report(self):
        W = 62
        safe_print("\n" + "█" * W)
        print("  FINAL CYCLE REPORT")
        safe_print("█" * W)
        print(f"  TOTAL CYCLES      : {self.total_cycles}")
        print(f"  PASSED (NORMAL)   : {self.passed}")
        print(f"  FAILED (ANOMALY)  : {self.failed}")
        print(f"  UNKNOWN           : {self.unknown}")
        safe_print("█" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def draw_seg_overlay(frame, pred_map, alpha=None):
    out = frame.copy()
    a   = alpha if alpha is not None else OVERLAY_ALPHA
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show:
            continue
        mask = pred_map == ci
        if not mask.any():
            continue
        layer        = np.zeros_like(frame)
        layer[mask]  = bgr
        out = cv2.addWeighted(out, 1.0, layer, a, 0)
    return out


def draw_raw_argmax_fallback(frame, raw_pred):
    out = frame.copy()
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show:
            continue
        mask = raw_pred == ci
        if not mask.any():
            continue
        layer       = np.zeros_like(frame)
        layer[mask] = tuple(int(v * 0.40) for v in bgr)
        out = cv2.addWeighted(out, 1.0, layer, 0.50, 0)
    H, W = out.shape[:2]
    cv2.putText(out, "RAW ARGMAX (pre-filter)", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 200), 1, cv2.LINE_AA)
    return out


def draw_socket_box(frame, hit):
    if hit is None:
        return frame
    x1, y1, x2, y2 = hit["bbox"]
    is_p  = hit["class"] == CLS_SOCKET
    col   = (0, 220, 100) if is_p else (50, 50, 230)
    label = f"{'Socket' if is_p else 'No Socket'}  {hit['conf']*100:.0f}%"
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    fs = 0.60
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    by = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, by - th - 6), (x1 + tw + 10, by + bl), col, -1)
    cv2.putText(frame, label, (x1 + 5, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_final_verdict_overlay(frame, verdict, cycle_no=None, stats=None):
    out     = frame.copy()
    H, W    = out.shape[:2]
    is_anom = verdict == "ANOMALY"
    dim     = np.zeros_like(out)
    dim[:]  = (0, 0, 100) if is_anom else (0, 70, 10)
    out     = cv2.addWeighted(out, 0.40, dim, 0.60, 0)
    tc      = (60, 60, 255) if is_anom else (60, 230, 60)
    label   = f"CYCLE #{cycle_no:03d}  FINAL: {verdict}" if cycle_no else f"FINAL: {verdict}"
    sub     = "Check cable tube order!" if is_anom else "Cable correctly inserted."
    fs_big  = max(1.6, min(W, H) / 240.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs_big, 3)
    tx, ty  = (W - tw) // 2, H // 3
    cv2.putText(out, label, (tx + 3, ty + 3),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, tc, 3, cv2.LINE_AA)
    fs_sub = max(0.8, fs_big * 0.42)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 2)
    sy = ty + th + max(20, int(H * 0.04))
    cv2.putText(out, sub, ((W - sw) // 2, sy),
                cv2.FONT_HERSHEY_SIMPLEX, fs_sub, tc, 2, cv2.LINE_AA)
    if stats:
        lines = [
            f"Cycle frames  : {stats.get('total_frames', '-')}",
            f"Warmup frames : {stats.get('warmup_frames', '-')}",
            f"Infer frames  : {stats.get('infer_frames', '-')}",
            f"Normal votes  : {stats.get('normal_votes', '-')}",
            f"Anomaly votes : {stats.get('anomaly_votes', '-')}",
            f"Anomaly ratio : {stats.get('anomaly_ratio', 0):.1%}",
            f"Channels      : {IN_CHANNELS}  radial={USE_RADIAL_CHANNEL}",
        ]
        fs_s = max(0.55, fs_big * 0.32)
        rh   = max(24, int(H * 0.038))
        bw   = max(300, int(W * 0.30))
        bh   = rh * len(lines) + 24
        bx   = (W - bw) // 2
        by   = sy + max(30, int(H * 0.05))
        cv2.rectangle(out, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8),
                      (30, 30, 30), -1)
        cv2.rectangle(out, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), tc, 1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (bx, by + (i + 1) * rh),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_s, (210, 210, 210),
                        1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DEBUG OVERLAY  (v49 — matches nearest-pixel dbg dict)
#  [FIX-I27] draws the rectangular search ROI instead of a circle.
# ══════════════════════════════════════════════════════════════════════════════
_TUBE_BGR  = {2: (0, 255, 255), 3: (255, 0, 0), 4: (255, 0, 255)}
_TUBE_NAME = {2: "Yel(2)", 3: "Blu(3)", 4: "Pnk(4)"}


def draw_debug_overlay(frame, dbg):
    if not dbg:
        return frame
    out = frame.copy()
    scx = int(dbg.get("scx", 0));  scy = int(dbg.get("scy", 0))
    rx1 = int(dbg.get("rx1", scx)); ry1 = int(dbg.get("ry1", scy))
    rx2 = int(dbg.get("rx2", scx)); ry2 = int(dbg.get("ry2", scy))
    # [FIX-I27] draw the rectangular search region (was a circle)
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (scx, scy), (255, 255, 255),
                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    anchors    = dbg.get("anchors", {})
    angles_deg = dbg.get("angles_deg", {})
    nearest_d  = dbg.get("nearest_dist", {})
    gaps       = dbg.get("gaps_deg", [])
    seq        = dbg.get("seq", [])
    result     = dbg.get("result", "?")
    max_gap    = max((g for _, _, g in gaps), default=0)
    for fc, tc2, gd in gaps:
        col = (0, 0, 220) if abs(gd - max_gap) < 0.01 else (100, 100, 100)
        af  = anchors.get(fc)
        at  = anchors.get(tc2)
        if af and at:
            mx, my = int((af[0] + at[0]) / 2), int((af[1] + at[1]) / 2)
            cv2.putText(out, f"{gd:.0f}°", (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    for ci, (ax, ay) in anchors.items():
        iax, iay = int(ax), int(ay)
        col = _TUBE_BGR[ci]
        # line from socket centre to the tube's NEAREST pixel [FIX-I18]
        cv2.line(out, (scx, scy), (iax, iay), col, 2, cv2.LINE_AA)
        cv2.circle(out, (iax, iay), 6, col, -1)
        cv2.circle(out, (iax, iay), 6, (255, 255, 255), 1)
        dtxt = f" d={nearest_d.get(ci, 0):.0f}px" if ci in nearest_d else ""
        cv2.putText(out, f"{_TUBE_NAME[ci]} {angles_deg.get(ci, 0):.1f}°{dtxt}",
                    (iax + 8, iay + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
    H, W    = out.shape[:2]
    seq_str = " > ".join(_TUBE_NAME[c] for c in seq) if seq else "?"
    col_b   = ((0, 200, 0) if result == "OK"
               else (0, 0, 220) if result == "ANOMALY"
               else (50, 170, 200))
    banner  = f"[DBG] NEAREST-PX SEQ: {seq_str}  |  RESULT: {result}"
    (bw, bh), bl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    by = H - 14
    cv2.rectangle(out, (6, by - bh - 6), (6 + bw + 12, by + bl + 2),
                  (20, 20, 20), -1)
    cv2.putText(out, banner, (12, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_b, 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  HUD  (v49 — includes production status bar + cycle-aware HUD)
# ══════════════════════════════════════════════════════════════════════════════
_STATE_STYLE = {
    STATE_IDLE:    {"chip_bg": (45, 45, 45),   "chip_fg": (150, 150, 150), "border": (80, 80, 80)},
    STATE_WARMUP:  {"chip_bg": (100, 80, 10),  "chip_fg": (255, 220, 50),  "border": (180, 140, 20)},
    STATE_HAND:    {"chip_bg": (0, 120, 210),  "chip_fg": (255, 255, 255), "border": (0, 165, 255)},
    STATE_INSPECT: {"chip_bg": (20, 90, 170),  "chip_fg": (255, 255, 255), "border": (40, 130, 210)},
    STATE_NORMAL:  {"chip_bg": (10, 140, 40),  "chip_fg": (255, 255, 255), "border": (30, 200, 60)},
    STATE_ANOMALY: {"chip_bg": (15, 15, 210),  "chip_fg": (255, 255, 255), "border": (30, 30, 240)},
    STATE_PARTIAL: {"chip_bg": (20, 130, 160), "chip_fg": (255, 255, 255), "border": (40, 175, 195)},
}


def _put(img, text, x, y, fs, col, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, col, thick, cv2.LINE_AA)


def draw_production_status_bar(frame, state, cycle_no, passed, failed, unknown):
    out  = frame.copy()
    H, W = out.shape[:2]
    S    = W / 1280.0

    sty        = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
    status_txt = _PROD_STATUS.get(state, state)
    cycle_txt  = f"CYCLE #{cycle_no:03d}" if cycle_no else "CYCLE  --"

    fs_main = max(0.75, 0.85 * S)
    fs_sub  = max(0.55, 0.62 * S)
    pad_x   = max(18, int(22 * S))
    pad_y   = max(10, int(12 * S))
    gap     = max(28, int(34 * S))

    seg1 = cycle_txt
    seg2 = f"STATUS: {status_txt}"
    seg3 = f"PASSED: {passed}"
    seg4 = f"FAILED: {failed}"

    (w1, h1), _ = cv2.getTextSize(seg1, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w2, h2), _ = cv2.getTextSize(seg2, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w3, h3), _ = cv2.getTextSize(seg3, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w4, h4), _ = cv2.getTextSize(seg4, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)

    total_w = w1 + w2 + w3 + w4 + gap * 3 + pad_x * 2
    bar_h   = max(h1, h2, h3, h4) + pad_y * 2
    bx1 = (W - total_w) // 2
    by1 = max(8, int(10 * S))
    bx2 = bx1 + total_w
    by2 = by1 + bar_h

    ovl = out.copy()
    cv2.rectangle(ovl, (bx1, by1), (bx2, by2), (12, 12, 12), -1)
    cv2.addWeighted(ovl, 0.80, out, 0.20, 0, out)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), sty["border"], 2)

    ty = by2 - pad_y
    cx = bx1 + pad_x
    _put(out, seg1, cx, ty, fs_main, (255, 255, 255), 2);  cx += w1 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg2, cx, ty, fs_main, sty["chip_fg"] if state not in (STATE_NORMAL, STATE_ANOMALY)
         else (60, 230, 60) if state == STATE_NORMAL else (60, 60, 255), 2)
    cx += w2 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg3, cx, ty, fs_main, (60, 230, 60), 2);  cx += w3 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg4, cx, ty, fs_main, (60, 60, 255), 2)

    if unknown:
        sub = f"unknown: {unknown}"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 1)
        _put(out, sub, (W - sw) // 2, by2 + int(18 * S), fs_sub, (150, 150, 150), 1)

    return out


def draw_hud(frame, fps, frame_idx, state, socket_hit,
             status_dict, order_status, detected_seq,
             anomaly_counter=0, hand_in_roi=False,
             warmup_frame=0, warmup_retry=0,
             vote_counter=None, infer_frames=0,
             seq_stable_ctr=0, cycle_no=0, frame_ms=0.0, frame_ms_avg=0.0,
             mask_is_locked=False):
    out  = frame.copy()
    H, W = out.shape[:2]
    S    = W / 1280.0
    PAD  = max(10, int(12 * S))
    FS_XS = max(0.40, 0.42 * S);  FS_SM = max(0.48, 0.52 * S)
    FS_MD = max(0.58, 0.62 * S);  FS_LG = max(0.70, 0.76 * S)
    TK1   = max(1, int(S));        TK2   = max(1, int(2 * S))
    ROW   = max(26, int(28 * S)); DOT   = max(5, int(6 * S))

    TOP_OFFSET = max(70, int(78 * S))  # leaves room for the production status bar

    LP_W  = max(280, int(300 * S));  LP_H = PAD * 2 + ROW * 10 + 8
    LP_X, LP_Y = 10, 10 + TOP_OFFSET
    RP_W  = max(300, int(325 * S));  RP_H = PAD * 2 + ROW * 9 + 20
    RP_X  = W - RP_W - 10;  RP_Y = 10 + TOP_OFFSET

    ovl = out.copy()
    for (px, py, pw, ph) in [(LP_X, LP_Y, LP_W, LP_H),
                             (RP_X, RP_Y, RP_W, RP_H)]:
        cv2.rectangle(ovl, (px, py), (px + pw, py + ph), (14, 14, 14), -1)
    cv2.addWeighted(ovl, 0.72, out, 0.28, 0, out)

    sty = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
    for (px, py, pw, ph), bc in [
        ((LP_X, LP_Y, LP_W, LP_H), sty["border"]),
        ((RP_X, RP_Y, RP_W, RP_H), (70, 70, 70))
    ]:
        cv2.rectangle(out, (px, py), (px + pw, py + ph), bc, 1)

    lx = LP_X + PAD;  ly = LP_Y + PAD + ROW - 4;  vx = lx + max(60, int(64 * S))
    _put(out, "FPS",   lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{fps:5.1f}", vx, ly, FS_LG, (220, 220, 220), TK2);  ly += ROW + 4
    _put(out, "FRAME", lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_idx:06d}", vx, ly, FS_MD, (200, 200, 200), TK1);  ly += ROW + 4

    # [FIX-I28] Per-frame latency (ms) — current frame and running average.
    ms_col = (60, 60, 255) if frame_ms_avg >= FRAME_MS_WARN_THRESHOLD else (200, 200, 200)
    _put(out, "MS",    lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_ms:5.1f} (avg {frame_ms_avg:5.1f})",
         vx, ly, FS_SM, ms_col, TK1);  ly += ROW + 6

    chip = _CHIP_LABEL.get(state, state)
    (cw, ch), bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, FS_SM, TK1)
    cpx, cpy = max(10, int(11 * S)), max(6, int(7 * S))
    cx1, cy1 = lx, ly;  cx2, cy2 = cx1 + cw + cpx * 2, cy1 + ch + bl + cpy * 2
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["chip_bg"], -1)
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["border"], 1)
    _put(out, chip, cx1 + cpx, cy1 + cpy + ch, FS_SM, sty["chip_fg"], TK1)
    if 0 < anomaly_counter < N_ANOMALY_CONFIRM:
        _put(out, f"({anomaly_counter}/{N_ANOMALY_CONFIRM})",
             cx2 + 6, cy1 + cpy + ch, FS_XS, (160, 80, 80), TK1)
    ly = cy2 + 6

    # Warmup progress bar
    if state == STATE_WARMUP and WARMUP_FRAMES > 0:
        bw   = cx2 - cx1;  bh = max(6, int(7 * S))
        prog = min(warmup_frame / (WARMUP_FRAMES * (warmup_retry + 1)), 1.0)
        cv2.rectangle(out, (cx1, ly), (cx1 + bw, ly + bh), (60, 60, 60), -1)
        cv2.rectangle(out, (cx1, ly), (cx1 + int(bw * prog), ly + bh),
                      (180, 140, 20), -1);  ly += bh + 4
        if warmup_retry > 0:
            _put(out, f"retry {warmup_retry}/{MAX_WARMUP_RETRIES}",
                 cx1, ly + ROW - 6, FS_XS, (160, 120, 40), TK1);  ly += ROW

    if state == STATE_HAND:
        # [FIX-I25] Nothing tube-related is being tracked while a hand is
        # in the ROI — say so plainly instead of showing a live counter.
        _put(out, "DETECTIONS PAUSED (hand in ROI)",
             lx, ly + ROW - 6, FS_XS, _ROI_COL_AMBER, TK1);  ly += ROW
    else:
        # Live inference frame counter (replaces frozen-age)
        _put(out, f"LIVE INFER   [{infer_frames}f]",
             lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        # [FIX-I37] Mask-freeze status — LOCKED (green) means the current
        # frame is reusing the exact frozen pred_map untouched (no
        # inference ran this frame); REFRESHED (amber) means motion was
        # detected and a fresh inference just ran.
        if mask_is_locked:
            _put(out, "MASK: LOCKED", lx + 190, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        else:
            _put(out, "MASK: REFRESHED", lx + 190, ly + ROW - 6, FS_XS, (0, 165, 255), TK1)
        ly += ROW

        # [FIX-I6] Sequence stability counter
        if seq_stable_ctr > 0:
            sc_col = (60, 220, 60) if seq_stable_ctr >= MIN_SEQ_STABLE else (160, 160, 40)
            _put(out, f"SEQ STABLE  {seq_stable_ctr}/{MIN_SEQ_STABLE}",
                 lx, ly + ROW - 6, FS_XS, sc_col, TK1);  ly += ROW

    if vote_counter is not None and vote_counter.total > 0:
        ly += 2
        _put(out, f"OK : {vote_counter.normal_votes}",
             lx, ly + ROW - 6, FS_XS, (60, 220, 60), TK1);  ly += ROW
        _put(out, f"AN : {vote_counter.anomaly_votes}",
             lx, ly + ROW - 6, FS_XS, (80, 80, 230), TK1);  ly += ROW

    rx = RP_X + PAD;  ry = RP_Y + PAD
    _put(out, "INSPECTION STATUS", rx, ry + ROW - 6, FS_XS, (100, 100, 100), TK1)
    ry += ROW + 4
    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        hc = _ROI_COL_AMBER;  ht = "HAND     IN ROI"
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, hc, -1)
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, (255, 255, 255), 1)
    else:
        hc = (70, 70, 70);  ht = "HAND     CLEAR"
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, hc, -1)
    _put(out, ht, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, hc, TK1);  ry += ROW + 4

    s_col, s_txt = (
        ((0, 210, 100), "SOCKET   PRESENT")
        if socket_hit and socket_hit["class"] == CLS_SOCKET
        else ((60, 60, 220), "SOCKET   ABSENT")
    )
    cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, s_col, -1)
    _put(out, s_txt, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, s_col, TK1);  ry += ROW + 4
    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        # [FIX-I25] Show explicit "Paused" for every tube row instead of
        # holding over the last-known present/absent status.
        for ci in (2, 3, 4):
            dc  = (50, 50, 50)
            tc3 = (140, 140, 140)
            cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   PAUSED",
                 rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4
    else:
        for ci in (2, 3, 4):
            pres = status_dict.get(ci, "Absent") == "Present"
            dc   = CLASS_INFO[ci][1] if pres else (50, 50, 50)
            tc3  = (180, 255, 180) if pres else (90, 90, 90)
            cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   {'PRESENT' if pres else 'ABSENT'}",
                 rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4

    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        seq_str = "-"
    else:
        seq_str = " > ".join(TUBE_SHORT[c] for c in detected_seq) if detected_seq else "-"
    _put(out, f"SEQ  {seq_str}", rx, ry + ROW - 6, FS_SM, (160, 160, 160), TK1);  ry += ROW + 6

    if hand_in_roi:
        _put(out, "INFERENCE   PAUSED", rx, ry + ROW - 4, FS_MD, _ROI_COL_AMBER, TK2)
    else:
        vcol, vtxt = {
            "OK":      ((60, 220, 60),  "ORDER  [OK]  NORMAL"),
            "ANOMALY": ((40, 40, 230),  "ORDER  [!!]  ANOMALY"),
            "PARTIAL": ((30, 160, 200), "ORDER  [??]  PARTIAL"),
        }.get(order_status, ((100, 100, 100), "ORDER  ---  N/A"))
        _put(out, vtxt, rx, ry + ROW - 4, FS_MD, vcol, TK2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_verdict_dir(out_dir, verdict):
    folder = verdict if verdict in ("NORMAL", "ANOMALY", "UNKNOWN") else "UNKNOWN"
    d      = os.path.join(out_dir, folder)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d, folder


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PROCESSOR — SINGLE VIDEO, MULTI-CYCLE
# ══════════════════════════════════════════════════════════════════════════════
def process_video_cycles(video_path, resolved_output_dir, seg_net,
                         yolo_socket, yolo_pose, print_summary,
                         enable_debug=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}");  return None

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {Path(video_path).name} — {src_w}x{src_h} @ {fps_src:.1f} fps")

    video_stem = Path(video_path).stem
    cycles     = CycleManager(resolved_output_dir, video_stem, fps_src, (src_w, src_h))

    seg_engine   = SegmentationEngine(seg_net)
    anomaly_gate = AnomalyConfirmGate(N_ANOMALY_CONFIRM)
    latch_gate   = ResultLatchGate(LATCH_FRAMES)
    vote_counter = VoteCounter(VERDICT_THR)
    seq_gate     = SequenceStabilityGate()

    WIN = f"v50_Merged | {os.path.basename(video_path)} | Q=quit D=debug F=fs M=maskdbg"
    if SHOW_PREVIEW:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(src_w, 1280), min(src_h, 720))

    ZERO_PRED  = np.zeros((src_h, src_w), dtype=np.int32)
    EMPTY_STAT = {2: "Absent", 3: "Absent", 4: "Absent"}

    state              = STATE_IDLE
    fps_ema            = fps_src
    frame_idx          = 0
    invisible_roi      = None
    socket_latched     = False
    fullscreen         = False
    socket_drop_frames = 0

    warmup_frame_count = 0
    warmup_retry       = 0
    warmup_done        = False
    infer_frames       = 0
    cycle_total_frames = 0

    # [FIX-I5] Last-known socket centre cache
    last_socket_centre = None
    # [FIX-I35] Last-known socket bbox size (w, h) cache — used to
    # auto-scale the quad_ellipse ROI even on frames where YOLO misses
    # the socket but it's still latched via SOCKET_RESET_GRACE.
    last_socket_bbox_size = None

    # [FIX-I37] Mask-freeze state — the frozen pred_map (reused untouched
    # while the tube hasn't moved) and the reference grayscale ROI crop
    # it was frozen against.
    mask_frozen_pred     = None
    mask_frozen_gray_roi = None
    mask_is_locked        = False   # for HUD/debug display only

    # [FIX-I28] Per-frame latency (ms) tracking
    frame_ms_ema        = 0.0

    prev_pred    = ZERO_PRED.copy()
    prev_status  = EMPTY_STAT.copy()
    prev_order   = "N/A"
    prev_seq     = []
    prev_dbg     = {}
    last_vis     = None

    peak_status = EMPTY_STAT.copy()
    peak_seq    = []
    peak_order  = "N/A"

    _prev_gray_ref[0] = None

    def _finalize_cycle():
        nonlocal last_vis
        if not cycles.active:
            return
        final_verdict = vote_counter.final_verdict()
        anomaly_ratio = vote_counter.anomaly_votes / max(vote_counter.total, 1)
        stats_card = {
            "total_frames":  cycle_total_frames,
            "warmup_frames": warmup_frame_count,
            "infer_frames":  infer_frames,
            "normal_votes":  vote_counter.normal_votes,
            "anomaly_votes": vote_counter.anomaly_votes,
            "anomaly_ratio": anomaly_ratio,
        }
        if last_vis is not None:
            card = draw_final_verdict_overlay(
                last_vis, final_verdict, cycle_no=cycles.cycle_no, stats=stats_card)
            cycles.hold_final_frame(card)
            if SHOW_PREVIEW:
                cv2.imshow(WIN, card);  cv2.waitKey(int(VERDICT_HOLD_SEC * 1000))

        display_verdict = _ORDER_DISPLAY.get(
            peak_order if peak_order != "N/A" else final_verdict, "N/A")
        seq_display = " > ".join(TUBE_SHORT[c] for c in peak_seq) if peak_seq else "-"

        # NOTE: Excel logging (append_to_excel) only persists a reduced set
        # of columns (cycle number, video name, status, output folder path)
        # — see _EXCEL_COLUMNS. The richer `extra` dict below is still kept
        # for the printed console summary / cycle_summaries bookkeeping.
        extra = {
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":          os.path.basename(video_path),
            "socket":            "Present",
            "tube_blue":         peak_status.get(2, "Absent"),
            "trans_mid_tube":    peak_status.get(3, "Absent"),
            "trans_end_tube":    peak_status.get(4, "Absent"),
            "detected_sequence": seq_display,
            "tube_order_result": display_verdict,
            "warmup_frames":     warmup_frame_count,
            "infer_frames":      infer_frames,
            "ok_votes":          vote_counter.normal_votes,
            "anomaly_votes":     vote_counter.anomaly_votes,
            "anomaly_ratio":     round(anomaly_ratio, 4),
            "total_frames":      cycle_total_frames,
            "avg_fps":           float(fps_ema),
            "ema_alpha":         EMA_ALPHA,
            "sharpening":        "ON" if TUBE_BOUNDARY_SHARPENING else "OFF",
            "seq_stable_min":    MIN_SEQ_STABLE,
            "channels":          IN_CHANNELS,
            "radial_channel":    "ON" if USE_RADIAL_CHANNEL else "OFF",
            "debug_mode":        "ON" if enable_debug else "OFF",
        }
        summary = cycles.end_cycle(final_verdict, extra_metrics=extra)
        if summary and print_summary:
            print("\n" + "=" * 62 +
                  f"\n CYCLE #{summary['cycle_no']:03d} SUMMARY\n" + "=" * 62)
            for k, v in summary.items():
                print(f"  {k:<26}: {v}")
            print("=" * 62 + "\n")
        if summary:
            append_to_excel(summary, resolved_output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0  = time.perf_counter()
        vis = frame.copy()

        # ── Socket detection ──────────────────────────────────────────────
        sock_hit      = detect_socket(yolo_socket, frame, YOLO_SOCKET_CONF)
        socket_now    = sock_hit is not None and sock_hit["class"] == CLS_SOCKET
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET

        # [FIX-I5] Cache socket centre whenever YOLO sees the socket
        if socket_now:
            x1, y1, x2, y2    = sock_hit["bbox"]
            last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            # [FIX-I35] Cache bbox size too, for ROI auto-scaling
            last_socket_bbox_size = (x2 - x1, y2 - y1)

        was_latched = socket_latched

        if socket_now:
            if socket_drop_frames > 0:
                print(f"[INFO] flicker absorbed: socket reappeared after "
                      f"{socket_drop_frames} 'no socket' frame(s) — "
                      f"still cycle #{cycles.cycle_no:03d}, not counted as new")
            invisible_roi      = build_roi(frame.shape, sock_hit["bbox"])
            socket_latched     = True
            socket_drop_frames = 0
        elif no_socket_now:
            socket_drop_frames += 1
            if socket_drop_frames >= SOCKET_RESET_GRACE:
                print(f"[INFO] removal confirmed after "
                      f"{socket_drop_frames} consecutive 'no socket' frames "
                      f"— closing cycle #{cycles.cycle_no:03d}")
                invisible_roi      = None
                socket_latched     = False
                warmup_frame_count = 0
                warmup_retry       = 0
                warmup_done        = False
                infer_frames       = 0
                cycle_total_frames = 0
                last_socket_centre = None
                last_socket_bbox_size = None
                mask_frozen_pred     = None
                mask_frozen_gray_roi = None
                mask_is_locked        = False
                seg_engine.reset()
                anomaly_gate.reset()
                latch_gate.reset()
                seq_gate.reset()
                _prev_gray_ref[0] = None
                prev_pred   = ZERO_PRED.copy()
                prev_status = EMPTY_STAT.copy()
                prev_order  = "N/A"
                prev_seq    = []
                prev_dbg    = {}
                peak_status = EMPTY_STAT.copy()
                peak_seq    = []
                peak_order  = "N/A"

        if socket_latched and not was_latched:
            cycles.start_cycle()
            vote_counter.reset()
            print(f"[INFO] vote counter reset for cycle #{cycles.cycle_no:03d} "
                  f"(normal=0 anomaly=0 — independent of previous cycle)")
        if was_latched and not socket_latched:
            _finalize_cycle()

        hand_in_roi = detect_hand_in_roi(
            yolo_pose, frame, invisible_roi, YOLO_POSE_CONF
        ) if invisible_roi is not None else False

        current_dbg = {}

        # ── State machine ────────────────────────────────────────────────
        if not socket_latched or invisible_roi is None:
            state        = STATE_IDLE
            pred_map     = ZERO_PRED
            status_dict  = EMPTY_STAT
            order_status = "N/A"
            detected_seq = []
            mask_is_locked = False
            anomaly_gate.reset()
            latch_gate.reset()
            vis = draw_socket_box(vis, sock_hit)

        elif hand_in_roi:
            # ── HAND: inference paused. [FIX-I25] Nothing tube-related is
            # shown or held over from the previous frame while a hand is in
            # the ROI — no segmentation overlay, no tube status, no
            # sequence. Only the socket bounding box (a YOLO detection
            # independent of tube segmentation) is still drawn so the
            # ROI/cycle context stays visible on screen.
            state        = STATE_HAND
            pred_map     = ZERO_PRED
            status_dict  = EMPTY_STAT
            order_status = "N/A"
            detected_seq = []
            current_dbg  = {}
            # [FIX-I37] A hand in the ROI may reposition the tube, so
            # invalidate any frozen mask — the next live-inference frame
            # after the hand leaves must run fresh inference rather than
            # blindly reusing a now-possibly-stale frozen snapshot.
            mask_frozen_pred     = None
            mask_frozen_gray_roi = None
            mask_is_locked        = False
            vis = draw_socket_box(vis, sock_hit)

        else:
            # ── LIVE INFERENCE (warmup) / MOTION-GATED MASK FREEZE (post-warmup) ──
            cycle_total_frames += 1
            # [FIX-I26] Bug fix: this call previously relied on infer()'s
            # apply_identity_lock default (True), so the identity lock was
            # silently enforcing from frame 1 of every cycle regardless of
            # LOCK_ENGAGE_MODE/warmup — before the EMA had converged. That
            # let a noisy/occluded first few frames (hand at frame edge,
            # partial view, etc.) bake in a wrong lock, which then had to
            # be fought back out by later real evidence — producing
            # exactly the kind of persistent flicker seen when a cycle's
            # warmup happens to start on an ambiguous frame.
            lock_engage = (
                True if LOCK_ENGAGE_MODE == "immediate" else warmup_done
            )

            # [FIX-I30/I35] Prefer this frame's freshly-detected socket
            # bbox centre/size; fall back to the cached last-known values
            # if YOLO missed the socket on this particular frame
            # (socket_latched is still True here thanks to
            # SOCKET_RESET_GRACE).
            roi_center = (
                ((sock_hit["bbox"][0] + sock_hit["bbox"][2]) / 2.0,
                 (sock_hit["bbox"][1] + sock_hit["bbox"][3]) / 2.0)
                if sock_hit else last_socket_centre
            )
            roi_bbox_size = (
                (sock_hit["bbox"][2] - sock_hit["bbox"][0],
                 sock_hit["bbox"][3] - sock_hit["bbox"][1])
                if sock_hit else last_socket_bbox_size
            )

            def _run_fresh_inference():
                pm = seg_engine.infer(
                    frame, socket_centre=last_socket_centre,
                    apply_identity_lock=lock_engage)
                # [FIX-I30] Confine tube-class detections (2/3/4) to the
                # socket-adjacent ROI only — suppress anything further out
                # along the coiled/looped part of the tubes.
                pm = restrict_mask_to_socket_roi(
                    pm, roi_center, bbox_size=roi_bbox_size)
                # [FIX-I36] Hard keep-out zone for a known distractor
                # tube/cable near the socket — applied after inclusion so
                # it always wins regardless of how the inclusion radii
                # are tuned.
                pm = apply_exclusion_zone(
                    pm, roi_center, bbox_size=roi_bbox_size)
                return pm

            if not warmup_done:
                # During warmup we always want the REAL per-frame result —
                # this is how we detect a blank/bad warmup and decide
                # whether to retry, and it's how the identity-lock EMA
                # actually converges before enforcement turns on. The mask
                # freeze below only ever applies once warmup_done is True.
                pred_map    = _run_fresh_inference()
                mask_is_locked = False
            elif not MASK_FREEZE_ENABLED:
                # Freeze feature disabled — original v45/v49 behaviour:
                # live inference every single frame, no freezing at all.
                pred_map    = _run_fresh_inference()
                mask_is_locked = False
            else:
                # [FIX-I37] Post-warmup, motion-gated: only re-run
                # inference (and re-freeze) if the tube has actually moved
                # since the last freeze; otherwise reuse the exact frozen
                # pred_map object untouched — this is not just "similar",
                # it is IDENTICAL frame-to-frame, so there is zero flicker
                # while the tube sits still.
                gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rx1, ry1, rx2, ry2 = invisible_roi
                crop_now = gray_now[ry1:ry2, rx1:rx2]

                need_refresh = (
                    mask_frozen_pred is None
                    or mask_frozen_gray_roi is None
                    or mask_has_moved(crop_now, mask_frozen_gray_roi)
                )

                if need_refresh:
                    pred_map = _run_fresh_inference()
                    mask_frozen_pred     = pred_map.copy()
                    mask_frozen_gray_roi = crop_now.copy()
                    mask_is_locked        = False
                    print(f"[INFO] [FIX-I37] mask refreshed (motion detected) — "
                          f"cycle#{cycles.cycle_no:03d} frame {frame_idx:05d}")
                else:
                    pred_map      = mask_frozen_pred
                    mask_is_locked = True

            socket_bbox = sock_hit["bbox"] if sock_hit else None
            status_dict, raw_order, detected_seq, current_dbg = \
                evaluate_tube_order(pred_map, socket_bbox, debug=enable_debug)

            # [FIX-I6] Stability gate before anomaly confirm gate
            stable_order = seq_gate.update(raw_order, detected_seq)
            gate_result  = anomaly_gate.update(stable_order)
            order_status = gate_result
            latch_gate.update(gate_result)

            has_tubes = any((pred_map == ci).any() for ci in (2, 3, 4))

            if not warmup_done:
                # ── Still in warmup ─────────────────────────────────────
                warmup_frame_count += 1
                state = STATE_WARMUP

                if has_tubes:
                    vis = draw_seg_overlay(vis, pred_map)
                elif seg_engine.last_raw_pred is not None:
                    vis = draw_raw_argmax_fallback(vis, seg_engine.last_raw_pred)

                target = WARMUP_FRAMES * (warmup_retry + 1)
                if warmup_frame_count >= target:
                    if not has_tubes and warmup_retry < MAX_WARMUP_RETRIES:
                        warmup_retry += 1
                        safe_print(f"[WARN] f{frame_idx:05d}: warmup blank → retry "
                              f"{warmup_retry}/{MAX_WARMUP_RETRIES}")
                    else:
                        warmup_done = True
                        print(f"[WARMUP DONE] cycle#{cycles.cycle_no:03d} f{frame_idx:05d}  "
                              f"tube_px={[(ci, int((pred_map == ci).sum())) for ci in (2, 3, 4)]}")

            else:
                # ── Post-warmup: live inference, mask follows object ────
                infer_frames += 1
                vote_counter.record(gate_result)

                state = {"OK":      STATE_NORMAL,
                         "ANOMALY": STATE_ANOMALY,
                         "PARTIAL": STATE_PARTIAL
                         }.get(order_status, STATE_INSPECT)

                if any(v == "Present" for v in status_dict.values()) and order_status != "N/A":
                    peak_status = dict(status_dict)
                    peak_seq    = list(detected_seq)
                    peak_order  = order_status

                if has_tubes:
                    vis = draw_seg_overlay(vis, pred_map)
                elif seg_engine.last_raw_pred is not None:
                    vis = draw_raw_argmax_fallback(vis, seg_engine.last_raw_pred)

            # Save for potential debug overlay only — [FIX-I25] no longer
            # used to "hold over" detections into a subsequent HAND state.
            prev_pred   = pred_map
            prev_status = status_dict
            prev_order  = order_status
            prev_seq    = detected_seq
            prev_dbg    = current_dbg

            vis = draw_socket_box(vis, sock_hit)

        if enable_debug and current_dbg:
            vis = draw_debug_overlay(vis, current_dbg)

        # [FIX-I28] Per-frame latency (ms) check — measured for the whole
        # per-frame pipeline (detection + inference + rendering), before
        # the HUD/status-bar drawing calls below are added on top.
        frame_ms = (time.perf_counter() - t0) * 1000.0
        frame_ms_ema = (
            frame_ms if frame_idx == 1
            else FRAME_MS_EMA_ALPHA * frame_ms + (1 - FRAME_MS_EMA_ALPHA) * frame_ms_ema
        )
        if frame_ms > FRAME_MS_WARN_THRESHOLD:
            print(f"[WARN] f{frame_idx:05d}: slow frame {frame_ms:.1f}ms "
                  f"(avg {frame_ms_ema:.1f}ms, threshold {FRAME_MS_WARN_THRESHOLD:.0f}ms)")

        fps_ema = 0.88 * fps_ema + 0.12 / max(time.perf_counter() - t0, 1e-6)

        vis = draw_production_status_bar(
            vis, state, cycles.cycle_no, cycles.passed, cycles.failed, cycles.unknown)
        vis = draw_hud(
            vis, fps_ema, frame_idx, state, sock_hit,
            status_dict, order_status, detected_seq,
            anomaly_counter  = anomaly_gate._count,
            hand_in_roi      = hand_in_roi,
            warmup_frame     = warmup_frame_count,
            warmup_retry     = warmup_retry,
            vote_counter     = vote_counter,
            infer_frames     = infer_frames,
            seq_stable_ctr   = seq_gate._stable_ct,
            cycle_no         = cycles.cycle_no,
            frame_ms         = frame_ms,
            frame_ms_avg     = frame_ms_ema,
            mask_is_locked   = mask_is_locked,
        )

        cycles.write(vis);  last_vis = vis

        if SHOW_PREVIEW:
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit");  break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("d"):
                enable_debug = not enable_debug
                print(f"[INFO] Debug {'ON' if enable_debug else 'OFF'}")
            if key == ord("m"):
                global MASK_DEBUG
                MASK_DEBUG = not MASK_DEBUG
                print(f"[INFO] Mask debug {'ON' if MASK_DEBUG else 'OFF'}")

    if cycles.active:
        _finalize_cycle()

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyWindow(WIN)

    return cycles


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL LOGGING  (v49 — cycle-based)
#  [FIX-I29] Reduced to only: Cycle No., Video File, Final Verdict (status),
#            and Output Path (video folder saving path). All socket / tube
#            presence, sequence, and other diagnostic columns have been
#            removed from the Excel log per request.
# ══════════════════════════════════════════════════════════════════════════════
_EXCEL_COLUMNS = [
    ("cycle_no",         "Cycle No."),
    ("filename",         "Video File"),
    ("final_verdict",    "Status"),
    ("output_path",      "Video Folder Saving Path"),
]
_VERDICT_FILLS  = {"NORMAL": "C6EFCE", "ANOMALY": "FFC7CE",
                   "PARTIAL": "FFEB9C", "N/A": "F2F2F2", "UNKNOWN": "FFEB9C"}


def append_to_excel(run_metrics, excel_dir):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[WARN] openpyxl not installed — skipping Excel.");  return

    Path(excel_dir).mkdir(parents=True, exist_ok=True)
    excel_path = os.path.join(excel_dir, "inspection_log.xlsx")

    def _create_new_workbook():
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Inspection Log"
        thin = Side(style="thin", color="BFBFBF")
        bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
        for ci, (_, header) in enumerate(_EXCEL_COLUMNS, 1):
            cell           = ws.cell(row=1, column=ci, value=header)
            cell.font      = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            cell.fill      = PatternFill(start_color="1F4E78", end_color="1F4E78",
                                         fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border    = bdr
        ws.row_dimensions[1].height = 32
        ws.freeze_panes = "A2"
        return wb, ws, 1

    if os.path.exists(excel_path):
        try:
            wb = openpyxl.load_workbook(excel_path)
            ws = wb.active
            next_sr = max(1, ws.max_row)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
            broken_path = os.path.join(
                excel_dir,
                f"inspection_log.broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            try:
                shutil.move(excel_path, broken_path)
                print(f"[WARN] Existing Excel log was unreadable ({type(exc).__name__}); "
                      f"moved to {broken_path}")
            except OSError:
                print(f"[WARN] Existing Excel log was unreadable ({type(exc).__name__}); "
                      f"starting a new file")
            wb, ws, next_sr = _create_new_workbook()
    else:
        wb, ws, next_sr = _create_new_workbook()

    row_vals = []
    for key, _ in _EXCEL_COLUMNS:
        if key == "cycle_no":
            row_vals.append(int(run_metrics.get(key, 0)))
        else:
            row_vals.append(run_metrics.get(key, "N/A"))
    ws.append(row_vals)

    cr   = ws.max_row
    thin = Side(style="thin", color="BFBFBF")
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
    verdict_keys = {"final_verdict"}
    center_keys  = {"cycle_no", "final_verdict"}

    for ci, (key, _) in enumerate(_EXCEL_COLUMNS, start=1):
        cell           = ws.cell(row=cr, column=ci)
        cell.font      = Font(name="Arial", size=10)
        cell.border    = bdr
        cell.alignment = Alignment(
            horizontal="center" if key in center_keys else "left",
            vertical="center")
        val = str(cell.value or "")
        if key in verdict_keys:
            hx        = _VERDICT_FILLS.get(val, "F2F2F2")
            cell.fill = PatternFill(start_color=hx, end_color=hx, fill_type="solid")
            cell.font = Font(name="Arial", size=10, bold=True)

    for ci in range(1, len(_EXCEL_COLUMNS) + 1):
        cl = get_column_letter(ci)
        mx = max(len(str(ws.cell(row=r, column=ci).value or ""))
                 for r in range(1, ws.max_row + 1))
        ws.column_dimensions[cl].width = max(mx + 4, 14)

    for attempt in range(1, MAX_EXCEL_RETRIES + 1):
        try:
            wb.save(excel_path)
            safe_print(f"[EXCEL] Cycle #{run_metrics.get('cycle_no','-')} → {excel_path}")
            return
        except PermissionError:
            if attempt == MAX_EXCEL_RETRIES:
                raise
            safe_print(f"[WARN] Excel locked, retry {attempt}/{MAX_EXCEL_RETRIES} in 5s…")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
def run_single_video(video_path, seg_model_path, out_base,
                     yolo_socket_path, hand_pose_path, print_summary,
                     enable_debug=False, forced_channels=None,
                     require_gpu=True, use_half=True, gpu_warmup_hw=None):

    # [FIX-I38] Resolve DEVICE here (with full diagnostics + GPU sanity
    # benchmark) instead of the old one-line silent
    # `"cuda" if torch.cuda.is_available() else "cpu"` at import time.
    # This raises immediately (by default) if CUDA is not usable, instead
    # of quietly falling back to CPU for all three models.
    global DEVICE, USE_HALF, GPU_WARMUP_HW
    DEVICE = resolve_device(require_gpu=require_gpu)

    # [FIX-I39] USE_HALF must be resolved BEFORE any model is built/loaded
    # below (seg_net, yolo_socket, yolo_pose all branch on it at load
    # time). FP16 only makes sense on an actual CUDA device.
    USE_HALF = bool(use_half) and DEVICE.startswith("cuda")
    if USE_HALF:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7:  # Pascal (GTX 10-series) or older
            print(f"  [WARN] GPU is compute capability {cap[0]}.{cap[1]} (Pascal or older).")
            print(f"  [WARN] Disabling FP16 (USE_HALF) as it is heavily bottlenecked on this architecture.")
            USE_HALF = False
    if gpu_warmup_hw is not None:
        GPU_WARMUP_HW = gpu_warmup_hw

    print(f"[INFO] Device              : {DEVICE}")
    print(f"[INFO] [FIX-I39] FP16 (half precision) : {USE_HALF}"
          + ("" if DEVICE.startswith("cuda") else "  (forced OFF — no CUDA device)"))
    print(f"[INFO] [FIX-I39] GPU warmup shape (HxW): {GPU_WARMUP_HW}")
    print(f"[INFO] Mode                : SINGLE VIDEO / MULTI-CYCLE (live inference, merged v45+v49)")
    print(f"[INFO] Input video         : {video_path}")
    print(f"[INFO] HSV gate            : {'ON' if USE_HSV_GATE else 'OFF'}")
    safe_print(f"[INFO] Warmup              : {WARMUP_FRAMES} frames × up to {MAX_WARMUP_RETRIES} retries")
    print(f"[INFO] [FIX-I1] EMA alpha  : {EMA_ALPHA}")
    safe_print(f"[INFO] [FIX-I2/I8] Sharpen τ : {TUBE_SHARPNESS_TEMP}  enabled={TUBE_BOUNDARY_SHARPENING}")
    print(f"[INFO] [FIX-I3/I8] Conf thr : {CLASS_CONF_THR}")
    print(f"[INFO] [FIX-I6] Seq stable : {MIN_SEQ_STABLE} frames")
    print(f"[INFO] [FIX-I8] MIN_TUBE_PX: {MIN_TUBE_PX}  MIN_AREA_PX: {MIN_AREA_PX}")
    print(f"[INFO] [FIX-I18/I27] Nearest-pixel angular gate, RECTANGULAR search "
          f"half-side: {NEAREST_SEARCH_RADIUS}px")
    print(f"[INFO] [FIX-I21/I41/I42] Mask dilation (width): enabled={MASK_DILATE_ENABLED}  "
          f"size={MASK_DILATE_SZ}px  classes={MASK_DILATE_CLASSES}")
    print(f"[INFO] [FIX-I43] Mask close (gap bridging)     : enabled={MASK_CLOSE_ENABLED}  "
          f"size={MASK_CLOSE_SZ}px")
    print(f"[INFO] [FIX-I44] Overlay alpha (solidity)      : {OVERLAY_ALPHA}")
    print(f"[INFO] [FIX-I23/I24] 3-way tube identity lock with optical-flow tracking: "
          f"enabled={IDENTITY_HYSTERESIS_ENABLED}  margin={TUBE_IDENTITY_MARGIN}  "
          f"ema_alpha={IDENTITY_EMA_ALPHA}")
    print(f"[INFO] [FIX-I25] Clean HAND state (no detections shown while hand in ROI): enabled")
    print(f"[INFO] [FIX-I28] Frame latency warn threshold: {FRAME_MS_WARN_THRESHOLD:.0f}ms")
    print(f"[INFO] [FIX-I29] Excel columns reduced to: cycle_no, filename, final_verdict, output_path")
    print(f"[INFO] [FIX-I30/I31/I32/I33/I34/I35] Socket-adjacent mask ROI gate: "
          f"shape={MASK_ROI_SHAPE}  offset=({MASK_ROI_OFFSET_X},{MASK_ROI_OFFSET_Y})")
    if MASK_ROI_AUTO_SCALE:
        print(f"[INFO]   auto-scale ON — quad_ellipse radii = socket-bbox-relative multipliers  "
              f"up={MASK_ROI_MULT_UP}x_h  down={MASK_ROI_MULT_DOWN}x_h  "
              f"left={MASK_ROI_MULT_LEFT}x_w  right={MASK_ROI_MULT_RIGHT}x_w  "
              f"(fallback fixed px: up={MASK_ROI_RADIUS_UP} down={MASK_ROI_RADIUS_DOWN} "
              f"left={MASK_ROI_RADIUS_LEFT} right={MASK_ROI_RADIUS_RIGHT})")
    else:
        print(f"[INFO]   auto-scale OFF — quad_ellipse radii  up={MASK_ROI_RADIUS_UP}  "
              f"down={MASK_ROI_RADIUS_DOWN}  left={MASK_ROI_RADIUS_LEFT}  "
              f"right={MASK_ROI_RADIUS_RIGHT}")
    print(f"[INFO]   ellipse radii       rx={MASK_ROI_RADIUS_X}  ry={MASK_ROI_RADIUS_Y}   "
          f"circle/square radius={MASK_ROI_RADIUS}   "
          f"polygon_pts={len(MASK_ROI_POLYGON) if MASK_ROI_POLYGON else 0}  "
          f"classes={MASK_ROI_CLASSES}")
    print(f"[INFO] [FIX-I36] Hard exclusion zone: enabled={MASK_EXCLUDE_ENABLED}  "
          f"auto_scale={MASK_EXCLUDE_AUTO_SCALE}")
    if MASK_EXCLUDE_AUTO_SCALE:
        print(f"[INFO]   exclude multipliers  offset=({MASK_EXCLUDE_OFFSET_MULT_X}x_w,"
              f"{MASK_EXCLUDE_OFFSET_MULT_Y}x_h)  radius=({MASK_EXCLUDE_RADIUS_MULT_X}x_w,"
              f"{MASK_EXCLUDE_RADIUS_MULT_Y}x_h)")
    else:
        print(f"[INFO]   exclude fixed px     offset=({MASK_EXCLUDE_OFFSET_X},"
              f"{MASK_EXCLUDE_OFFSET_Y})  radius=({MASK_EXCLUDE_RADIUS_X},{MASK_EXCLUDE_RADIUS_Y})")
    print(f"[INFO] [FIX-I37] Mask freeze (post-warmup motion-gated re-inference): "
          f"enabled={MASK_FREEZE_ENABLED}  motion_thr={MASK_FREEZE_MOTION_THR}px  "
          f"motion_frac={MASK_FREEZE_MOTION_FRAC}")
    print(f"[INFO] Socket reset grace  : {SOCKET_RESET_GRACE} frames")
    print(f"[INFO] Mask debug          : {'ON' if MASK_DEBUG else 'OFF (toggle with M key or --mask_debug)'}")
    print(f"[INFO] PERSIST_FRAMES      : {PERSIST_FRAMES}")
    print(f"[INFO] Verdict thr         : {VERDICT_THR:.0%}")
    print(f"[INFO] Output root         : {out_base}")

    global IN_CHANNELS, USE_RADIAL_CHANNEL
    if forced_channels is not None:
        IN_CHANNELS = forced_channels
        print(f"[INFO] in_channels forced  : {IN_CHANNELS}")
    else:
        IN_CHANNELS = detect_in_channels_from_ckpt(seg_model_path)
    USE_RADIAL_CHANNEL = (IN_CHANNELS == 4)
    print(f"[INFO] in_channels         : {IN_CHANNELS}  radial={USE_RADIAL_CHANNEL}")

    seg_net = smp.UnetPlusPlus(
        encoder_name="tu-hrnet_w18",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(DEVICE)
    ckpt = torch.load(seg_model_path, map_location=DEVICE)
    seg_net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    seg_net.eval()
    # [FIX-I39] Cast the whole model to FP16 when enabled, so raw_infer's
    # autocast block is running actual half-precision weights/activations,
    # not just autocast-wrapped FP32 ones — this is what gets the real
    # throughput/VRAM win on top of TF32.
    if USE_HALF:
        seg_net = seg_net.half()
    warmup_dtype = torch.float16 if USE_HALF else torch.float32
    with torch.no_grad():
        seg_net(torch.zeros(1, IN_CHANNELS, *IMG_SIZE, device=DEVICE, dtype=warmup_dtype))
    print("[INFO] GPU warmup done.")
    # [FIX-I38] Confirm (don't assume) that the segmentation model's
    # parameters really did end up on the expected device.
    _log_model_device("Segmentation UNet++", f"{next(seg_net.parameters()).device}  half={USE_HALF}")

    # [FIX-I38] Explicitly bind both YOLO models onto DEVICE — see the
    # updated load_yolo() for the .to(device) + realistic-size FP16
    # warmup-inference call.
    yolo_socket = load_yolo(yolo_socket_path, "Socket", device=DEVICE)
    yolo_pose   = load_yolo(hand_pose_path,   "Pose",   device=DEVICE)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    current_date        = datetime.now().strftime("%Y-%m-%d")
    resolved_output_dir = os.path.join(out_base, current_date)
    for sub in ("NORMAL", "ANOMALY", "UNKNOWN"):
        Path(os.path.join(resolved_output_dir, sub)).mkdir(parents=True, exist_ok=True)

    cycles = process_video_cycles(
        video_path, resolved_output_dir, seg_net,
        yolo_socket, yolo_pose, print_summary, enable_debug=enable_debug)

    if cycles is not None:
        cycles.final_report()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Diagnostic Engine v50_MergedSingleVideo — v45 live-inference "
                    "engine + v49 nearest-pixel tube order + cycle logic + "
                    "3-way tracked identity lock + clean HAND state + "
                    "socket-adjacent mask ROI gate + [FIX-I38/I39] enforced GPU "
                    "execution with full GPU utilization (TF32 + FP16 + GPU-side "
                    "normalization) + [FIX-I41/I42/I43/I44] thick, continuous, "
                    "high-opacity mask overlay")
    ap.add_argument("--video",         default=DEFAULT_VIDEO, required=True)
    ap.add_argument("--model",         default=DEFAULT_MODEL)
    ap.add_argument("--out_base",      default=_OUT_BASE)
    ap.add_argument("--yolo",          default=DEFAULT_YOLO, required=True)
    ap.add_argument("--hand_yolo",     default=DEFAULT_POSE)
    ap.add_argument("--channels",      type=int,   default=None,
                    help="Force in_channels (3 or 4). Auto-detected if omitted.")
    ap.add_argument("--warmup",        type=int,   default=WARMUP_FRAMES)
    ap.add_argument("--verdict_thr",   type=float, default=VERDICT_THR)
    ap.add_argument("--seq_stable",    type=int,   default=MIN_SEQ_STABLE,
                    help="Frames the sequence must be stable before voting [FIX-I6]")
    ap.add_argument("--no_sharpening", action="store_true",
                    help="Disable tube boundary sharpening [FIX-I2]")
    ap.add_argument("--ema_alpha",     type=float, default=EMA_ALPHA,
                    help="EMA smoothing alpha (default 0.20) [FIX-I1]")
    ap.add_argument("--sharpen_temp",  type=float, default=TUBE_SHARPNESS_TEMP,
                    help="[FIX-I8] Boundary-sharpening temperature for classes 3/4. "
                         "Higher = softer split. Default 0.85.")
    ap.add_argument("--conf_thr_mid",  type=float, default=CLASS_CONF_THR[3],
                    help="[FIX-I8] Confidence threshold for class 3 (mid-tube/Blue). Default 0.22.")
    ap.add_argument("--conf_thr_end",  type=float, default=CLASS_CONF_THR[4],
                    help="[FIX-I8] Confidence threshold for class 4 (end-tube/Pink). Default 0.22.")
    ap.add_argument("--min_tube_px",   type=int,   default=MIN_TUBE_PX,
                    help="[FIX-I8] Minimum pixels for a tube class to be reported Present. Default 40.")
    ap.add_argument("--min_area_px",   type=int,   default=MIN_AREA_PX,
                    help="[FIX-I8] Minimum connected-component area kept in the mask. Default 40.")
    ap.add_argument("--search_radius", type=int,   default=NEAREST_SEARCH_RADIUS,
                    help="[FIX-I18/I27] Half-side (px) of the rectangular search region "
                         "around the socket centre for nearest-pixel tube order detection. "
                         "Default 350.")
    ap.add_argument("--mask_roi_radius", type=int, default=MASK_ROI_RADIUS,
                    help="[FIX-I30] Radius (px) around the socket centre OUTSIDE of which "
                         "tube-class pixels (2/3/4) are suppressed entirely — confines the "
                         "segmentation/overlay to the socket-adjacent stretch of each tube "
                         "instead of the whole coiled/looped tube. Default 160.")
    ap.add_argument("--mask_roi_shape",
                    choices=["circle", "square", "ellipse", "quad_ellipse", "polygon"],
                    default=MASK_ROI_SHAPE,
                    help="[FIX-I31/I32/I33/I34] Shape of the socket-adjacent keep-region. "
                         "'quad_ellipse' (default) uses 4 independent directional radii "
                         "(--mask_roi_up/down/left/right) — simplest way to match an "
                         "irregular, non-symmetric boundary without clicking a polygon. "
                         "'ellipse' uses a single --mask_roi_rx/ry pair. 'circle' hugs the "
                         "socket uniformly. 'square' reaches ~1.41x farther on the "
                         "diagonals. 'polygon' uses an exact traced boundary (see "
                         "--mask_roi_polygon_file / tools/pick_roi_polygon.py).")
    ap.add_argument("--mask_roi_up", type=int, default=MASK_ROI_RADIUS_UP,
                    help="[FIX-I34] Reach (px) above the socket centre for shape=quad_ellipse. "
                         "Default 330.")
    ap.add_argument("--mask_roi_down", type=int, default=MASK_ROI_RADIUS_DOWN,
                    help="[FIX-I34] Reach (px) below the socket centre for shape=quad_ellipse. "
                         "Default 120.")
    ap.add_argument("--mask_roi_left", type=int, default=MASK_ROI_RADIUS_LEFT,
                    help="[FIX-I34] Reach (px) left of the socket centre for shape=quad_ellipse. "
                         "Default 160.")
    ap.add_argument("--mask_roi_right", type=int, default=MASK_ROI_RADIUS_RIGHT,
                    help="[FIX-I34] Reach (px) right of the socket centre for shape=quad_ellipse. "
                         "Fixed-pixel fallback used only when auto-scale is off or no bbox "
                         "size is available yet. Default 220.")
    ap.add_argument("--no_roi_auto_scale", action="store_true",
                    help="[FIX-I35] Disable bbox-relative auto-scaling of the quad_ellipse "
                         "ROI and use the fixed --mask_roi_up/down/left/right pixel values "
                         "on every frame instead. By default (auto-scale ON) those 4 values "
                         "are only a fallback for frames with no socket bbox size yet.")
    ap.add_argument("--mask_roi_mult_up", type=float, default=MASK_ROI_MULT_UP,
                    help="[FIX-I35] Auto-scale multiplier: reach above socket = this x "
                         "socket bbox HEIGHT. Default 1.8.")
    ap.add_argument("--mask_roi_mult_down", type=float, default=MASK_ROI_MULT_DOWN,
                    help="[FIX-I35] Auto-scale multiplier: reach below socket = this x "
                         "socket bbox HEIGHT. Default 0.65.")
    ap.add_argument("--mask_roi_mult_left", type=float, default=MASK_ROI_MULT_LEFT,
                    help="[FIX-I35] Auto-scale multiplier: reach left of socket = this x "
                         "socket bbox WIDTH. Default 0.75.")
    ap.add_argument("--mask_roi_mult_right", type=float, default=MASK_ROI_MULT_RIGHT,
                    help="[FIX-I35] Auto-scale multiplier: reach right of socket = this x "
                         "socket bbox WIDTH. Default 1.05.")
    ap.add_argument("--no_mask_exclude", action="store_true",
                    help="[FIX-I36] Disable the hard exclusion (keep-out) zone entirely.")
    ap.add_argument("--no_exclude_auto_scale", action="store_true",
                    help="[FIX-I36] Disable bbox-relative auto-scaling of the exclusion "
                         "zone and use the fixed --mask_exclude_offset_x/y/rx/ry pixel "
                         "values on every frame instead.")
    ap.add_argument("--mask_exclude_mult_offset_x", type=float, default=MASK_EXCLUDE_OFFSET_MULT_X,
                    help="[FIX-I36] Exclusion zone centre X offset from socket = this x "
                         "socket bbox WIDTH (negative = left). Default -1.3.")
    ap.add_argument("--mask_exclude_mult_offset_y", type=float, default=MASK_EXCLUDE_OFFSET_MULT_Y,
                    help="[FIX-I36] Exclusion zone centre Y offset from socket = this x "
                         "socket bbox HEIGHT (negative = above). Default -0.9.")
    ap.add_argument("--mask_exclude_mult_rx", type=float, default=MASK_EXCLUDE_RADIUS_MULT_X,
                    help="[FIX-I36] Exclusion zone horizontal radius = this x socket bbox "
                         "WIDTH. Default 1.1.")
    ap.add_argument("--mask_exclude_mult_ry", type=float, default=MASK_EXCLUDE_RADIUS_MULT_Y,
                    help="[FIX-I36] Exclusion zone vertical radius = this x socket bbox "
                         "HEIGHT. Default 1.1.")
    ap.add_argument("--mask_exclude_offset_x", type=int, default=MASK_EXCLUDE_OFFSET_X,
                    help="[FIX-I36] Fixed-pixel fallback: exclusion zone centre X offset "
                         "from socket. Default -250.")
    ap.add_argument("--mask_exclude_offset_y", type=int, default=MASK_EXCLUDE_OFFSET_Y,
                    help="[FIX-I36] Fixed-pixel fallback: exclusion zone centre Y offset "
                         "from socket. Default -150.")
    ap.add_argument("--mask_exclude_rx", type=int, default=MASK_EXCLUDE_RADIUS_X,
                    help="[FIX-I36] Fixed-pixel fallback: exclusion zone horizontal radius. "
                         "Default 200.")
    ap.add_argument("--mask_exclude_ry", type=int, default=MASK_EXCLUDE_RADIUS_Y,
                    help="[FIX-I36] Fixed-pixel fallback: exclusion zone vertical radius. "
                         "Default 200.")
    ap.add_argument("--no_mask_freeze", action="store_true",
                    help="[FIX-I37] Disable the post-warmup mask freeze and go back to "
                         "running live inference every single frame (original v45/v49 "
                         "behaviour). By default the mask is frozen after warmup and only "
                         "re-inferred when the tube is detected to have actually moved.")
    ap.add_argument("--mask_freeze_motion_thr", type=float, default=MASK_FREEZE_MOTION_THR,
                    help="[FIX-I37] Per-pixel optical-flow magnitude (px/frame) considered "
                         "'moved' when checking whether to refresh the frozen mask. Lower = "
                         "more sensitive to small movements. Default 2.5.")
    ap.add_argument("--mask_freeze_motion_frac", type=float, default=MASK_FREEZE_MOTION_FRAC,
                    help="[FIX-I37] Fraction of the socket ROI that must exceed the motion "
                         "threshold before the mask is considered to have moved and gets "
                         "refreshed. Lower = triggers a refresh more easily. Default 0.04.")
    ap.add_argument("--mask_roi_polygon_file", type=str, default=None,
                    help="[FIX-I33] Path to a JSON file containing a list of [dx, dy] "
                         "offsets FROM THE SOCKET CENTRE defining the exact keep-region "
                         "boundary for shape=polygon. Generate this file by running "
                         "tools/pick_roi_polygon.py on a representative frame and clicking "
                         "the boundary. If omitted, falls back to the MASK_ROI_POLYGON "
                         "constant hard-coded near the top of this file.")
    ap.add_argument("--mask_roi_rx", type=int, default=MASK_ROI_RADIUS_X,
                    help="[FIX-I32] Horizontal radius (px) for shape=ellipse. Default 160.")
    ap.add_argument("--mask_roi_ry", type=int, default=MASK_ROI_RADIUS_Y,
                    help="[FIX-I32] Vertical radius (px) for shape=ellipse. Default 330.")
    ap.add_argument("--mask_roi_offset_x", type=int, default=MASK_ROI_OFFSET_X,
                    help="[FIX-I32] Shift the ROI centre left(-)/right(+) of the socket "
                         "centre, in px. Default 0.")
    ap.add_argument("--mask_roi_offset_y", type=int, default=MASK_ROI_OFFSET_Y,
                    help="[FIX-I32] Shift the ROI centre up(-)/down(+) of the socket "
                         "centre, in px. Default -120 (shifted up, toward the tube run).")
    ap.add_argument("--socket_grace",  type=int,   default=SOCKET_RESET_GRACE,
                    help="Consecutive 'no socket' frames required to close a cycle. Default 45.")
    ap.add_argument("--dilate_sz",     type=int,   default=MASK_DILATE_SZ,
                    help="[FIX-I21/I41/I42] Kernel size (px) used to widen tube-class masks "
                         "(2/3/4) so they render with visible width. Larger = thicker/more "
                         "filled-in overlay. Default 28 (raised from 16 for an even bolder, "
                         "thick-band look). Try 32-40 for an extremely bold overlay.")
    ap.add_argument("--close_sz",      type=int,   default=MASK_CLOSE_SZ,
                    help="[FIX-I43] Kernel size (px) used for the post-dilation morphological "
                         "CLOSE that bridges small gaps BETWEEN nearby same-class blobs so a "
                         "tube reads as one continuous stroke instead of several separate "
                         "fattened pieces. Should usually be >= --dilate_sz. Default 40. Set "
                         "to 0 to disable just the close pass while keeping dilation.")
    ap.add_argument("--overlay_alpha", type=float, default=OVERLAY_ALPHA,
                    help="[FIX-I44] Blend alpha for the tube-class overlay layer (0-1). "
                         "Higher = more solid/opaque painted color, lower = more see-through "
                         "tint. Default 0.90.")
    ap.add_argument("--no_dilate",     action="store_true",
                    help="[FIX-I21] Disable mask-widening dilation (and the FIX-I43 close "
                         "pass, which runs after it) and keep the raw (thinner) segmentation "
                         "mask width.")
    ap.add_argument("--identity_margin", type=float, default=TUBE_IDENTITY_MARGIN,
                    help="[FIX-I23] EMA margin a class must exceed the locked class by "
                         "to flip a pixel's tube identity. Default 0.08.")
    ap.add_argument("--no_identity_lock", action="store_true",
                    help="[FIX-I23] Disable the 3-way tube identity lock and use raw "
                         "per-frame argmax for classes 2/3/4 instead.")
    ap.add_argument("--lock_engage_mode", choices=["post_warmup", "immediate"],
                    default=LOCK_ENGAGE_MODE,
                    help="[FIX-I26] 'post_warmup' (default) only lets the identity lock "
                         "start overwriting pred once warmup has finished. 'immediate' "
                         "enforces it from frame 1 of the cycle. Default post_warmup.")
    ap.add_argument("--locked_conf_floor", type=float, default=LOCKED_PIXEL_MIN_CONF,
                    help="[FIX-I26] Relaxed confidence floor applied ONLY to pixels the "
                         "identity lock just overrode (instead of the full per-class "
                         "threshold), to stop the lock and the confidence filter from "
                         "fighting each other every frame. Default 0.08.")
    ap.add_argument("--frame_ms_warn", type=float, default=FRAME_MS_WARN_THRESHOLD,
                    help="[FIX-I28] Per-frame processing latency (ms) above which a "
                         "frame is logged as slow. Default 150.0.")
    ap.add_argument("--hsv_gate",      action="store_true")
    ap.add_argument("--print_summary", action="store_true")
    ap.add_argument("--debug",         action="store_true")
    ap.add_argument("--mask_debug",    action="store_true",
                    help="[FIX-I8] Print per-stage pixel-count trace for classes 2/3/4 on "
                         "every frame. Toggle live with 'm'.")
    ap.add_argument("--no_require_gpu", action="store_true",
                    help="[FIX-I38] By default the script REFUSES to run if CUDA is not "
                         "available (this is what previously let all 3 models silently run "
                         "on CPU while GPU 0 sat idle). Pass this flag to explicitly allow "
                         "CPU execution instead of raising an error.")
    ap.add_argument("--half",          dest="half", action="store_true", default=True,
                    help="[FIX-I39] Run the segmentation model and both YOLO models in "
                         "FP16 for full GPU throughput. This is the DEFAULT whenever a "
                         "CUDA device is active (ignored/forced-off on CPU).")
    ap.add_argument("--no_half",       dest="half", action="store_false",
                    help="[FIX-I39] Force FP32 everywhere instead of FP16 (useful for "
                         "debugging numerical differences, or a GPU with poor FP16 "
                         "throughput). Slower than --half on modern GPUs.")
    ap.add_argument("--gpu_warmup_hw", type=int, nargs=2, default=None,
                    metavar=("HEIGHT", "WIDTH"),
                    help="[FIX-I39] H W of the dummy frame used to warm up the YOLO "
                         "models' cudnn kernels before the video loop starts. Should "
                         "roughly match your real video resolution so cudnn.benchmark "
                         "autotunes the kernel shape that will actually be used. "
                         "Default: 720 1280.")
    args = ap.parse_args()

    WARMUP_FRAMES            = args.warmup
    VERDICT_THR              = args.verdict_thr
    MIN_SEQ_STABLE           = args.seq_stable
    TUBE_BOUNDARY_SHARPENING = not args.no_sharpening
    EMA_ALPHA                = args.ema_alpha
    TUBE_SHARPNESS_TEMP      = args.sharpen_temp
    CLASS_CONF_THR[3]        = args.conf_thr_mid
    CLASS_CONF_THR[4]        = args.conf_thr_end
    MIN_TUBE_PX              = args.min_tube_px
    MIN_AREA_PX              = args.min_area_px
    NEAREST_SEARCH_RADIUS    = args.search_radius
    MASK_ROI_RADIUS          = args.mask_roi_radius
    MASK_ROI_SHAPE           = args.mask_roi_shape
    MASK_ROI_RADIUS_X        = args.mask_roi_rx
    MASK_ROI_RADIUS_Y        = args.mask_roi_ry
    MASK_ROI_RADIUS_UP       = args.mask_roi_up
    MASK_ROI_RADIUS_DOWN     = args.mask_roi_down
    MASK_ROI_RADIUS_LEFT     = args.mask_roi_left
    MASK_ROI_RADIUS_RIGHT    = args.mask_roi_right
    MASK_ROI_AUTO_SCALE      = not args.no_roi_auto_scale
    MASK_ROI_MULT_UP         = args.mask_roi_mult_up
    MASK_ROI_MULT_DOWN       = args.mask_roi_mult_down
    MASK_ROI_MULT_LEFT       = args.mask_roi_mult_left
    MASK_ROI_MULT_RIGHT      = args.mask_roi_mult_right
    MASK_EXCLUDE_ENABLED        = not args.no_mask_exclude
    MASK_EXCLUDE_AUTO_SCALE      = not args.no_exclude_auto_scale
    MASK_EXCLUDE_OFFSET_MULT_X   = args.mask_exclude_mult_offset_x
    MASK_EXCLUDE_OFFSET_MULT_Y   = args.mask_exclude_mult_offset_y
    MASK_EXCLUDE_RADIUS_MULT_X   = args.mask_exclude_mult_rx
    MASK_EXCLUDE_RADIUS_MULT_Y   = args.mask_exclude_mult_ry
    MASK_EXCLUDE_OFFSET_X        = args.mask_exclude_offset_x
    MASK_EXCLUDE_OFFSET_Y        = args.mask_exclude_offset_y
    MASK_EXCLUDE_RADIUS_X        = args.mask_exclude_rx
    MASK_EXCLUDE_RADIUS_Y        = args.mask_exclude_ry
    MASK_FREEZE_ENABLED          = not args.no_mask_freeze
    MASK_FREEZE_MOTION_THR       = args.mask_freeze_motion_thr
    MASK_FREEZE_MOTION_FRAC      = args.mask_freeze_motion_frac
    MASK_ROI_OFFSET_X        = args.mask_roi_offset_x
    MASK_ROI_OFFSET_Y        = args.mask_roi_offset_y
    if args.mask_roi_polygon_file:
        with open(args.mask_roi_polygon_file, "r") as f:
            _pts = json.load(f)
        MASK_ROI_POLYGON = [(int(p[0]), int(p[1])) for p in _pts]
        print(f"[INFO] [FIX-I33] Loaded {len(MASK_ROI_POLYGON)} polygon points from "
              f"{args.mask_roi_polygon_file}")
    SOCKET_RESET_GRACE       = args.socket_grace
    MASK_DILATE_SZ            = args.dilate_sz
    MASK_CLOSE_SZ             = args.close_sz
    MASK_CLOSE_ENABLED        = args.close_sz > 0
    OVERLAY_ALPHA             = args.overlay_alpha
    MASK_DILATE_ENABLED       = not args.no_dilate
    TUBE_IDENTITY_MARGIN      = args.identity_margin
    IDENTITY_HYSTERESIS_ENABLED = not args.no_identity_lock
    LOCK_ENGAGE_MODE          = args.lock_engage_mode
    LOCKED_PIXEL_MIN_CONF     = args.locked_conf_floor
    FRAME_MS_WARN_THRESHOLD   = args.frame_ms_warn
    MASK_DEBUG                = args.mask_debug
    USE_HSV_GATE              = args.hsv_gate

    run_single_video(
        video_path        = args.video,
        seg_model_path    = args.model,
        out_base          = args.out_base,
        yolo_socket_path  = args.yolo,
        hand_pose_path    = args.hand_yolo,
        print_summary     = args.print_summary,
        enable_debug      = args.debug,
        forced_channels   = args.channels,
        require_gpu       = not args.no_require_gpu,
        use_half          = args.half,
        gpu_warmup_hw     = tuple(args.gpu_warmup_hw) if args.gpu_warmup_hw else None,
    )