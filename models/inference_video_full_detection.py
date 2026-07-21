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

Carried-over fixes from v45
----------------------------
  [FIX-I1] EMA alpha 0.20
  [FIX-I2] Tube boundary sharpening (temperature rescaling)
  [FIX-I3] Tighter confidence thresholds for classes 3 & 4
  [FIX-I5] Last-known socket centre cache
  [FIX-I6] Sequence stability gate
  [FIX-I8] Mid-tube mask dropout fix (loosened conf/area thresholds,
           OPEN skipped for classes 3/4, --mask_debug instrumentation)

Carried-over fixes from v49
----------------------------
  [FIX-I18] Nearest-Pixel Angular Gate — tube order is judged from the
            angle, at the socket, from the socket centre to each tube's
            NEAREST detected pixel (circular search region of radius
            NEAREST_SEARCH_RADIUS), instead of a shifted-ROI / far-distance
            weighted centroid.
  Cycle management — CycleManager opens a new "cycle" every time the socket
            is (re)acquired after being fully absent for SOCKET_RESET_GRACE
            frames, writes a per-cycle output video into NORMAL/ANOMALY/
            UNKNOWN, keeps running PASS/FAIL/UNKNOWN totals, and logs one
            Excel row per cycle.
  Production status bar — top-of-frame bar showing CYCLE #, STATUS,
            PASSED, FAILED, UNKNOWN counts.

NEW in this revision (v50b)
----------------------------
  [FIX-I23] 3-way tube identity lock — the per-pixel identity hysteresis
            introduced in FIX-I22 only arbitrated between classes 3 and 4.
            It is now generalized to all three tube classes (2 / 3 / 4), so
            a pixel that is locked in as "yellow" (class 2) near the socket
            junction cannot flip to pink/blue on ordinary per-frame noise
            either — the same sustained-margin rule now applies across all
            three classes instead of just two of them.
  [FIX-I24] Optical-flow lock tracking — previously the identity lock/EMA
            arrays were pinned to fixed pixel coordinates, so if the tube
            itself moved (vibration, being nudged, camera shake) the lock
            would lag behind the object instead of moving with it. Before
            merging each frame's evidence, the existing lock/EMA state is
            now warped forward with dense optical flow (Farneback) computed
            between the previous and current frame, so the "memory" of
            which pixel belongs to which tube class travels with the
            object instead of staying at a fixed screen location.
  [FIX-I25] Clean HAND state — while a hand is in the ROI, no segmentation
            overlay or tube-status/sequence information is shown or held
            over from the previous frame. The state machine now blanks out
            detections entirely during HAND so nothing stale or misleading
            is displayed while the operator's hand may be occluding or
            disturbing the tubes. (Socket bounding box, since it reflects
            YOLO socket presence rather than tube segmentation, is still
            drawn so the ROI/cycle context stays visible.)

State machine (unchanged)
  IDLE    → no socket detected
  WARMUP  → first WARMUP_FRAMES frames accumulating EMA
  HAND    → hand in ROI; inference paused, NO detections/segmentation shown
  INSPECT → live inference running, sequence not yet stable
  NORMAL  → live inference, stable OK sequence
  ANOMALY → live inference, stable ANOMALY sequence
  PARTIAL → live inference, not all tubes visible
"""

import os, sys, time, platform, argparse, math, shutil, zipfile, json, base64
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

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


# [WEB-INTEGRATION] The backend (InferenceEngine.build_environment) launches
# this script as a subprocess with HEADLESS=1 in its environment whenever it
# is driving the web UI, specifically so cv2.imshow() never pops a real
# window on the server. That env var was previously never read here, so
# SHOW_PREVIEW stayed hardcoded True and a window popped up regardless.
# HEADLESS=1 => no window. Anything else (unset, "0", running the script
# by hand) keeps the old default of showing the preview.
SHOW_PREVIEW = os.environ.get("HEADLESS", "0").strip().lower() not in ("1", "true", "yes")
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
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAY_ALPHA  = 0.65

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

# [FIX-I18, from v49] Circular search radius (px) around the socket centre
# used to find each tube's nearest-to-socket pixel for angular order
# detection — replaces the old shifted-ROI / far-distance-weighted centroid.
NEAREST_SEARCH_RADIUS = 350

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
MASK_DILATE_ENABLED = True
MASK_DILATE_SZ      = 9
MASK_DILATE_CLASSES = (2, 3, 4)

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
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        for key, tensor in sd.items():
            if tensor.ndim == 4 and key.endswith(".weight"):
                ic = tensor.shape[1]
                if ic in (3, 4):
                    print(f"[AUTO] Detected in_channels={ic} from key: {key}")
                    return ic
        print("[AUTO] Could not detect in_channels - defaulting to 3")
        return 3
    except Exception as e:
        print(f"[AUTO] Channel detection failed ({e}) - defaulting to 3")
        return 3


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_yolo(path, label):
    if not path:
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        print(f"[OK ] YOLO {label}: {path}")
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
    res = next(model(frame, stream=True, verbose=False))
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
    res = next(pose_model(frame_bgr, stream=True, verbose=False))
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
_tf_pipeline = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE[0], interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(min_height=IMG_SIZE[0], min_width=IMG_SIZE[1],
                  border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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


@torch.no_grad()
def raw_infer(seg_model, frame_bgr, socket_centre=None):
    """
    Single forward pass. Appends a distance-from-socket radial channel when
    the model was trained with 4 input channels.

    [FIX-I5] socket_centre is cached by the caller so a YOLO miss on one
    frame does not silently fall back to image centre mid-sequence.
    """
    oh, ow = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t      = _tf_pipeline(image=rgb)["image"]

    if USE_RADIAL_CHANNEL:
        scale = IMG_SIZE[0] / max(oh, ow)
        new_h = int(oh * scale)
        new_w = int(ow * scale)
        pad_y = (IMG_SIZE[0] - new_h) // 2
        pad_x = (IMG_SIZE[1] - new_w) // 2
        scx   = socket_centre[0] * scale + pad_x if socket_centre else IMG_SIZE[1] / 2.0
        scy   = socket_centre[1] * scale + pad_y if socket_centre else IMG_SIZE[0] / 2.0
        rad   = make_radial_channel_np(IMG_SIZE[0], IMG_SIZE[1], cx=scx, cy=scy)
        t     = torch.cat([t, torch.from_numpy(rad).unsqueeze(0)], dim=0)

    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        logits = seg_model(t.unsqueeze(0).to(DEVICE))

    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    scale  = IMG_SIZE[0] / max(oh, ow)
    new_h  = int(oh * scale)
    new_w  = int(ow * scale)
    pad_y  = (IMG_SIZE[0] - new_h) // 2
    pad_x  = (IMG_SIZE[1] - new_w) // 2
    crop   = probs[:, pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    return np.stack([
        cv2.resize(crop[c].astype(np.float32), (ow, oh),
                   interpolation=cv2.INTER_LINEAR)
        for c in range(NUM_CLASSES)
    ])


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
        """Call after changing MASK_DILATE_SZ at runtime/CLI to rebuild the kernel."""
        self._dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_DILATE_SZ, MASK_DILATE_SZ))

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
        if self._prev_gray_track is None or self.identity_lock is None:
            self._prev_gray_track = gray
            return
        if gray.shape != self._prev_gray_track.shape:
            # Defensive: frame size changed mid-run — drop tracking state
            # rather than remap onto mismatched dimensions.
            self._prev_gray_track = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray_track, gray, None, pyr_scale=0.5, levels=2,
            winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

        h, w = gray.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)

        self.identity_lock = cv2.remap(
            self.identity_lock, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        for i in range(self.identity_ema.shape[0]):
            self.identity_ema[i] = cv2.remap(
                self.identity_ema[i], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        self._prev_gray_track = gray

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

        self.identity_ema[:, present] = (
            IDENTITY_EMA_ALPHA * class_probs[:, present]
            + (1.0 - IDENTITY_EMA_ALPHA) * self.identity_ema[:, present]
        )

        # tube no longer present here at all -> forget the lock
        self.identity_lock[~present] = 0

        # index (0/1/2) of the class currently winning the EMA, and its value
        best_idx = np.argmax(self.identity_ema, axis=0)
        best_val = np.take_along_axis(
            self.identity_ema, best_idx[None, :, :], axis=0)[0]

        # index (0/1/2) corresponding to the CURRENTLY locked class (or 0
        # as a placeholder where nothing is locked yet — masked out below)
        lock_idx = np.clip(self.identity_lock.astype(np.int32) - 2, 0, 2)
        cur_val  = np.take_along_axis(
            self.identity_ema, lock_idx[None, :, :], axis=0)[0]

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
        self.ema_probs = (
            probs_raw.copy() if self.ema_probs is None
            else EMA_ALPHA * probs_raw + (1 - EMA_ALPHA) * self.ema_probs
        )

        probs = self.ema_probs.copy()

        if MASK_DEBUG:
            ema_argmax = probs.argmax(axis=0)
            counts_ema = {ci: int((ema_argmax == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_ema       px={counts_ema}")

        # [FIX-I2] Sharpen Mid/End boundary via temperature rescaling.
        # Total (3+4) probability mass is conserved; only the 3-vs-4 ratio changes.
        if TUBE_BOUNDARY_SHARPENING:
            tube_stack  = np.stack([probs[3], probs[4]], axis=0)
            tube_stack -= tube_stack.max(axis=0, keepdims=True)
            tube_stack /= TUBE_SHARPNESS_TEMP
            exp_t        = np.exp(tube_stack)
            tube_softmax = exp_t / (exp_t.sum(axis=0, keepdims=True) + 1e-7)
            tube_mass    = probs[3] + probs[4]
            probs[3]     = tube_softmax[0] * tube_mass
            probs[4]     = tube_softmax[1] * tube_mass

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
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_tube_order(pred_map, socket_bbox=None, debug=False):
    """
    Determine cyclic angular order of tube classes around the socket centre.

    [FIX-I18] Nearest-Pixel Angular Gate:
    For each tube class, search a circular region of radius
    NEAREST_SEARCH_RADIUS centred on the socket, and take the SINGLE pixel
    of that class closest to the socket centre (i.e. the point where the
    tube first meets/enters the socket housing). The bearing of that
    nearest pixel from the socket centre is the tube's angle. Order is
    then just the cyclic sort of those three bearings. This replaces the
    old shifted-ROI + far-distance-weighted-centroid approach: no upward
    Y shift, no weighting towards pixels far from the socket.
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

    # ── CIRCULAR SEARCH REGION AROUND THE SOCKET CENTRE ───────────────────────
    rx1 = max(0, int(scx - R))
    ry1 = max(0, int(scy - R))
    rx2 = min(W - 1, int(scx + R))
    ry2 = min(H - 1, int(scy + R))

    circle_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(circle_mask, (int(scx), int(scy)), R, 1, -1)
    pred_roi = pred_map * circle_mask

    status    = {2: "Absent", 3: "Absent", 4: "Absent"}
    angles    = {}
    anchors   = {}
    nearest_d = {}

    if MASK_DEBUG:
        tube_px_counts = {ci: int((pred_roi == ci).sum()) for ci in (2, 3, 4)}
        print(f"[MASKDBG] evaluate_tube_order (nearest-pixel) ROI px counts={tube_px_counts}  "
              f"MIN_TUBE_PX={MIN_TUBE_PX}  radius={R}")

    for ci in (2, 3, 4):
        ty, tx = np.where(pred_roi == ci)
        if len(tx) < MIN_TUBE_PX:
            if MASK_DEBUG and len(tx) > 0:
                print(f"[MASKDBG] class {ci} has {len(tx)}px in search circle but "
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
        final_name = f"{self.video_stem}_cycle{self.cycle_no:03d}_{verdict}.mp4"
        final_path = os.path.join(dest_dir, final_name)
        if Path(final_path).exists():
            Path(final_path).unlink()
        shutil.move(self.temp_path, final_path)

        if verdict == "NORMAL":
            self.passed += 1
        elif verdict == "ANOMALY":
            self.failed += 1
        else:
            self.unknown += 1

        duration_s = time.time() - (self.start_time or time.time())
        print(f"[CYCLE END]    #{self.cycle_no:03d}  ->  {verdict:<8}  "
              f"({duration_s:.1f}s)  ->  {final_path}")
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
        print("\n" + "#" * W)
        print("  FINAL CYCLE REPORT")
        print("#" * W)
        print(f"  TOTAL CYCLES      : {self.total_cycles}")
        print(f"  PASSED (NORMAL)   : {self.passed}")
        print(f"  FAILED (ANOMALY)  : {self.failed}")
        print(f"  UNKNOWN           : {self.unknown}")
        print("#" * W + "\n")


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
# ══════════════════════════════════════════════════════════════════════════════
_TUBE_BGR  = {2: (0, 255, 255), 3: (255, 0, 0), 4: (255, 0, 255)}
_TUBE_NAME = {2: "Yel(2)", 3: "Blu(3)", 4: "Pnk(4)"}


def draw_debug_overlay(frame, dbg):
    if not dbg:
        return frame
    out = frame.copy()
    scx = int(dbg.get("scx", 0));  scy = int(dbg.get("scy", 0))
    radius = int(dbg.get("radius", NEAREST_SEARCH_RADIUS))
    # [FIX-I18] draw the circular search region (was a shifted rectangle)
    cv2.circle(out, (scx, scy), radius, (0, 255, 0), 1, cv2.LINE_AA)
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
             seq_stable_ctr=0, cycle_no=0):
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
    _put(out, f"{frame_idx:06d}", vx, ly, FS_MD, (200, 200, 200), TK1);  ly += ROW + 10

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
             lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1);  ly += ROW

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
                         enable_debug=False, render_mode="opencv", original_name="output",
                         on_message=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}");  return None

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {Path(video_path).name} - {src_w}x{src_h} @ {fps_src:.1f} fps")

    video_stem = Path(original_name).stem
    cycles     = CycleManager(resolved_output_dir, video_stem, fps_src, (src_w, src_h))

    seg_engine   = SegmentationEngine(seg_net)
    anomaly_gate = AnomalyConfirmGate(N_ANOMALY_CONFIRM)
    latch_gate   = ResultLatchGate(LATCH_FRAMES)
    vote_counter = VoteCounter(VERDICT_THR)
    seq_gate     = SequenceStabilityGate()

    WIN = f"v50_Merged | {os.path.basename(video_path)} | Q=quit D=debug F=fs M=maskdbg"
    show_preview_active = SHOW_PREVIEW and render_mode != "frontend"

    if show_preview_active:
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

    # [WEB-INTEGRATION] last state we emitted a [STATUS] line for
    last_emitted_state = None

    # [FIX-I5] Last-known socket centre cache
    last_socket_centre = None

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
            if show_preview_active:
                cv2.imshow(WIN, card);  cv2.waitKey(int(VERDICT_HOLD_SEC * 1000))

        display_verdict = _ORDER_DISPLAY.get(
            peak_order if peak_order != "N/A" else final_verdict, "N/A")
        seq_display = " > ".join(TUBE_SHORT[c] for c in peak_seq) if peak_seq else "-"

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
        # frame_idx += 1
        frame_idx += 1
        if frame_idx % 15 == 0:
            if on_message:
                on_message(f"[PROGRESS] frame={frame_idx}")
            else:
                print(f"[PROGRESS] frame={frame_idx}", flush=True)
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
            vis = draw_socket_box(vis, sock_hit)

        else:
            # ── LIVE INFERENCE every frame ────────────────────────────────
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
            pred_map = seg_engine.infer(
                frame, socket_centre=last_socket_centre,
                apply_identity_lock=lock_engage)

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
                        print(f"[WARN] f{frame_idx:05d}: warmup blank -> retry "
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

        fps_ema = 0.88 * fps_ema + 0.12 / max(time.perf_counter() - t0, 1e-6)

        # Branch frames: 'vis_stream' for the frontend, 'vis' for local OpenCV & saving
        vis_stream = vis.copy() if render_mode == "frontend" else vis

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
        )

        # [WEB-INTEGRATION] Structured status for the web UI. Mirrors exactly
        # what draw_hud()/draw_production_status_bar() just drew onto the
        # frame, but as data instead of pixels, so the frontend can render
        # its own badges/panels instead of relying on the annotated video.
        # Throttled to every 5 frames, plus immediately on any state change
        # (e.g. NORMAL -> ANOMALY) so the UI doesn't lag a full 5 frames
        # behind a verdict flip.
        if frame_idx % 5 == 0 or state != last_emitted_state:
            status_payload = {
                "frame":            frame_idx,
                "fps":              round(float(fps_ema), 1),
                "state":            state,
                "cycle_no":         cycles.cycle_no,
                "passed":           cycles.passed,
                "failed":           cycles.failed,
                "unknown":          cycles.unknown,
                "hand_in_roi":      bool(hand_in_roi),
                "socket_present":   bool(sock_hit and sock_hit["class"] == CLS_SOCKET),
                "tubes": {
                    TUBE_SHORT[ci]: status_dict.get(ci, "Absent") for ci in (2, 3, 4)
                },
                "sequence":         [TUBE_SHORT[c] for c in detected_seq] if detected_seq else [],
                "order_status":     order_status,
                "anomaly_counter":  anomaly_gate._count,
                "warmup_frame":     warmup_frame_count,
                "warmup_retry":     warmup_retry,
                "infer_frames":     infer_frames,
                "seq_stable":       seq_gate._stable_ct,
                "seq_stable_min":   MIN_SEQ_STABLE,
                "ok_votes":         vote_counter.normal_votes,
                "anomaly_votes":    vote_counter.anomaly_votes,
            }
            if on_message:
                on_message(f"[STATUS] {json.dumps(status_payload)}")
            else:
                print(f"[STATUS] {json.dumps(status_payload)}", flush=True)
            last_emitted_state = state

        # Emit frame as base64 over stdout for the web UI
        vis_stream_out = vis_stream
        if render_mode == "frontend":
            h, w = vis_stream.shape[:2]
            if w > 1280:
                scale = 1280.0 / w
                vis_stream_out = cv2.resize(vis_stream, (1280, int(h * scale)))
            success, buffer = cv2.imencode('.jpg', vis_stream_out, [cv2.IMWRITE_JPEG_QUALITY, 65])
        else:
            success, buffer = cv2.imencode('.jpg', vis_stream_out, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if success:
            b64 = base64.b64encode(buffer).decode('utf-8')
            if on_message:
                on_message(f"[FRAME] {b64}")
            else:
                print(f"[FRAME] {b64}", flush=True)

        cycles.write(vis);  last_vis = vis

        if show_preview_active:
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
    if show_preview_active:
        cv2.destroyWindow(WIN)

    return cycles


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL LOGGING  (v49 — cycle-based)
# ══════════════════════════════════════════════════════════════════════════════
_EXCEL_COLUMNS = [
    ("sr_no",            "Sr. No."),
    ("cycle_no",         "Cycle No."),
    ("timestamp",        "Timestamp"),
    ("filename",         "Video File"),
    ("final_verdict",    "Final Verdict"),
    ("output_folder",    "Output Folder"),
    ("socket",           "Socket"),
    ("tube_blue",        "tube_blue (Yellow)"),
    ("trans_mid_tube",   "trans_mid_tube (Blue)"),
    ("trans_end_tube",   "trans_end_tube (Pink)"),
    ("detected_sequence","Detected Sequence"),
    ("tube_order_result","Tube Order Result"),
    ("ok_votes",         "OK Votes"),
    ("anomaly_votes",    "Anomaly Votes"),
    ("anomaly_ratio",    "Anomaly Ratio"),
    ("warmup_frames",    "Warmup Frames"),
    ("infer_frames",     "Live Infer Frames"),
    ("total_frames",     "Total Frames"),
    ("avg_fps",          "Avg FPS"),
    ("ema_alpha",        "EMA Alpha"),
    ("sharpening",       "Boundary Sharpening"),
    ("seq_stable_min",   "Seq Stable Min"),
    ("channels",         "Channels"),
    ("radial_channel",   "Radial Channel"),
    ("output_path",      "Output Path"),
    ("debug_mode",       "Debug Mode"),
]
_VERDICT_FILLS  = {"NORMAL": "C6EFCE", "ANOMALY": "FFC7CE",
                   "PARTIAL": "FFEB9C", "N/A": "F2F2F2", "UNKNOWN": "FFEB9C"}
_PRESENCE_FILLS = {"Present": "C6EFCE", "Absent": "FFC7CE"}


def append_to_excel(run_metrics, excel_dir):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[WARN] openpyxl not installed - skipping Excel.");  return

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
        if key == "sr_no":
            row_vals.append(next_sr)
        elif key == "avg_fps":
            row_vals.append(round(run_metrics.get(key, 0.0), 2))
        elif key in ("ok_votes", "anomaly_votes", "infer_frames", "cycle_no",
                     "warmup_frames", "total_frames", "channels", "seq_stable_min"):
            row_vals.append(int(run_metrics.get(key, 0)))
        elif key == "anomaly_ratio":
            row_vals.append(round(run_metrics.get(key, 0.0), 4))
        elif key == "ema_alpha":
            row_vals.append(round(run_metrics.get(key, EMA_ALPHA), 3))
        else:
            row_vals.append(run_metrics.get(key, "N/A"))
    ws.append(row_vals)

    cr   = ws.max_row
    thin = Side(style="thin", color="BFBFBF")
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
    verdict_keys  = {"tube_order_result", "final_verdict", "output_folder"}
    presence_keys = {"socket", "tube_blue", "trans_mid_tube", "trans_end_tube"}
    center_keys   = {"sr_no", "cycle_no", "total_frames", "ok_votes", "anomaly_votes",
                     "infer_frames", "warmup_frames", "avg_fps", "anomaly_ratio",
                     "socket", "tube_blue", "trans_mid_tube", "trans_end_tube",
                     "detected_sequence", "tube_order_result", "final_verdict",
                     "output_folder", "debug_mode", "sharpening", "radial_channel",
                     "ema_alpha", "channels", "seq_stable_min"}

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
        elif key in presence_keys:
            hx        = _PRESENCE_FILLS.get(val, "F2F2F2")
            cell.fill = PatternFill(start_color=hx, end_color=hx, fill_type="solid")

    for ci in range(1, len(_EXCEL_COLUMNS) + 1):
        cl = get_column_letter(ci)
        mx = max(len(str(ws.cell(row=r, column=ci).value or ""))
                 for r in range(1, ws.max_row + 1))
        ws.column_dimensions[cl].width = max(mx + 4, 14)

    for attempt in range(1, MAX_EXCEL_RETRIES + 1):
        try:
            wb.save(excel_path)
            print(f"[EXCEL] Row #{next_sr} (cycle #{run_metrics.get('cycle_no','-')}) -> {excel_path}")
            return
        except PermissionError:
            if attempt == MAX_EXCEL_RETRIES:
                raise
            print(f"[WARN] Excel locked, retry {attempt}/{MAX_EXCEL_RETRIES} in 5s...")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
def run_single_video(video_path, seg_model_path, out_base,
                     yolo_socket_path, hand_pose_path, print_summary,
                     enable_debug=False, forced_channels=None,
                     render_mode="opencv", original_name="output",
                     models=None, on_message=None):

    print(f"[INFO] Device              : {DEVICE}")
    print(f"[INFO] Mode                : SINGLE VIDEO / MULTI-CYCLE (live inference, merged v45+v49)")
    print(f"[INFO] Input video         : {video_path}")
    print(f"[INFO] HSV gate            : {'ON' if USE_HSV_GATE else 'OFF'}")
    print(f"[INFO] Warmup              : {WARMUP_FRAMES} frames x up to {MAX_WARMUP_RETRIES} retries")
    print(f"[INFO] [FIX-I1] EMA alpha  : {EMA_ALPHA}")
    print(f"[INFO] [FIX-I2/I8] Sharpen tau : {TUBE_SHARPNESS_TEMP}  enabled={TUBE_BOUNDARY_SHARPENING}")
    print(f"[INFO] [FIX-I3/I8] Conf thr : {CLASS_CONF_THR}")
    print(f"[INFO] [FIX-I6] Seq stable : {MIN_SEQ_STABLE} frames")
    print(f"[INFO] [FIX-I8] MIN_TUBE_PX: {MIN_TUBE_PX}  MIN_AREA_PX: {MIN_AREA_PX}")
    print(f"[INFO] [FIX-I18] Nearest-pixel angular gate, search radius: {NEAREST_SEARCH_RADIUS}px")
    print(f"[INFO] [FIX-I21] Mask dilation (width): enabled={MASK_DILATE_ENABLED}  "
          f"size={MASK_DILATE_SZ}px  classes={MASK_DILATE_CLASSES}")
    print(f"[INFO] [FIX-I23/I24] 3-way tube identity lock with optical-flow tracking: "
          f"enabled={IDENTITY_HYSTERESIS_ENABLED}  margin={TUBE_IDENTITY_MARGIN}  "
          f"ema_alpha={IDENTITY_EMA_ALPHA}")
    print(f"[INFO] [FIX-I25] Clean HAND state (no detections shown while hand in ROI): enabled")
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

    if models:
        seg_net, yolo_socket, yolo_pose = models
        print("[INFO] Using pre-loaded models from ModelManager.")
    else:
        seg_net = smp.UnetPlusPlus(
            encoder_name="tu-hrnet_w18",
            encoder_weights=None,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        ).to(DEVICE)
        ckpt = torch.load(seg_model_path, map_location=DEVICE, weights_only=False)
        seg_net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        seg_net.eval()
        with torch.no_grad():
            seg_net(torch.zeros(1, IN_CHANNELS, *IMG_SIZE, device=DEVICE))
        print("[INFO] GPU warmup done.")

        yolo_socket = load_yolo(yolo_socket_path, "Socket")
        yolo_pose   = load_yolo(hand_pose_path,   "Pose")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    resolved_output_dir = out_base
    for sub in ("NORMAL", "ANOMALY", "UNKNOWN"):
        Path(os.path.join(resolved_output_dir, sub)).mkdir(parents=True, exist_ok=True)

    cycles = process_video_cycles(
        video_path, resolved_output_dir, seg_net,
        yolo_socket, yolo_pose, print_summary, enable_debug=enable_debug,
        render_mode=render_mode, original_name=original_name, on_message=on_message)

    if cycles is not None:
        cycles.final_report()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Diagnostic Engine v50_MergedSingleVideo — v45 live-inference "
                    "engine + v49 nearest-pixel tube order + cycle logic + "
                    "3-way tracked identity lock + clean HAND state")
    ap.add_argument("--video",         default=DEFAULT_VIDEO, required=True)
    ap.add_argument("--original_name", default="output",      help="Original filename stem for outputs")
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
                    help="[FIX-I18] Circular search radius (px) around the socket centre "
                         "for nearest-pixel tube order detection. Default 350.")
    ap.add_argument("--socket_grace",  type=int,   default=SOCKET_RESET_GRACE,
                    help="Consecutive 'no socket' frames required to close a cycle. Default 45.")
    ap.add_argument("--dilate_sz",     type=int,   default=MASK_DILATE_SZ,
                    help="[FIX-I21] Kernel size (px) used to widen tube-class masks (2/3/4) "
                         "so they render with visible width. Default 9.")
    ap.add_argument("--no_dilate",     action="store_true",
                    help="[FIX-I21] Disable mask-widening dilation and keep the raw "
                         "(thinner) segmentation mask width.")
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
    ap.add_argument("--hsv_gate",      action="store_true")
    ap.add_argument("--print_summary", action="store_true")
    ap.add_argument("--debug",         action="store_true")
    ap.add_argument("--mask_debug",    action="store_true",
                    help="[FIX-I8] Print per-stage pixel-count trace for classes 2/3/4 on "
                         "every frame. Toggle live with 'm'.")
    ap.add_argument("--render_mode", choices=["opencv", "frontend"], default="opencv",
                    help="Rendering mode: opencv (full HUD) or frontend (AI overlays only)")
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
    SOCKET_RESET_GRACE       = args.socket_grace
    MASK_DILATE_SZ            = args.dilate_sz
    MASK_DILATE_ENABLED       = not args.no_dilate
    TUBE_IDENTITY_MARGIN      = args.identity_margin
    IDENTITY_HYSTERESIS_ENABLED = not args.no_identity_lock
    LOCK_ENGAGE_MODE          = args.lock_engage_mode
    LOCKED_PIXEL_MIN_CONF     = args.locked_conf_floor
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
        render_mode       = args.render_mode,
        original_name     = args.original_name,
    )