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

[PATCH FIX-I54] Mask-bleeding fix applied: _dilate_multiclass_protected and
_close_multiclass_protected now resolve any pixel that more than one tube
class grows into by nearest-original-pixel (Voronoi) ownership instead of
fixed loop order, so one tube's color can never bleed into a neighbouring
tube's rightful territory. See the FIX-I54 docstrings on those two
functions further down for the full explanation.

[PATCH FIX-I55] Excel log now includes a running Sr No. column and a
Timestamp column, in addition to the existing cycle_no/filename/status/
output_path columns.
"""

import os, sys, time, platform, argparse, math, shutil, zipfile, json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# [FIX-I52] Make sure OpenCV's own internal thread pool (used by resize,
# morphology, optical flow, etc.) isn't accidentally pinned to 1 thread by
# an inherited OMP/OPENCV env var in some deployment environments.
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 4))
except Exception:
    pass

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
OVERLAY_ALPHA  = 0.90
SOCKET_BOX_FILL_ALPHA = 0.28
DEVICE = "cpu"
USE_HALF = False
GPU_WARMUP_HW = (720, 1280)
_NORM_MEAN = None
_NORM_STD  = None
EMA_ALPHA = 0.20
TUBE_BOUNDARY_SHARPENING = True
TUBE_SHARPNESS_TEMP      = 0.85

CLASS_CONF_THR = {
    1: 0.35,   # device
    2: 0.40,   # tube_blue
    3: 0.45,   # trans_mid_tube
    4: 0.38,   # trans_end_tube
}

MIN_SEQ_STABLE = 3
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

NEAREST_SEARCH_RADIUS = 350

MASK_ROI_RADIUS  = 160
MASK_ROI_CLASSES = (2, 3, 4)
MASK_ROI_SHAPE   = "quad_ellipse"
MASK_ROI_RADIUS_X = 160
MASK_ROI_RADIUS_Y = 330
MASK_ROI_OFFSET_X = 0
MASK_ROI_OFFSET_Y = -120

MASK_ROI_RADIUS_UP    = 330
MASK_ROI_RADIUS_DOWN  = 120
MASK_ROI_RADIUS_LEFT  = 160
MASK_ROI_RADIUS_RIGHT = 220

MASK_ROI_AUTO_SCALE = True
MASK_ROI_MULT_UP    = 1.3
MASK_ROI_MULT_DOWN   = 0.65
MASK_ROI_MULT_LEFT   = 0.75
MASK_ROI_MULT_RIGHT  = 1.05

MASK_EXCLUDE_ENABLED    = True
MASK_EXCLUDE_CLASSES    = (2, 3, 4)
MASK_EXCLUDE_AUTO_SCALE = True
MASK_EXCLUDE_OFFSET_MULT_X = -1.3
MASK_EXCLUDE_OFFSET_MULT_Y = -0.9
MASK_EXCLUDE_RADIUS_MULT_X = 1.1
MASK_EXCLUDE_RADIUS_MULT_Y = 1.1
MASK_EXCLUDE_OFFSET_X = -250
MASK_EXCLUDE_OFFSET_Y = -150
MASK_EXCLUDE_RADIUS_X = 200
MASK_EXCLUDE_RADIUS_Y = 200

MASK_FREEZE_ENABLED     = True
MASK_FREEZE_MOTION_THR  = 2.5
MASK_FREEZE_MOTION_FRAC = 0.04

OPT_FLOW_DOWNSCALE = 0.5

MASK_ROI_POLYGON = None

MIN_TUBE_PX     = 40
N_ANOMALY_CONFIRM = 4
LATCH_FRAMES    = 10
MIN_AREA_PX     = 120
MORPH_KERNEL_SZ = 5
PERSIST_FRAMES  = 1

MASK_DILATE_ENABLED = True
MASK_DILATE_SZ      = 6
MASK_DILATE_CLASSES = (2, 3, 4)

MASK_CLOSE_ENABLED = True
MASK_CLOSE_SZ       = 12

IDENTITY_HYSTERESIS_ENABLED = True
IDENTITY_EMA_ALPHA          = 0.05
TUBE_IDENTITY_MARGIN        = 0.12
TUBE_PRESENT_THR            = 0.15

LOCK_ENGAGE_MODE = "post_warmup"

LOCKED_PIXEL_MIN_CONF = 0.08

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

YOLO_SOCKET_CONF   = 0.35
YOLO_POSE_CONF     = 0.40
CLS_NO_SOCKET      = 0
CLS_SOCKET         = 1
ROI_PAD_X          = 180
ROI_PAD_Y          = 160
OF_MOTION_THR      = 3.5
OF_MOTION_FRAC     = 0.06
SOCKET_RESET_GRACE = 45
_prev_gray_ref     = [None]
_ROI_COL_AMBER     = (0, 165, 255)

FRAME_MS_WARN_THRESHOLD = 150.0
FRAME_MS_EMA_ALPHA      = 0.12

PERF_ROI_ENABLED = True
PERF_ROI_PAD     = 550

_YOLO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo")

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
#  [FIX-I49] SHARED DOWNSAMPLED-OPTICAL-FLOW HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _farneback_flow_downscaled(img_ref, img_now, downscale=None):
    downscale = OPT_FLOW_DOWNSCALE if downscale is None else downscale
    h, w = img_now.shape[:2]

    if downscale >= 0.999:
        return cv2.calcOpticalFlowFarneback(
            img_ref, img_now, None, pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

    sw = max(2, int(round(w * downscale)))
    sh = max(2, int(round(h * downscale)))
    ref_small = cv2.resize(img_ref, (sw, sh), interpolation=cv2.INTER_AREA)
    now_small = cv2.resize(img_now, (sw, sh), interpolation=cv2.INTER_AREA)

    flow_small = cv2.calcOpticalFlowFarneback(
        ref_small, now_small, None, pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= (w / sw)
    flow[..., 1] *= (h / sh)
    return flow


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I40] GPU SANITY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def _gpu_sanity_benchmark(device: str):
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
#  [FIX-I38/I39] GPU / DEVICE RESOLUTION + DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def resolve_device(require_gpu: bool = True) -> str:
    global _NORM_MEAN, _NORM_STD

    print("\n" + "═" * 78)
    print("  [FIX-I38] GPU / DEVICE DIAGNOSTICS")
    print("═" * 78)
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
        print("═" * 78 + "\n")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    print(f"  [FIX-I38] cudnn.benchmark      : True")
    print(f"  [FIX-I39] TF32 matmul/cudnn    : True")
    print(f"  RESOLVED DEVICE       : {device}")

    _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _NORM_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    _gpu_sanity_benchmark(device)
    print("═" * 78 + "\n")

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
    if not path:
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(path)
        if device is not None:
            m.to(device)
            try:
                wh, ww = GPU_WARMUP_HW
                dummy = np.zeros((wh, ww, 3), dtype=np.uint8)
                m.predict(dummy, device=device, half=USE_HALF, verbose=False)
            except Exception as warm_e:
                print(f"[WARN] YOLO {label} warmup inference failed: {warm_e}")
        task = getattr(m, "task", "unknown")
        print(f"[OK ] YOLO {label}: {path}  half={USE_HALF}  task={task}")
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
    res = model(frame, verbose=False, device=DEVICE, half=USE_HALF)[0]
    best, best_c = None, -1.0

    if getattr(res, "boxes", None) is not None:
        for box in res.boxes:
            c = float(box.conf[0])
            if c >= conf_thr and c > best_c:
                best_c = c
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                best = {"bbox": (x1, y1, x2, y2),
                        "class": int(box.cls[0]), "conf": c,
                        "obb_points": None}

    elif getattr(res, "obb", None) is not None and len(res.obb) > 0:
        obb = res.obb
        for i in range(len(obb)):
            c = float(obb.conf[i])
            if c >= conf_thr and c > best_c:
                best_c = c
                pts = obb.xyxyxyxy[i].tolist()
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                best = {"bbox": (x1, y1, x2, y2),
                        "class": int(obb.cls[i]), "conf": c,
                        "obb_points": [(int(p[0]), int(p[1])) for p in pts]}

    return best


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I37] MASK-FREEZE MOTION CHECK
# ══════════════════════════════════════════════════════════════════════════════
def mask_has_moved(gray_now, gray_ref, thr=MASK_FREEZE_MOTION_THR,
                    frac=MASK_FREEZE_MOTION_FRAC):
    if gray_now.shape != gray_ref.shape or gray_now.size == 0:
        return True
    flow = _farneback_flow_downscaled(gray_ref, gray_now)
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
            flow = _farneback_flow_downscaled(rp, rc)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if float((mag > OF_MOTION_THR).sum()) / roi_area >= OF_MOTION_FRAC:
                motion = True
    _prev_gray_ref[0] = gray
    if motion:
        return True
    if pose_model is None:
        return False
    res = pose_model(frame_bgr, verbose=False, device=DEVICE, half=USE_HALF)[0]
    if hasattr(res, "keypoints") and res.keypoints is not None:
        for kpts in res.keypoints.data:
            if kpts.shape[0] < 11:
                continue
            for idx in (5, 6, 7, 8, 9, 10):
                kx, ky, kc = kpts[idx].tolist()
                if (kc >= pose_conf_thr and
                        rx1 <= int(kx) <= rx2 and ry1 <= int(ky) <= ry2):
                    return True
    if getattr(res, "boxes", None) is not None:
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
    ToTensorV2(),
])

USE_RADIAL_CHANNEL = False
IN_CHANNELS        = 3


def _dilate_class_protected(bm, kernel, protect_mask=None):
    """(Kept for backward compatibility — no longer used by infer(), which
    now calls the batched, Voronoi-conflict-resolved versions below.)"""
    dilated = cv2.dilate(bm, kernel)
    if protect_mask is not None:
        dilated[protect_mask == 1] = 0
    return dilated


def _close_class_protected(bm, kernel, protect_mask=None):
    """(Kept for backward compatibility — no longer used by infer().)"""
    dilated = cv2.dilate(bm, kernel)
    if protect_mask is not None:
        dilated[protect_mask == 1] = 0
    closed = cv2.erode(dilated, kernel)
    closed = np.maximum(closed, bm)
    if protect_mask is not None:
        closed[protect_mask == 1] = 0
    return closed


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I54] TUBE-CLASS CONFLICT RESOLUTION — FIXES MASK BLEEDING
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_tube_conflicts(masks: dict, dists: dict) -> dict:
    """
    Given a dict {class_id: binary_mask} where the SAME pixel may be set to
    1 in more than one class's mask (a "contested" pixel — grown there by
    dilation/close from two different neighbouring tubes), resolve every
    contested pixel to the single class whose ORIGINAL (pre-growth) pixels
    are geometrically NEAREST, using a per-class distance transform
    (`dists`). Non-contested pixels are left untouched.

    This is what stops one tube's color from bleeding into a neighbouring
    tube: instead of "whichever class is pasted back last wins" (the old,
    order-dependent behaviour), the boundary between any two tubes is
    always the perpendicular bisector between their nearest real pixels —
    identical regardless of processing order.

    Returns a NEW dict of mutually-exclusive masks: no pixel is 1 in more
    than one output mask.
    """
    classes = list(masks.keys())
    stack = np.stack([masks[ci] for ci in classes], axis=0)  # (K, H, W) uint8
    claim_count = stack.sum(axis=0)
    contested = claim_count > 1
    if not contested.any():
        return {ci: masks[ci] for ci in classes}

    dist_stack = np.stack([dists[ci] for ci in classes], axis=0).astype(np.float32)
    # Only classes that actually claimed a given pixel are eligible to win
    # it — everyone else gets +inf so they can never "win" a pixel they
    # never grew into in the first place.
    dist_eligible = np.where(stack == 1, dist_stack, np.inf)
    winner_idx = np.argmin(dist_eligible, axis=0)  # (H, W) index into `classes`

    out = {}
    for i, ci in enumerate(classes):
        m = masks[ci].copy()
        lose = contested & (winner_idx != i)
        m[lose] = 0
        out[ci] = m
    return out


def _dilate_multiclass_protected(work, classes, kernel):
    """
    [FIX-I50 + FIX-I54] Batched dilation across all tube classes (2/3/4) in
    a single cv2 call, with two protection rules:
      1. A class is never allowed to grow into pixels already claimed by a
         DIFFERENT non-tube class (e.g. class 1 / device).
      2. [FIX-I54] Where two tube classes' dilation both reach the SAME
         background pixel, that pixel is awarded to whichever class's
         ORIGINAL (pre-dilation) pixels are nearest — not to whichever
         class happens to be processed last. This is what stops one tube's
         painted band from bleeding into a neighbouring tube's territory
         (previously the fixed loop order (2, 3, 4) let the later class
         silently overwrite the earlier one in every contested gap pixel).
    Returns a dict {class_id: dilated_binary_mask}, guaranteed mutually
    exclusive (no pixel set in more than one class's mask).
    """
    originals = {ci: (work == ci).astype(np.uint8) for ci in classes}
    if not any(o.any() for o in originals.values()):
        return {ci: originals[ci] for ci in classes}

    stack = np.stack([originals[ci] for ci in classes], axis=-1)
    dilated = cv2.dilate(stack, kernel)
    if dilated.ndim == 2:  # OpenCV squeezes a single-channel result
        dilated = dilated[..., None]

    masks = {}
    for i, ci in enumerate(classes):
        d = dilated[..., i].copy()
        other = ((work != 0) & (work != ci)).astype(np.uint8)  # protect vs. device/other non-tube classes
        d[other == 1] = 0
        masks[ci] = d

    # [FIX-I54] Resolve any pixel that more than one tube class grew into.
    dists = {
        ci: cv2.distanceTransform((originals[ci] == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for ci in classes
    }
    masks = _resolve_tube_conflicts(masks, dists)
    return masks


def _close_multiclass_protected(work, classes, kernel):
    """
    [FIX-I50 + FIX-I54] Batched morphological CLOSE (dilate-then-erode)
    across all tube classes at once, used to bridge small gaps within a
    single tube's own blobs. Same two protection rules as
    _dilate_multiclass_protected above — including the [FIX-I54]
    nearest-original-pixel conflict resolution, so the gap-bridging CLOSE
    can never accidentally fuse/bleed into a NEIGHBOURING tube's blobs
    either, only into that same class's own nearby pieces.
    Returns a dict {class_id: closed_binary_mask}, guaranteed mutually
    exclusive.
    """
    originals = {ci: (work == ci).astype(np.uint8) for ci in classes}
    if not any(o.any() for o in originals.values()):
        return {ci: originals[ci] for ci in classes}

    stack = np.stack([originals[ci] for ci in classes], axis=-1)
    dilated = cv2.dilate(stack, kernel)
    if dilated.ndim == 2:
        dilated = dilated[..., None]

    dil_masks = {}
    for i, ci in enumerate(classes):
        d = dilated[..., i].copy()
        other = ((work != 0) & (work != ci)).astype(np.uint8)
        d[other == 1] = 0
        dil_masks[ci] = d

    dil_stack = np.stack([dil_masks[ci] for ci in classes], axis=-1)
    eroded = cv2.erode(dil_stack, kernel)
    if eroded.ndim == 2:
        eroded = eroded[..., None]

    closed = {}
    for i, ci in enumerate(classes):
        # Re-OR the original pixels back in — erode can occasionally eat
        # into small isolated original blobs faster than the bridging
        # dilate step re-grew them; this guarantees CLOSE never shrinks
        # the mask below what was already there.
        c = np.maximum(eroded[..., i], originals[ci])
        other = ((work != 0) & (work != ci)).astype(np.uint8)
        c = c.copy()
        c[other == 1] = 0
        closed[ci] = c

    # [FIX-I54] Resolve any pixel the CLOSE step let two different tube
    # classes both claim.
    dists = {
        ci: cv2.distanceTransform((originals[ci] == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for ci in classes
    }
    closed = _resolve_tube_conflicts(closed, dists)
    return closed


@torch.no_grad()
def raw_infer(seg_model, frame_bgr, socket_centre=None):
    oh, ow = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t_u8   = _tf_pipeline(image=rgb)["image"]  # uint8 CHW tensor, still on CPU

    compute_dtype = torch.float16 if (USE_HALF and DEVICE.startswith("cuda")) else torch.float32

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
        logits = seg_model(t)

    probs = F.softmax(logits.float(), dim=1)          # (1, C, 512, 512), still on GPU
    scale = IMG_SIZE[0] / max(oh, ow)
    new_h = int(oh * scale)
    new_w = int(ow * scale)
    pad_y = (IMG_SIZE[0] - new_h) // 2
    pad_x = (IMG_SIZE[1] - new_w) // 2
    crop  = probs[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w]  # still GPU

    resized = F.interpolate(crop, size=(oh, ow), mode="bilinear",
                             align_corners=False)
    return resized.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class SegmentationEngine:
    """
    Runs live inference every frame. EMA is maintained continuously so
    temporal smoothing still applies, but the mask is never frozen — it
    follows the object as it moves.
    """

    _TUBE_CLASSES = (2, 3, 4)

    def __init__(self, model):
        self.model         = model
        self.ema_probs     = None
        self._persist      = {c: 0 for c in range(1, NUM_CLASSES)}
        self._morph_k      = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SZ, MORPH_KERNEL_SZ))
        self._dilate_k     = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_DILATE_SZ, MASK_DILATE_SZ))
        self._close_k       = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_CLOSE_SZ, MASK_CLOSE_SZ))
        self.last_raw_pred = None
        self._dbg_frame_ct = 0
        self.identity_ema  = None
        self.identity_lock = None
        self._prev_gray_track = None

    def reset_dilate_kernel(self):
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

    def _get_perf_roi(self, center, frame_shape, pad=None):
        if not PERF_ROI_ENABLED or center is None:
            return None
        pad = PERF_ROI_PAD if pad is None else pad
        h, w = frame_shape[:2]
        cx, cy = center
        x1 = max(0, int(cx - pad)); x2 = min(w, int(cx + pad))
        y1 = max(0, int(cy - pad)); y2 = min(h, int(cy + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _track_lock_with_flow(self, frame_bgr, roi=None):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray_track is None or self.identity_lock is None:
            self._prev_gray_track = gray
            return
        if gray.shape != self._prev_gray_track.shape:
            self._prev_gray_track = gray
            return

        if roi is not None:
            x1, y1, x2, y2 = roi
        else:
            x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]

        gray_crop = gray[y1:y2, x1:x2]
        prev_crop = self._prev_gray_track[y1:y2, x1:x2]

        flow = _farneback_flow_downscaled(prev_crop, gray_crop)

        hc, wc = gray_crop.shape
        gx, gy = np.meshgrid(np.arange(wc), np.arange(hc))
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)

        lock_crop = self.identity_lock[y1:y2, x1:x2]
        self.identity_lock[y1:y2, x1:x2] = cv2.remap(
            lock_crop, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        for i in range(self.identity_ema.shape[0]):
            ema_crop = self.identity_ema[i, y1:y2, x1:x2]
            self.identity_ema[i, y1:y2, x1:x2] = cv2.remap(
                ema_crop, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        self._prev_gray_track = gray

    def _apply_identity_hysteresis(self, probs, pred, enforce=True):
        h, w = probs.shape[1], probs.shape[2]
        if self.identity_ema is None:
            self.identity_ema  = np.zeros((3, h, w), dtype=np.float32)
            self.identity_lock = np.zeros((h, w), dtype=np.int8)

        class_probs = np.stack([probs[c] for c in self._TUBE_CLASSES], axis=0)
        tube_mass   = class_probs.sum(axis=0)
        present     = tube_mass >= TUBE_PRESENT_THR

        self.identity_ema[:, present] = (
            IDENTITY_EMA_ALPHA * class_probs[:, present]
            + (1.0 - IDENTITY_EMA_ALPHA) * self.identity_ema[:, present]
        )

        self.identity_lock[~present] = 0

        best_idx = np.argmax(self.identity_ema, axis=0)
        best_val = np.take_along_axis(
            self.identity_ema, best_idx[None, :, :], axis=0)[0]

        lock_idx = np.clip(self.identity_lock.astype(np.int32) - 2, 0, 2)
        cur_val  = np.take_along_axis(
            self.identity_ema, lock_idx[None, :, :], axis=0)[0]

        unset = present & (self.identity_lock == 0)
        self.identity_lock[unset] = (best_idx[unset] + 2).astype(np.int8)

        locked = present & (self.identity_lock != 0) & (~unset)
        flip   = locked & (best_idx != lock_idx) & (best_val > cur_val + TUBE_IDENTITY_MARGIN)
        self.identity_lock[flip] = (best_idx[flip] + 2).astype(np.int8)

        override = present & np.isin(pred, self._TUBE_CLASSES) & (self.identity_lock != 0)
        if enforce:
            pred[override] = self.identity_lock[override]
        else:
            override = np.zeros_like(override)

        if MASK_DEBUG:
            n_flip = int(flip.sum())
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=identity_lock   "
                  f"present_px={int(present.sum())}  flips_this_frame={n_flip}  "
                  f"enforce={enforce}  overridden_px={int(override.sum())}")

        return pred, override

    def infer(self, frame_bgr, socket_centre=None, apply_identity_lock=True):
        self._dbg_frame_ct += 1

        perf_roi = self._get_perf_roi(socket_centre, frame_bgr.shape)

        self._track_lock_with_flow(frame_bgr, roi=perf_roi)

        probs_raw = raw_infer(self.model, frame_bgr, socket_centre=socket_centre)

        if MASK_DEBUG:
            raw_argmax = probs_raw.argmax(axis=0)
            counts_raw = {ci: int((raw_argmax == ci).sum()) for ci in (2, 3, 4)}
            maxp_raw   = {ci: float(probs_raw[ci].max()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=raw_argmax      "
                  f"px={counts_raw}  maxprob={ {k: round(v,3) for k,v in maxp_raw.items()} }")

        self.ema_probs = (
            probs_raw.copy() if self.ema_probs is None
            else EMA_ALPHA * probs_raw + (1 - EMA_ALPHA) * self.ema_probs
        )

        probs = self.ema_probs.copy()

        if MASK_DEBUG:
            ema_argmax = probs.argmax(axis=0)
            counts_ema = {ci: int((ema_argmax == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_ema       px={counts_ema}")

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

        lock_override_mask = None
        if IDENTITY_HYSTERESIS_ENABLED:
            pred, lock_override_mask = self._apply_identity_hysteresis(
                probs, pred, enforce=apply_identity_lock)

        active_conf_thr = CLASS_CONF_THR if apply_identity_lock else WARMUP_CLASS_CONF_THR

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

        if perf_roi is not None:
            px1, py1, px2, py2 = perf_roi
            if px1 > 0:
                pred[:, :px1] = 0
            if px2 < pred.shape[1]:
                pred[:, px2:] = 0
            if py1 > 0:
                pred[:py1, :] = 0
            if py2 < pred.shape[0]:
                pred[py2:, :] = 0
            work = pred[py1:py2, px1:px2]
        else:
            work = pred

        if MASK_DILATE_ENABLED:
            any_tube = any((work == ci).any() for ci in MASK_DILATE_CLASSES)
            if any_tube:
                # [FIX-I54] These two calls now internally resolve any
                # pixel more than one tube class tries to claim by
                # nearest-original-pixel ownership — see
                # _resolve_tube_conflicts / _dilate_multiclass_protected /
                # _close_multiclass_protected above. The paste-back loop
                # below is now safe regardless of class order because the
                # returned masks are already guaranteed mutually exclusive.
                dilated_by_class = _dilate_multiclass_protected(
                    work, MASK_DILATE_CLASSES, self._dilate_k)
                for ci in MASK_DILATE_CLASSES:
                    work[work == ci] = 0
                for ci in MASK_DILATE_CLASSES:
                    work[dilated_by_class[ci] == 1] = ci

            if MASK_DEBUG:
                counts_dilate = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
                print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_dilate    "
                      f"px={counts_dilate}  dilate_sz={MASK_DILATE_SZ} "
                      f"classes={MASK_DILATE_CLASSES}")

            if MASK_CLOSE_ENABLED and MASK_CLOSE_SZ > 0:
                any_tube2 = any((work == ci).any() for ci in MASK_DILATE_CLASSES)
                if any_tube2:
                    closed_by_class = _close_multiclass_protected(
                        work, MASK_DILATE_CLASSES, self._close_k)
                    for ci in MASK_DILATE_CLASSES:
                        work[work == ci] = 0
                    for ci in MASK_DILATE_CLASSES:
                        work[closed_by_class[ci] == 1] = ci

                if MASK_DEBUG:
                    counts_close = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
                    print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_close     "
                          f"px={counts_close}  close_sz={MASK_CLOSE_SZ} "
                          f"classes={MASK_DILATE_CLASSES}")

        for ci in range(1, NUM_CLASSES):
            bm = (work == ci).astype(np.uint8)
            bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, self._morph_k)
            if ci not in (3, 4):
                bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, self._morph_k)
            work[work == ci] = 0
            work[bm == 1]    = ci

        if MASK_DEBUG:
            counts_morph = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_morph     px={counts_morph}")

        for ci in range(1, NUM_CLASSES):
            bm = (work == ci).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bm)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PX:
                    work[labels == i] = 0

        if MASK_DEBUG:
            counts_cc = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=after_ccfilter  "
                  f"px={counts_cc}  min_area={MIN_AREA_PX}")

        for ci in range(1, NUM_CLASSES):
            self._persist[ci] = self._persist[ci] + 1 if (pred == ci).any() else 0
            if self._persist[ci] < PERSIST_FRAMES:
                pred[pred == ci] = 0

        if MASK_DEBUG:
            counts_final = {ci: int((pred == ci).sum()) for ci in (2, 3, 4)}
            print(f"[MASKDBG f{self._dbg_frame_ct:05d}] STAGE=final           px={counts_final}\n")

        return pred.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
#  VOTE COUNTER
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
#  TUBE ORDER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_tube_order(pred_map, socket_bbox=None, debug=False):
    if socket_bbox is not None:
        x1, y1, x2, y2 = socket_bbox
        scx, scy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    else:
        ys, xs = np.where(pred_map == 1)
        scx    = float(xs.mean()) if len(xs) else pred_map.shape[1] / 2.0
        scy    = float(ys.mean()) if len(ys) else pred_map.shape[0] / 2.0

    H, W = pred_map.shape[:2]
    R    = NEAREST_SEARCH_RADIUS

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
#  CYCLE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class CycleManager:
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

        self.passed  = 0
        self.failed  = 0
        self.unknown = 0

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
        shutil.move(self.temp_path, final_path)

        if verdict == "NORMAL":
            self.passed += 1
        elif verdict == "ANOMALY":
            self.failed += 1
        else:
            self.unknown += 1

        duration_s = time.time() - (self.start_time or time.time())
        print(f"[CYCLE END]    #{self.cycle_no:03d}  →  {verdict:<8}  "
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
        print("\n" + "█" * W)
        print("  FINAL CYCLE REPORT")
        print("█" * W)
        print(f"  TOTAL CYCLES      : {self.total_cycles}")
        print(f"  PASSED (NORMAL)   : {self.passed}")
        print(f"  FAILED (ANOMALY)  : {self.failed}")
        print(f"  UNKNOWN           : {self.unknown}")
        print("█" * W + "\n")


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


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-I53] SOCKET / NO-SOCKET BOUNDING BOX — FILLED
# ══════════════════════════════════════════════════════════════════════════════
def draw_socket_box(frame, hit, fill_alpha=None):
    if hit is None:
        return frame

    fill_alpha = SOCKET_BOX_FILL_ALPHA if fill_alpha is None else fill_alpha

    x1, y1, x2, y2 = hit["bbox"]
    is_p  = hit["class"] == CLS_SOCKET

    col      = (0, 220, 100) if is_p else (50, 50, 230)
    fill_col = (150, 255, 190) if is_p else (140, 140, 255)

    label = f"{'Socket' if is_p else 'No Socket'}  {hit['conf']*100:.0f}%"

    obb_pts = hit.get("obb_points")

    overlay = frame.copy()
    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], fill_col)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_col, -1)
    frame = cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0)

    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=col,
                      thickness=2, lineType=cv2.LINE_AA)
        label_x = min(p[0] for p in obb_pts)
        label_y = min(p[1] for p in obb_pts)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label_x, label_y = x1, y1

    fs = 0.60
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    by = max(label_y - 6, th + 6)
    cv2.rectangle(frame, (label_x, by - th - 6), (label_x + tw + 10, by + bl), col, -1)
    cv2.putText(frame, label, (label_x + 5, by - 2),
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
#  DEBUG OVERLAY
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
#  HUD
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

    TOP_OFFSET = max(70, int(78 * S))

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
        _put(out, "DETECTIONS PAUSED (hand in ROI)",
             lx, ly + ROW - 6, FS_XS, _ROI_COL_AMBER, TK1);  ly += ROW
    else:
        _put(out, f"LIVE INFER   [{infer_frames}f]",
             lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        if mask_is_locked:
            _put(out, "MASK: LOCKED", lx + 190, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        else:
            _put(out, "MASK: REFRESHED", lx + 190, ly + ROW - 6, FS_XS, (0, 165, 255), TK1)
        ly += ROW

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] {Path(video_path).name} — {src_w}x{src_h} @ {fps_src:.1f} fps ({total_frames} frames)")

    is_short_clip = (total_frames > 0 and total_frames < 250)
    effective_warmup_frames = 2 if is_short_clip else WARMUP_FRAMES
    effective_anomaly_confirm = 2 if is_short_clip else N_ANOMALY_CONFIRM
    effective_max_retries = 0 if is_short_clip else MAX_WARMUP_RETRIES

    if is_short_clip:
        print(f"[AUTO-DETECT] Short single-cycle clip detected ({total_frames} frames). "
              f"Auto-configured warmup={effective_warmup_frames} frames, "
              f"anomaly_confirm={effective_anomaly_confirm} frames.")

    video_stem = Path(video_path).stem
    cycles     = CycleManager(resolved_output_dir, video_stem, fps_src, (src_w, src_h))

    seg_engine   = SegmentationEngine(seg_net)
    anomaly_gate = AnomalyConfirmGate(effective_anomaly_confirm)
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

    last_socket_centre = None
    last_socket_bbox_size = None

    mask_frozen_pred     = None
    mask_frozen_gray_roi = None
    mask_is_locked        = False

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

        sock_future = _YOLO_EXECUTOR.submit(
            detect_socket, yolo_socket, frame, YOLO_SOCKET_CONF)
        hand_future = None
        if invisible_roi is not None:
            hand_future = _YOLO_EXECUTOR.submit(
                detect_hand_in_roi, yolo_pose, frame, invisible_roi, YOLO_POSE_CONF)

        sock_hit      = sock_future.result()
        socket_now    = sock_hit is not None and sock_hit["class"] == CLS_SOCKET
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET

        if socket_now:
            x1, y1, x2, y2    = sock_hit["bbox"]
            last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
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

        hand_in_roi = hand_future.result() if hand_future is not None else False

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
            # ── HAND: inference paused, segmentation mask NEVER shown ──
            # [FIX-I25] pred_map is forced to all-background, no
            # draw_seg_overlay call happens in this branch at all, and
            # the mask-freeze cache is invalidated (a hand may have
            # repositioned the tube). Only the independent socket YOLO
            # box is drawn so ROI/cycle context stays visible.
            state        = STATE_HAND
            pred_map     = ZERO_PRED
            status_dict  = EMPTY_STAT
            order_status = "N/A"
            detected_seq = []
            current_dbg  = {}
            mask_frozen_pred     = None
            mask_frozen_gray_roi = None
            mask_is_locked        = False
            vis = draw_socket_box(vis, sock_hit)

        else:
            # ── LIVE INFERENCE (warmup) / MOTION-GATED MASK FREEZE (post-warmup) ──
            # This branch only runs when hand_in_roi is False — the mask
            # is only ever drawn here, confirming the "mask only when NO
            # hand" requirement end-to-end.
            cycle_total_frames += 1
            lock_engage = (
                True if LOCK_ENGAGE_MODE == "immediate" else warmup_done
            )

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
                pm = restrict_mask_to_socket_roi(
                    pm, roi_center, bbox_size=roi_bbox_size)
                pm = apply_exclusion_zone(
                    pm, roi_center, bbox_size=roi_bbox_size)
                return pm

            if not warmup_done:
                pred_map    = _run_fresh_inference()
                mask_is_locked = False
            elif not MASK_FREEZE_ENABLED:
                pred_map    = _run_fresh_inference()
                mask_is_locked = False
            else:
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

            stable_order = seq_gate.update(raw_order, detected_seq)
            gate_result  = anomaly_gate.update(stable_order)
            order_status = gate_result
            latch_gate.update(gate_result)

            has_tubes = any((pred_map == ci).any() for ci in (2, 3, 4))

            if not warmup_done:
                warmup_frame_count += 1
                state = STATE_WARMUP

                if has_tubes:
                    vis = draw_seg_overlay(vis, pred_map)
                elif seg_engine.last_raw_pred is not None:
                    vis = draw_raw_argmax_fallback(vis, seg_engine.last_raw_pred)

                target = effective_warmup_frames * (warmup_retry + 1)
                if warmup_frame_count >= target:
                    if not has_tubes and warmup_retry < effective_max_retries:
                        warmup_retry += 1
                        print(f"[WARN] f{frame_idx:05d}: warmup blank → retry "
                              f"{warmup_retry}/{effective_max_retries}")
                    else:
                        warmup_done = True
                        print(f"[WARMUP DONE] cycle#{cycles.cycle_no:03d} f{frame_idx:05d}  "
                              f"tube_px={[(ci, int((pred_map == ci).sum())) for ci in (2, 3, 4)]}")

            else:
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

            prev_pred   = pred_map
            prev_status = status_dict
            prev_order  = order_status
            prev_seq    = detected_seq
            prev_dbg    = current_dbg

            vis = draw_socket_box(vis, sock_hit)

        if enable_debug and current_dbg:
            vis = draw_debug_overlay(vis, current_dbg)

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
#  EXCEL LOGGING
# ══════════════════════════════════════════════════════════════════════════════
_EXCEL_COLUMNS = [
    ("sr_no",             "Sr No."),
    ("timestamp",         "Timestamp"),
    ("cycle_no",          "Cycle No."),
    ("filename",          "Video File"),
    ("final_verdict",     "Status"),
    ("output_path",       "Video Folder Saving Path"),
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

    # [FIX-I55] Sr No. is a simple running count of data rows already in
    # the sheet (ws.max_row includes the header row, so existing data rows
    # = ws.max_row - 1; the next Sr No. is that count + 1). This is
    # computed fresh from the sheet itself rather than tracked in memory,
    # so it stays correct even across separate runs of the script that
    # append to the same inspection_log.xlsx.
    existing_data_rows = max(0, ws.max_row - 1)
    sr_no = existing_data_rows + 1

    row_vals = []
    for key, _ in _EXCEL_COLUMNS:
        if key == "sr_no":
            row_vals.append(sr_no)
        elif key == "cycle_no":
            row_vals.append(int(run_metrics.get(key, 0)))
        elif key == "timestamp":
            row_vals.append(run_metrics.get(key, "N/A"))
        else:
            row_vals.append(run_metrics.get(key, "N/A"))
    ws.append(row_vals)

    cr   = ws.max_row
    thin = Side(style="thin", color="BFBFBF")
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
    verdict_keys = {"final_verdict"}
    center_keys  = {"sr_no", "cycle_no", "final_verdict", "timestamp"}

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
            print(f"[EXCEL] Sr#{sr_no}  Cycle #{run_metrics.get('cycle_no','-')} → {excel_path}")
            return
        except PermissionError:
            if attempt == MAX_EXCEL_RETRIES:
                raise
            print(f"[WARN] Excel locked, retry {attempt}/{MAX_EXCEL_RETRIES} in 5s…")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN MANAGER
# ══════════════════════════════════════════════════════════════════════════════
def run_single_video(video_path, seg_model_path, out_base,
                     yolo_socket_path, hand_pose_path, print_summary,
                     enable_debug=False, forced_channels=None,
                     require_gpu=True, use_half=True, gpu_warmup_hw=None):

    global DEVICE, USE_HALF, GPU_WARMUP_HW
    DEVICE = resolve_device(require_gpu=require_gpu)

    USE_HALF = bool(use_half) and DEVICE.startswith("cuda")
    if gpu_warmup_hw is not None:
        GPU_WARMUP_HW = gpu_warmup_hw

    print(f"[INFO] Device              : {DEVICE}")
    print(f"[INFO] [FIX-I39] FP16 (half precision) : {USE_HALF}"
          + ("" if DEVICE.startswith("cuda") else "  (forced OFF — no CUDA device)"))
    print(f"[INFO] [FIX-I39] GPU warmup shape (HxW): {GPU_WARMUP_HW}")
    print(f"[INFO] Mode                : SINGLE VIDEO / MULTI-CYCLE (live inference, merged v45+v49)")
    print(f"[INFO] Input video         : {video_path}")
    print(f"[INFO] HSV gate            : {'ON' if USE_HSV_GATE else 'OFF'}")
    print(f"[INFO] Warmup              : {WARMUP_FRAMES} frames × up to {MAX_WARMUP_RETRIES} retries")
    print(f"[INFO] [FIX-I1] EMA alpha  : {EMA_ALPHA}")
    print(f"[INFO] [FIX-I2/I8] Sharpen τ : {TUBE_SHARPNESS_TEMP}  enabled={TUBE_BOUNDARY_SHARPENING}")
    print(f"[INFO] [FIX-I3/I8] Conf thr : {CLASS_CONF_THR}")
    print(f"[INFO] [FIX-I6] Seq stable : {MIN_SEQ_STABLE} frames")
    print(f"[INFO] [FIX-I8] MIN_TUBE_PX: {MIN_TUBE_PX}  MIN_AREA_PX: {MIN_AREA_PX}")
    print(f"[INFO] [FIX-I18/I27] Nearest-pixel angular gate, RECTANGULAR search "
          f"half-side: {NEAREST_SEARCH_RADIUS}px")
    print(f"[INFO] [FIX-I21/I41/I42/I50/I54] Mask dilation (width, batched, conflict-resolved): "
          f"enabled={MASK_DILATE_ENABLED}  size={MASK_DILATE_SZ}px  classes={MASK_DILATE_CLASSES}")
    print(f"[INFO] [FIX-I43/I50/I54] Mask close (gap bridging, batched, conflict-resolved) : "
          f"enabled={MASK_CLOSE_ENABLED}  size={MASK_CLOSE_SZ}px")
    print(f"[INFO] [FIX-I44] Overlay alpha (solidity)      : {OVERLAY_ALPHA}")
    print(f"[INFO] [FIX-I53] Socket/No-Socket box fill alpha: {SOCKET_BOX_FILL_ALPHA}")
    print(f"[INFO] [FIX-I23/I24] 3-way tube identity lock with optical-flow tracking: "
          f"enabled={IDENTITY_HYSTERESIS_ENABLED}  margin={TUBE_IDENTITY_MARGIN}  "
          f"ema_alpha={IDENTITY_EMA_ALPHA}")
    print(f"[INFO] [FIX-I25] Clean HAND state (segmentation mask NEVER shown while hand in ROI): enabled")
    print(f"[INFO] [FIX-I28] Frame latency warn threshold: {FRAME_MS_WARN_THRESHOLD:.0f}ms")
    print(f"[INFO] [FIX-I29/I55] Excel columns: sr_no, timestamp, cycle_no, filename, final_verdict, output_path")
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
    print(f"[INFO] [FIX-I45] Perf ROI (bounds expensive CPU morphology/flow work): "
          f"enabled={PERF_ROI_ENABLED}  pad={PERF_ROI_PAD}px")
    print(f"[INFO] [FIX-I46/I47] Socket detector accepts standard-detect OR OBB YOLO weights, "
          f"and OBB detections are now rendered as TRUE rotated boxes")
    print(f"[INFO] [FIX-I48] GPU-side mask resize (single F.interpolate call, one .cpu() sync)")
    print(f"[INFO] [FIX-I49] Downsampled optical flow (all 3 Farneback call sites): "
          f"downscale={OPT_FLOW_DOWNSCALE}")
    print(f"[INFO] [FIX-I50] Batched multi-class dilate/close (1 cv2 call instead of 3 per stage)")
    print(f"[INFO] [FIX-I51] Concurrent socket + hand/pose YOLO detection (ThreadPoolExecutor)")
    print(f"[INFO] [FIX-I52] cv2.setNumThreads explicitly set to {cv2.getNumThreads()}")
    print(f"[INFO] [FIX-I53] Filled socket/no-socket bounding box (light green / light red wash)")
    print(f"[INFO] [FIX-I54] Nearest-original-pixel conflict resolution for tube dilate/close "
          f"(prevents one tube color bleeding into a neighbouring tube)")
    print(f"[INFO] [FIX-I55] Excel log Sr No. + Timestamp columns added")
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
    if USE_HALF:
        seg_net = seg_net.half()
    warmup_dtype = torch.float16 if USE_HALF else torch.float32
    with torch.no_grad():
        seg_net(torch.zeros(1, IN_CHANNELS, *IMG_SIZE, device=DEVICE, dtype=warmup_dtype))
    print("[INFO] GPU warmup done.")
    _log_model_device("Segmentation UNet++", f"{next(seg_net.parameters()).device}  half={USE_HALF}")

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
                    "3-way tracked identity lock + clean HAND state (mask never "
                    "shown while hand in ROI) + socket-adjacent mask ROI gate + "
                    "[FIX-I38/I39] enforced GPU execution with full GPU utilization "
                    "(TF32 + FP16 + GPU-side normalization) + [FIX-I41/I42/I43/I44] "
                    "thick, continuous, high-opacity mask overlay + [FIX-I45] "
                    "socket-centred perf ROI + [FIX-I46/I47] OBB-compatible socket "
                    "detector with TRUE rotated-box rendering + [FIX-I48..I52] "
                    "real-time CPU-bottleneck fixes + [FIX-I53] filled socket/no-socket "
                    "bounding box + [FIX-I54] nearest-original-pixel conflict "
                    "resolution for tube dilate/close so tube colors never bleed "
                    "into each other + [FIX-I55] Excel log Sr No. + Timestamp columns")
    ap.add_argument("--video",         default=DEFAULT_VIDEO, required=True)
    ap.add_argument("--model",         default=DEFAULT_MODEL)
    ap.add_argument("--out_base",      default=_OUT_BASE)
    ap.add_argument("--yolo",          default=DEFAULT_YOLO, required=True,
                    help="Path to the socket-detector YOLO weights. Accepts EITHER a "
                         "standard axis-aligned detect-head model OR a YOLO-OBB "
                         "(oriented bounding box) model — auto-detected [FIX-I46], and "
                         "drawn with its true rotation [FIX-I47].")
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
                    help="[FIX-I31/I32/I33/I34] Shape of the socket-adjacent keep-region.")
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
                         "on every frame instead.")
    ap.add_argument("--mask_roi_mult_up", type=float, default=MASK_ROI_MULT_UP,
                    help="[FIX-I35] Auto-scale multiplier: reach above socket = this x "
                         "socket bbox HEIGHT. Default 1.3.")
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
                         "running live inference every single frame.")
    ap.add_argument("--mask_freeze_motion_thr", type=float, default=MASK_FREEZE_MOTION_THR,
                    help="[FIX-I37] Per-pixel optical-flow magnitude (px/frame) considered "
                         "'moved'. Default 2.5.")
    ap.add_argument("--mask_freeze_motion_frac", type=float, default=MASK_FREEZE_MOTION_FRAC,
                    help="[FIX-I37] Fraction of the socket ROI that must exceed the motion "
                         "threshold before the mask is refreshed. Default 0.04.")
    ap.add_argument("--opt_flow_downscale", type=float, default=OPT_FLOW_DOWNSCALE,
                    help="[FIX-I49] Downscale factor for Farneback optical flow. Default 0.5.")
    ap.add_argument("--mask_roi_polygon_file", type=str, default=None,
                    help="[FIX-I33] Path to a JSON file containing a list of [dx, dy] "
                         "offsets FROM THE SOCKET CENTRE defining the exact keep-region "
                         "boundary for shape=polygon.")
    ap.add_argument("--mask_roi_rx", type=int, default=MASK_ROI_RADIUS_X,
                    help="[FIX-I32] Horizontal radius (px) for shape=ellipse. Default 160.")
    ap.add_argument("--mask_roi_ry", type=int, default=MASK_ROI_RADIUS_Y,
                    help="[FIX-I32] Vertical radius (px) for shape=ellipse. Default 330.")
    ap.add_argument("--mask_roi_offset_x", type=int, default=MASK_ROI_OFFSET_X,
                    help="[FIX-I32] Shift the ROI centre left(-)/right(+) of the socket "
                         "centre, in px. Default 0.")
    ap.add_argument("--mask_roi_offset_y", type=int, default=MASK_ROI_OFFSET_Y,
                    help="[FIX-I32] Shift the ROI centre up(-)/down(+) of the socket "
                         "centre, in px. Default -120.")
    ap.add_argument("--socket_grace",  type=int,   default=SOCKET_RESET_GRACE,
                    help="Consecutive 'no socket' frames required to close a cycle. Default 45.")
    ap.add_argument("--dilate_sz",     type=int,   default=MASK_DILATE_SZ,
                    help="[FIX-I21/I41/I42/I50/I54] Kernel size (px) used to widen tube-class "
                         "masks (2/3/4). Default 18.")
    ap.add_argument("--close_sz",      type=int,   default=MASK_CLOSE_SZ,
                    help="[FIX-I43/I50/I54] Kernel size (px) for the post-dilation gap-bridging "
                         "CLOSE. Default 40. Set to 0 to disable just the close pass.")
    ap.add_argument("--overlay_alpha", type=float, default=OVERLAY_ALPHA,
                    help="[FIX-I44] Blend alpha for the tube-class overlay layer (0-1). "
                         "Default 0.90.")
    ap.add_argument("--socket_fill_alpha", type=float, default=SOCKET_BOX_FILL_ALPHA,
                    help="[FIX-I53] Blend alpha for the translucent interior fill of the "
                         "socket / no-socket detection box (0-1). Default 0.28.")
    ap.add_argument("--no_dilate",     action="store_true",
                    help="[FIX-I21] Disable mask-widening dilation (and the close pass).")
    ap.add_argument("--identity_margin", type=float, default=TUBE_IDENTITY_MARGIN,
                    help="[FIX-I23] EMA margin a class must exceed the locked class by "
                         "to flip a pixel's tube identity. Default 0.12.")
    ap.add_argument("--no_identity_lock", action="store_true",
                    help="[FIX-I23] Disable the 3-way tube identity lock.")
    ap.add_argument("--lock_engage_mode", choices=["post_warmup", "immediate"],
                    default=LOCK_ENGAGE_MODE,
                    help="[FIX-I26] Default post_warmup.")
    ap.add_argument("--locked_conf_floor", type=float, default=LOCKED_PIXEL_MIN_CONF,
                    help="[FIX-I26] Relaxed confidence floor for locked pixels. Default 0.08.")
    ap.add_argument("--frame_ms_warn", type=float, default=FRAME_MS_WARN_THRESHOLD,
                    help="[FIX-I28] Per-frame latency (ms) warn threshold. Default 150.0.")
    ap.add_argument("--perf_roi_pad", type=int, default=PERF_ROI_PAD,
                    help="[FIX-I45] Half-width/half-height (px) of the socket-centred perf "
                         "ROI. Default 550.")
    ap.add_argument("--no_perf_roi", action="store_true",
                    help="[FIX-I45] Disable the socket-centred perf ROI crop.")
    ap.add_argument("--hsv_gate",      action="store_true")
    ap.add_argument("--print_summary", action="store_true")
    ap.add_argument("--debug",         action="store_true")
    ap.add_argument("--mask_debug",    action="store_true",
                    help="[FIX-I8] Print per-stage pixel-count trace. Toggle live with 'm'.")
    ap.add_argument("--no_require_gpu", action="store_true",
                    help="[FIX-I38] Allow CPU execution instead of raising an error.")
    ap.add_argument("--half",          dest="half", action="store_true", default=True,
                    help="[FIX-I39] Run models in FP16. Default whenever CUDA is active.")
    ap.add_argument("--no_half",       dest="half", action="store_false",
                    help="[FIX-I39] Force FP32 everywhere instead of FP16.")
    ap.add_argument("--gpu_warmup_hw", type=int, nargs=2, default=None,
                    metavar=("HEIGHT", "WIDTH"),
                    help="[FIX-I39] H W of the dummy warmup frame. Default: 720 1280.")
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
    OPT_FLOW_DOWNSCALE           = args.opt_flow_downscale
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
    SOCKET_BOX_FILL_ALPHA     = args.socket_fill_alpha
    MASK_DILATE_ENABLED       = not args.no_dilate
    TUBE_IDENTITY_MARGIN      = args.identity_margin
    IDENTITY_HYSTERESIS_ENABLED = not args.no_identity_lock
    LOCK_ENGAGE_MODE          = args.lock_engage_mode
    LOCKED_PIXEL_MIN_CONF     = args.locked_conf_floor
    FRAME_MS_WARN_THRESHOLD   = args.frame_ms_warn
    PERF_ROI_PAD              = args.perf_roi_pad
    PERF_ROI_ENABLED          = not args.no_perf_roi
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
