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
"""

import os, sys, time, platform, argparse, math, shutil, zipfile, json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from models.renderers.base_renderer import FrameMetadata, VisionMetadata, BoundingBox, SegmentationMask, InspectionProgress
from models.renderers.vision_overlay_renderer import VisionOverlayRenderer

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
import logging
try:
    from ultralytics.utils import LOGGER as YOLO_LOGGER
    YOLO_LOGGER.setLevel(logging.ERROR)
except ImportError:
    pass

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

SHOW_PREVIEW      = False
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

STATE_WARMUP          = "WARMUP"
STATE_WAIT_FOR_SOCKET = "WAIT_FOR_SOCKET"
STATE_MODEL1          = "MODEL1_VALIDATION"
STATE_SKIP            = "MODEL2_SKIP"
STATE_MODEL2          = "MODEL2_VALIDATION"
STATE_WAIT_REMOVAL    = "WAIT_SOCKET_REMOVAL"
STATE_CYCLE_COMPLETE  = "CYCLE_COMPLETE"
STATE_IDLE            = "IDLE"
STATE_HAND            = "HAND"
STATE_INSPECT         = "INSPECT"
STATE_NORMAL          = "NORMAL"
STATE_ANOMALY         = "ANOMALY"
STATE_PARTIAL         = "PARTIAL"

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
                return "PARTIAL"
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
        self.aborted = 0

        self.cycle_summaries = []

    @property
    def total_cycles(self):
        return self.passed + self.failed + self.unknown + self.aborted

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

    def end_cycle(self, verdict, extra_metrics=None, is_aborted=False):
        if not self.active:
            return None

        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.active = False

        if is_aborted:
            folder_name = "ABORTED"
            filename_verdict = "ABORTED"
            verdict = "ABORTED"
        else:
            folder_name = verdict if verdict in ("NORMAL", "ANOMALY", "UNKNOWN") else "UNKNOWN"
            filename_verdict = verdict

        dest_dir = os.path.join(self.resolved_output_dir, folder_name)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        final_name = f"{self.video_stem}_{filename_verdict}_cycle{self.cycle_no:03d}.mp4"
        final_path = os.path.join(dest_dir, final_name)
        if Path(final_path).exists():
            Path(final_path).unlink()
        shutil.move(self.temp_path, final_path)

        if verdict == "NORMAL":
            self.passed += 1
        elif verdict == "ANOMALY":
            self.failed += 1
        elif verdict == "ABORTED":
            self.aborted += 1
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
        print(f"  ABORTED           : {self.aborted}")
        print("█" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED def draw_seg_overlay(frame, pred_map, alpha=None):
# DEPRECATED     from models.renderers.current_renderer import draw_seg_overlay
# DEPRECATED     return draw_seg_overlay(frame, pred_map, alpha)
# DEPRECATED 
# DEPRECATED 

# DEPRECATED def draw_raw_argmax_fallback(frame, raw_pred):
# DEPRECATED     from models.renderers.current_renderer import draw_raw_argmax_fallback
# DEPRECATED     return draw_raw_argmax_fallback(frame, raw_pred)
# DEPRECATED 
# DEPRECATED 
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED #  [FIX-I53] SOCKET / NO-SOCKET BOUNDING BOX — FILLED
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════

# DEPRECATED def draw_socket_box(frame, hit, fill_alpha=None):
# DEPRECATED     from models.renderers.current_renderer import draw_socket_box
# DEPRECATED     return draw_socket_box(frame, hit, fill_alpha)
# DEPRECATED 
# DEPRECATED 

# DEPRECATED def draw_final_verdict_overlay(frame, verdict, cycle_no=None, stats=None):
# DEPRECATED     from models.renderers.current_renderer import draw_final_verdict_overlay
# DEPRECATED     return draw_final_verdict_overlay(frame, verdict, cycle_no, stats)
# DEPRECATED 
# DEPRECATED 

# DEPRECATED def draw_debug_overlay(frame, dbg):
# DEPRECATED     from models.renderers.current_renderer import draw_debug_overlay
# DEPRECATED     return draw_debug_overlay(frame, dbg)
# DEPRECATED 
# DEPRECATED 

# DEPRECATED def draw_production_status_bar(frame, state, cycle_no, passed, failed, unknown):
# DEPRECATED     from models.renderers.current_renderer import draw_production_status_bar
# DEPRECATED     return draw_production_status_bar(frame, state, cycle_no, passed, failed, unknown)
# DEPRECATED 
# DEPRECATED 

# DEPRECATED def draw_hud(frame, fps, frame_idx, state, socket_hit,
# DEPRECATED              status_dict, order_status, detected_seq,
# DEPRECATED              anomaly_counter=0, hand_in_roi=False,
# DEPRECATED              warmup_frame=0, warmup_retry=0,
# DEPRECATED              vote_counter=None, infer_frames=0,
# DEPRECATED              seq_stable_ctr=0, cycle_no=0, frame_ms=0.0, frame_ms_avg=0.0,
# DEPRECATED              mask_is_locked=False):
# DEPRECATED     out  = frame.copy()
# DEPRECATED     H, W = out.shape[:2]
# DEPRECATED     S    = W / 1280.0
# DEPRECATED     PAD  = max(10, int(12 * S))
# DEPRECATED     FS_XS = max(0.40, 0.42 * S);  FS_SM = max(0.48, 0.52 * S)
# DEPRECATED     FS_MD = max(0.58, 0.62 * S);  FS_LG = max(0.70, 0.76 * S)
# DEPRECATED     TK1   = max(1, int(S));        TK2   = max(1, int(2 * S))
# DEPRECATED     ROW   = max(26, int(28 * S)); DOT   = max(5, int(6 * S))
# DEPRECATED 
# DEPRECATED     TOP_OFFSET = max(70, int(78 * S))
# DEPRECATED 
# DEPRECATED     LP_W  = max(280, int(300 * S));  LP_H = PAD * 2 + ROW * 10 + 8
# DEPRECATED     LP_X, LP_Y = 10, 10 + TOP_OFFSET
# DEPRECATED     RP_W  = max(300, int(325 * S));  RP_H = PAD * 2 + ROW * 9 + 20
# DEPRECATED     RP_X  = W - RP_W - 10;  RP_Y = 10 + TOP_OFFSET
# DEPRECATED 
# DEPRECATED     ovl = out.copy()
# DEPRECATED     for (px, py, pw, ph) in [(LP_X, LP_Y, LP_W, LP_H),
# DEPRECATED                              (RP_X, RP_Y, RP_W, RP_H)]:
# DEPRECATED         cv2.rectangle(ovl, (px, py), (px + pw, py + ph), (14, 14, 14), -1)
# DEPRECATED     cv2.addWeighted(ovl, 0.72, out, 0.28, 0, out)
# DEPRECATED 
# DEPRECATED     sty = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
# DEPRECATED     for (px, py, pw, ph), bc in [
# DEPRECATED         ((LP_X, LP_Y, LP_W, LP_H), sty["border"]),
# DEPRECATED         ((RP_X, RP_Y, RP_W, RP_H), (70, 70, 70))
# DEPRECATED     ]:
# DEPRECATED         cv2.rectangle(out, (px, py), (px + pw, py + ph), bc, 1)
# DEPRECATED 
# DEPRECATED     lx = LP_X + PAD;  ly = LP_Y + PAD + ROW - 4;  vx = lx + max(60, int(64 * S))
# DEPRECATED     _put(out, "FPS",   lx, ly, FS_XS, (120, 120, 120), TK1)
# DEPRECATED     _put(out, f"{fps:5.1f}", vx, ly, FS_LG, (220, 220, 220), TK2);  ly += ROW + 4
# DEPRECATED     _put(out, "FRAME", lx, ly, FS_XS, (120, 120, 120), TK1)
# DEPRECATED     _put(out, f"{frame_idx:06d}", vx, ly, FS_MD, (200, 200, 200), TK1);  ly += ROW + 4
# DEPRECATED 
# DEPRECATED     ms_col = (60, 60, 255) if frame_ms_avg >= FRAME_MS_WARN_THRESHOLD else (200, 200, 200)
# DEPRECATED     _put(out, "MS",    lx, ly, FS_XS, (120, 120, 120), TK1)
# DEPRECATED     _put(out, f"{frame_ms:5.1f} (avg {frame_ms_avg:5.1f})",
# DEPRECATED          vx, ly, FS_SM, ms_col, TK1);  ly += ROW + 6
# DEPRECATED 
# DEPRECATED     chip = _CHIP_LABEL.get(state, state)
# DEPRECATED     (cw, ch), bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, FS_SM, TK1)
# DEPRECATED     cpx, cpy = max(10, int(11 * S)), max(6, int(7 * S))
# DEPRECATED     cx1, cy1 = lx, ly;  cx2, cy2 = cx1 + cw + cpx * 2, cy1 + ch + bl + cpy * 2
# DEPRECATED     cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["chip_bg"], -1)
# DEPRECATED     cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["border"], 1)
# DEPRECATED     _put(out, chip, cx1 + cpx, cy1 + cpy + ch, FS_SM, sty["chip_fg"], TK1)
# DEPRECATED     if 0 < anomaly_counter < N_ANOMALY_CONFIRM:
# DEPRECATED         _put(out, f"({anomaly_counter}/{N_ANOMALY_CONFIRM})",
# DEPRECATED              cx2 + 6, cy1 + cpy + ch, FS_XS, (160, 80, 80), TK1)
# DEPRECATED     ly = cy2 + 6
# DEPRECATED 
# DEPRECATED     if state == STATE_WARMUP and WARMUP_FRAMES > 0:
# DEPRECATED         bw   = cx2 - cx1;  bh = max(6, int(7 * S))
# DEPRECATED         prog = min(warmup_frame / (WARMUP_FRAMES * (warmup_retry + 1)), 1.0)
# DEPRECATED         cv2.rectangle(out, (cx1, ly), (cx1 + bw, ly + bh), (60, 60, 60), -1)
# DEPRECATED         cv2.rectangle(out, (cx1, ly), (cx1 + int(bw * prog), ly + bh),
# DEPRECATED                       (180, 140, 20), -1);  ly += bh + 4
# DEPRECATED         if warmup_retry > 0:
# DEPRECATED             _put(out, f"retry {warmup_retry}/{MAX_WARMUP_RETRIES}",
# DEPRECATED                  cx1, ly + ROW - 6, FS_XS, (160, 120, 40), TK1);  ly += ROW
# DEPRECATED 
# DEPRECATED     if state == STATE_HAND:
# DEPRECATED         _put(out, "DETECTIONS PAUSED (hand in ROI)",
# DEPRECATED              lx, ly + ROW - 6, FS_XS, _ROI_COL_AMBER, TK1);  ly += ROW
# DEPRECATED     else:
# DEPRECATED         _put(out, f"LIVE INFER   [{infer_frames}f]",
# DEPRECATED              lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
# DEPRECATED         if mask_is_locked:
# DEPRECATED             _put(out, "MASK: LOCKED", lx + 190, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
# DEPRECATED         else:
# DEPRECATED             _put(out, "MASK: REFRESHED", lx + 190, ly + ROW - 6, FS_XS, (0, 165, 255), TK1)
# DEPRECATED         ly += ROW
# DEPRECATED 
# DEPRECATED         if seq_stable_ctr > 0:
# DEPRECATED             sc_col = (60, 220, 60) if seq_stable_ctr >= MIN_SEQ_STABLE else (160, 160, 40)
# DEPRECATED             _put(out, f"SEQ STABLE  {seq_stable_ctr}/{MIN_SEQ_STABLE}",
# DEPRECATED                  lx, ly + ROW - 6, FS_XS, sc_col, TK1);  ly += ROW
# DEPRECATED 
# DEPRECATED     if vote_counter is not None and vote_counter.total > 0:
# DEPRECATED         ly += 2
# DEPRECATED         _put(out, f"OK : {vote_counter.normal_votes}",
# DEPRECATED              lx, ly + ROW - 6, FS_XS, (60, 220, 60), TK1);  ly += ROW
# DEPRECATED         _put(out, f"AN : {vote_counter.anomaly_votes}",
# DEPRECATED              lx, ly + ROW - 6, FS_XS, (80, 80, 230), TK1);  ly += ROW
# DEPRECATED 
# DEPRECATED     rx = RP_X + PAD;  ry = RP_Y + PAD
# DEPRECATED     _put(out, "INSPECTION STATUS", rx, ry + ROW - 6, FS_XS, (100, 100, 100), TK1)
# DEPRECATED     ry += ROW + 4
# DEPRECATED     cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1);  ry += 8
# DEPRECATED 
# DEPRECATED     if hand_in_roi:
# DEPRECATED         hc = _ROI_COL_AMBER;  ht = "HAND     IN ROI"
# DEPRECATED         cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, hc, -1)
# DEPRECATED         cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, (255, 255, 255), 1)
# DEPRECATED     else:
# DEPRECATED         hc = (70, 70, 70);  ht = "HAND     CLEAR"
# DEPRECATED         cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, hc, -1)
# DEPRECATED     _put(out, ht, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, hc, TK1);  ry += ROW + 4
# DEPRECATED             dc  = (50, 50, 50)
# DEPRECATED             tc3 = (140, 140, 140)
# DEPRECATED             cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
# DEPRECATED             _put(out, f"{TUBE_LABELS[ci]}   PAUSED",
# DEPRECATED                  rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4
# DEPRECATED     else:
# DEPRECATED         for ci in (2, 3, 4):
# DEPRECATED             pres = status_dict.get(ci, "Absent") == "Present"
# DEPRECATED             dc   = CLASS_INFO[ci][1] if pres else (50, 50, 50)
# DEPRECATED             tc3  = (180, 255, 180) if pres else (90, 90, 90)
# DEPRECATED             cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
# DEPRECATED             _put(out, f"{TUBE_LABELS[ci]}   {'PRESENT' if pres else 'ABSENT'}",
# DEPRECATED                  rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4
# DEPRECATED 
# DEPRECATED     cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1);  ry += 8
# DEPRECATED 
# DEPRECATED     if hand_in_roi:
# DEPRECATED         seq_str = "-"
# DEPRECATED     else:
# DEPRECATED         seq_str = " > ".join(TUBE_SHORT[c] for c in detected_seq) if detected_seq else "-"
# DEPRECATED     _put(out, f"SEQ  {seq_str}", rx, ry + ROW - 6, FS_SM, (160, 160, 160), TK1);  ry += ROW + 6
# DEPRECATED 
# DEPRECATED     if hand_in_roi:
# DEPRECATED         _put(out, "INFERENCE   PAUSED", rx, ry + ROW - 4, FS_MD, _ROI_COL_AMBER, TK2)
# DEPRECATED     else:
# DEPRECATED         vcol, vtxt = {
# DEPRECATED             "OK":      ((60, 220, 60),  "ORDER  [OK]  NORMAL"),
# DEPRECATED             "ANOMALY": ((40, 40, 230),  "ORDER  [!!]  ANOMALY"),
# DEPRECATED             "PARTIAL": ((30, 160, 200), "ORDER  [??]  PARTIAL"),
# DEPRECATED         }.get(order_status, ((100, 100, 100), "ORDER  ---  N/A"))
# DEPRECATED         _put(out, vtxt, rx, ry + ROW - 4, FS_MD, vcol, TK2)
# DEPRECATED     return out
# DEPRECATED 
# DEPRECATED 
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED #  OUTPUT PATH HELPERS
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED def get_verdict_dir(out_dir, verdict):
# DEPRECATED     folder = verdict if verdict in ("NORMAL", "ANOMALY", "UNKNOWN") else "UNKNOWN"
# DEPRECATED     d      = os.path.join(out_dir, folder)
# DEPRECATED     Path(d).mkdir(parents=True, exist_ok=True)
# DEPRECATED     return d, folder
# DEPRECATED 
# DEPRECATED 
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED #  CORE PROCESSOR — SINGLE VIDEO, MULTI-CYCLE
# DEPRECATED # ══════════════════════════════════════════════════════════════════════════════

def process_video_cycles(video_path, resolved_output_dir, seg_net,
                         yolo_socket, yolo_pose, print_summary,
                         enable_debug=False, yield_mode=False,
                         renderer=None,
                         fps_src=30.0,
                         src_w=1920,
                         src_h=1080,
                         config=None):
    if config is None: config = {}
    
    model1_frame_count       = config.get("model1_frame_count", 30)
    model1_pass_frames       = config.get("model1_pass_frames", 28)
    model2_start_skip_frame  = config.get("model2_start_skip_frame", 10)
    model2_frame_count       = config.get("model2_frame_count", 20)
    model2_pass_frames       = config.get("model2_pass_frames", 18)
    socket_absent_frames     = config.get("socket_absent_frames", 10)
    socket_loss_abort_frames = config.get("socket_loss_abort_frames", 15)
    if renderer is None:
        from models.renderers.renderer_factory import get_renderer
        renderer = get_renderer("current")
    if yield_mode:
        cap = None
        # fps_src, src_w, src_h are provided as arguments
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {video_path}");  return None

    if not yield_mode:
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

    app_warmup_done    = False
    app_warmup_frames  = 10

    sm_state           = STATE_WARMUP
    m1_pass_count      = 0
    m1_loss_count      = 0
    m2_skip_count      = 0
    m2_pass_count      = 0
    removal_count      = 0

    ui_progress = {"current": 0, "target": 1, "label": "Loading"}
    ui_models = {
        "socket": {"state": "LOADING", "result": "-", "progress": 0},
        "tube": {"state": "LOADING", "result": "-", "progress": 0}
    }

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
    last_metadata = None
    
    vision_renderer = VisionOverlayRenderer()

    peak_status = EMPTY_STAT.copy()
    peak_seq    = []
    peak_order  = "N/A"

    _prev_gray_ref[0] = None

    def _finalize_cycle(aborted=False):
        nonlocal last_vis, last_metadata, renderer, peak_order, peak_status, peak_seq, invisible_roi
        if not cycles.active:
            return
            
        if aborted:
            final_verdict = "ABORTED"
            peak_order = "N/A"
            peak_status = {}
            peak_seq = []
            invisible_roi = None
        else:
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
        if last_vis is not None and last_metadata is not None:
            import dataclasses
            final_metadata = dataclasses.replace(
                last_metadata,
                is_final=True,
                verdict=final_verdict,
                counters={**last_metadata.counters, "stats_card": stats_card}
            )
            try:
                card = renderer.render(last_vis, final_metadata)
            except Exception as e:
                print(f"[WARN] Final render failed: {e}")
                card = last_vis
            if not aborted:
                cycles.hold_final_frame(card)
            if SHOW_PREVIEW and not yield_mode:
                cv2.imshow(WIN, card);
            if not yield_mode and not aborted: cv2.waitKey(int(VERDICT_HOLD_SEC * 1000))

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
        summary = cycles.end_cycle(final_verdict, extra_metrics=extra, is_aborted=aborted)
        if summary and print_summary:
            print("\n" + "=" * 62 +
                  f"\n CYCLE #{summary['cycle_no']:03d} SUMMARY\n" + "=" * 62)
            for k, v in summary.items():
                print(f"  {k:<26}: {v}")
            print("=" * 62 + "\n")
        if summary:
            append_to_excel(summary, resolved_output_dir)

    aborted = False

    current_dbg = {}
    while True:
        show_roi_flag = False  # Always initialised; overridden below in yield_mode
        if yield_mode:
            payload = yield
            if payload is None:
                aborted = True
                break
            if payload == "EOF":
                aborted = False
                break
            if isinstance(payload, tuple):
                frame, show_roi_flag = payload
            else:
                frame = payload
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        effective_debug = enable_debug or show_roi_flag
        frame_idx += 1
        t0  = time.perf_counter()
        vis = frame.copy()
        has_tubes = False
        
        socket_ms = 0.0
        tube_ms = 0.0

        t_sock = time.perf_counter()
        sock_hit = detect_socket(yolo_socket, frame, YOLO_SOCKET_CONF)
        socket_ms = (time.perf_counter() - t_sock) * 1000.0
        hand_hit = None
        if invisible_roi is not None:
            hand_hit = detect_hand_in_roi(yolo_pose, frame, invisible_roi, YOLO_POSE_CONF)
        socket_now    = sock_hit is not None and sock_hit["class"] == CLS_SOCKET
        no_socket_now = sock_hit is not None and sock_hit["class"] == CLS_NO_SOCKET

        if socket_now:
            x1, y1, x2, y2    = sock_hit["bbox"]
            last_socket_centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            last_socket_bbox_size = (x2 - x1, y2 - y1)

        # --- NATIVE SEQUENTIAL STATE MACHINE ---
        hand_in_roi = bool(hand_hit)
        socket_now  = sock_hit is not None and sock_hit["class"] == CLS_SOCKET

        
        if socket_now:
            invisible_roi = build_roi(frame.shape, sock_hit["bbox"])
            x1, y1, x2, y2 = sock_hit["bbox"]
            scx, scy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            current_dbg["scx"] = scx
            current_dbg["scy"] = scy
            current_dbg["radius"] = NEAREST_SEARCH_RADIUS
        
        def _safe_pct(c, t): return min(100, int((c / max(1, t)) * 100))

        run_segmentation = False

        if not app_warmup_done:
            sm_state = STATE_WARMUP
            run_segmentation = True # warm up GPU
            app_warmup_frames -= 1
            if app_warmup_frames <= 0:
                app_warmup_done = True
                sm_state = STATE_WAIT_FOR_SOCKET
            
            ui_progress = {"current": 0, "target": 1, "label": "Warming Up GPU..."}
            ui_models = {
                "socket": {"state": "LOADING", "result": "-", "progress": 0},
                "tube": {"state": "LOADING", "result": "-", "progress": 0}
            }

        elif sm_state == STATE_WAIT_FOR_SOCKET:
            if socket_now:
                cycles.start_cycle()
                vote_counter.reset()
                seq_gate.reset()
                anomaly_gate.reset()
                latch_gate.reset()
                peak_status = EMPTY_STAT.copy()
                peak_seq    = []
                peak_order  = "N/A"
                m1_pass_count = 0
                m1_loss_count = 0
                m1_total_count = 0
                m2_total_count = 0
                sm_state = STATE_MODEL1
            
            ui_progress = {"current": 0, "target": 1, "label": "Waiting for Socket"}
            ui_models = {
                "socket": {"state": "WAITING", "result": "-", "progress": 0},
                "tube": {"state": "IDLE", "result": "-", "progress": 0}
            }
            mask_is_locked = False
            mask_frozen_pred = None
            mask_frozen_gray_roi = None

        elif sm_state == STATE_MODEL1:
            if not hand_in_roi:
                m1_total_count += 1
                if socket_now:
                    m1_pass_count += 1
                    m1_loss_count = 0
                else:
                    m1_loss_count += 1
                
                if m1_loss_count >= socket_loss_abort_frames:
                    _finalize_cycle()
                    sm_state = STATE_WAIT_FOR_SOCKET
                elif m1_pass_count >= model1_pass_frames:
                    m2_skip_count = 0
                    sm_state = STATE_SKIP
                elif m1_total_count >= model1_frame_count:
                    _finalize_cycle()
                    sm_state = STATE_WAIT_FOR_SOCKET

            label = f"Socket Validation ({m1_total_count}/{model1_frame_count})"
            if hand_in_roi: label += " (PAUSED)"
            ui_progress = {"current": m1_total_count, "target": model1_frame_count, "label": label}
            ui_models = {
                "socket": {"state": "RUNNING", "result": f"{m1_total_count} / {model1_frame_count}", "progress": _safe_pct(m1_total_count, model1_frame_count)},
                "tube": {"state": "IDLE", "result": "-", "progress": 0}
            }

        elif sm_state == STATE_SKIP:
            if not hand_in_roi:
                m2_skip_count += 1
                if m2_skip_count >= model2_start_skip_frame:
                    m2_pass_count = 0
                    sm_state = STATE_MODEL2
            
            label = f"Stabilizing Tubes ({m2_skip_count}/{model2_start_skip_frame})"
            if hand_in_roi: label += " (PAUSED)"
            ui_progress = {"current": m2_skip_count, "target": model2_start_skip_frame, "label": label}
            ui_models = {
                "socket": {"state": "COMPLETE", "result": f"{m1_total_count} / {model1_frame_count}", "progress": 100},
                "tube": {"state": "PREPARING", "result": "-", "progress": 0}
            }

        elif sm_state == STATE_MODEL2:
            run_segmentation = True
            
            label = f"Tube Validation ({m2_total_count}/{model2_frame_count})"
            if hand_in_roi: label += " (PAUSED)"
            ui_progress = {"current": m2_total_count, "target": model2_frame_count, "label": label}
            ui_models = {
                "socket": {"state": "COMPLETE", "result": f"{m1_total_count} / {model1_frame_count}", "progress": 100},
                "tube": {"state": "RUNNING", "result": f"{m2_total_count} / {model2_frame_count}", "progress": _safe_pct(m2_total_count, model2_frame_count)}
            }
            # Note: m2_pass_count and m2_total_count increment happens below after segmentation runs

        elif sm_state == STATE_WAIT_REMOVAL:
            if not hand_in_roi:
                if not socket_now:
                    removal_count += 1
                else:
                    removal_count = 0
                
                if removal_count >= socket_absent_frames:
                    _finalize_cycle()
                    sm_state = STATE_WAIT_FOR_SOCKET # or CYCLE_COMPLETE if we want, but wait for socket loop starts

            label = f"Waiting for Removal ({removal_count}/{socket_absent_frames})"
            if hand_in_roi: label += " (PAUSED)"
            ui_progress = {"current": removal_count, "target": socket_absent_frames, "label": label}
            ui_models = {
                "socket": {"state": "COMPLETE", "result": f"{m1_total_count} / {model1_frame_count}", "progress": 100},
                "tube": {"state": "COMPLETE", "result": f"{m2_total_count} / {model2_frame_count}", "progress": 100}
            }
        
        # --- EXECUTE SEGMENTATION IF PERMITTED ---
        status_dict  = EMPTY_STAT
        order_status = "N/A"
        detected_seq = []
        pred_map     = ZERO_PRED
        
        ui_status_dict  = EMPTY_STAT.copy()
        ui_detected_seq = []
        
        if run_segmentation and invisible_roi is not None:
            if hand_in_roi:
                mask_frozen_pred = None
                mask_frozen_gray_roi = None
                mask_is_locked = False
            else:
                cycle_total_frames += 1
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
                    nonlocal tube_ms
                    t_tube = time.perf_counter()
                    pm = seg_engine.infer(
                        frame, socket_centre=last_socket_centre,
                        apply_identity_lock=True)
                    tube_ms = (time.perf_counter() - t_tube) * 1000.0
                    pm = restrict_mask_to_socket_roi(
                        pm, roi_center, bbox_size=roi_bbox_size)
                    pm = apply_exclusion_zone(
                        pm, roi_center, bbox_size=roi_bbox_size)
                    return pm

                if not MASK_FREEZE_ENABLED or sm_state == STATE_WARMUP:
                    pred_map = _run_fresh_inference()
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
                        mask_frozen_pred = pred_map.copy()
                        mask_frozen_gray_roi = crop_now.copy()
                        mask_is_locked = False
                    else:
                        pred_map = mask_frozen_pred
                        mask_is_locked = True
                
                socket_bbox = sock_hit["bbox"] if sock_hit else None
                status_dict, raw_order, detected_seq, current_dbg = evaluate_tube_order(pred_map, socket_bbox, debug=True)

                has_tubes = any((pred_map == ci).any() for ci in (2, 3, 4))
                
                if sm_state == STATE_MODEL2:
                    ui_status_dict  = dict(status_dict)
                    ui_detected_seq = list(detected_seq)
                    
                    if not has_tubes and seg_engine.last_raw_pred is not None:
                        ui_status_dict, _, ui_detected_seq, fallback_dbg = evaluate_tube_order(seg_engine.last_raw_pred, socket_bbox, debug=True)
                        current_dbg = fallback_dbg
                
                    stable_order = seq_gate.update(raw_order, detected_seq)
                    gate_result  = anomaly_gate.update(stable_order)
                    order_status = gate_result
                    latch_gate.update(gate_result)

                    infer_frames += 1
                    vote_counter.record(gate_result)
                    
                    m2_total_count += 1
                    if has_tubes:
                        m2_pass_count += 1
                        
                    if m2_pass_count >= model2_pass_frames:
                        removal_count = 0
                        peak_order = order_status
                        sm_state = STATE_WAIT_REMOVAL
                    elif m2_total_count >= model2_frame_count:
                        # cycle failed to reach pass frame count!
                        _finalize_cycle()
                        sm_state = STATE_WAIT_FOR_SOCKET
                
                    if any(v == "Present" for v in status_dict.values()) and order_status != "N/A":
                        peak_status = dict(status_dict)
                        peak_seq    = list(detected_seq)
                        peak_order  = order_status
        
        elif sm_state == STATE_WAIT_REMOVAL and mask_frozen_pred is not None and not hand_in_roi:
            has_tubes = any((mask_frozen_pred == ci).any() for ci in (2, 3, 4))
            if has_tubes:
                ui_status_dict  = dict(peak_status)
                ui_detected_seq = list(peak_seq)
        
        state = sm_state
        frame_ms = (time.perf_counter() - t0) * 1000.0
        frame_ms_ema = (
            frame_ms if frame_idx == 1
            else FRAME_MS_EMA_ALPHA * frame_ms + (1 - FRAME_MS_EMA_ALPHA) * frame_ms_ema
        )
        if frame_ms > FRAME_MS_WARN_THRESHOLD:
            print(f"[WARN] f{frame_idx:05d}: slow frame {frame_ms:.1f}ms "
                  f"(avg {frame_ms_ema:.1f}ms, threshold {FRAME_MS_WARN_THRESHOLD:.0f}ms)")

        fps_ema = 0.88 * fps_ema + 0.12 / max(time.perf_counter() - t0, 1e-6)

        # -------------------------------------------------------------
        # STAGE 1: METADATA BUILDER
        # -------------------------------------------------------------
        
        # Build VisionMetadata
        vision_metadata = VisionMetadata(
            version=1,
            pred_map=pred_map,
            mask_frozen_pred=mask_frozen_pred,
            raw_pred=seg_engine.last_raw_pred,
            socket_hit=sock_hit,
            hand_in_roi=hand_in_roi,
            has_tubes=has_tubes,
            sm_state=sm_state
        )

        # Build FrameMetadata
        # strict verdict mapping according to architectural rule
        cv = vote_counter.final_verdict()
        final_verdict = cv if cv != "UNKNOWN" else ""
        
        # Build strict InspectionProgress based on user requirements
        prog = InspectionProgress(
            socket_current=m1_total_count if 'm1_total_count' in locals() else 0,
            socket_required=model1_frame_count,
            tube_current=m2_total_count if 'm2_total_count' in locals() else 0,
            tube_required=model2_frame_count,
        )
        
        metadata = FrameMetadata(
            version=1,
            fps=fps_ema,
            frame_number=frame_idx,
            elapsed_ms=frame_ms,
            avg_elapsed_ms=frame_ms_ema,
            state=state,
            cycle_number=cycles.cycle_no,
            verdict=final_verdict,
            order_status=order_status,
            socket_hit=sock_hit,
            detected_seq=ui_detected_seq,
            status_dict=ui_status_dict,
            hand_in_roi=hand_in_roi,
            mask_is_locked=mask_is_locked,
            ui_models=ui_models,
            is_final=False,
            counters={
                "passed": cycles.passed,
                "failed": cycles.failed,
                "unknown": cycles.unknown,
                "aborted": cycles.aborted,
                "anomaly_counter": anomaly_gate._count,
                "warmup_frame": warmup_frame_count,
                "warmup_retry": warmup_retry,
                "vote_counter": vote_counter,
                "infer_frames": infer_frames,
                "seq_stable_ctr": seq_gate._stable_ct
            },
            debug_info={
                "current_dbg": current_dbg if effective_debug else None
            },
            inspection_progress=prog
        )

        # -------------------------------------------------------------
        # STAGE 2: VISION OVERLAY
        # -------------------------------------------------------------
        try:
            vis = vision_renderer.render(vis, vision_metadata)
            vis_for_disk = vis.copy()
        except Exception as e:
            print(f"[WARN] VisionRenderer failed on frame {frame_idx}: {e}")

        if show_roi_flag:
            if current_dbg:
                scx, scy = current_dbg.get("scx", 0), current_dbg.get("scy", 0)
                R = current_dbg.get("radius", 0)
                if R > 0:
                    cv2.circle(vis, (int(scx), int(scy)), int(R), (255, 0, 255), 1)
                    cv2.circle(vis, (int(scx), int(scy)), 4, (255, 0, 255), -1)
                for ci, (ax, ay) in current_dbg.get("anchors", {}).items():
                    cv2.circle(vis, (int(ax), int(ay)), 4, (0, 255, 255), -1)
                    cv2.line(vis, (int(scx), int(scy)), (int(ax), int(ay)), (0, 255, 255), 1)
                    d = current_dbg.get("nearest_dist", {}).get(ci, 0)
                    a = current_dbg.get("angles_deg", {}).get(ci, 0)
                    cv2.putText(vis, f"T{ci}: d={d:.1f} a={a:.0f}", (int(ax)+5, int(ay)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                gaps = current_dbg.get("gaps_deg", [])
                for i, (c1, c2, gap) in enumerate(gaps):
                    cv2.putText(vis, f"Gap T{c1}->T{c2}: {gap:.0f} deg", (10, 320 + i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # STAGE 3: UI HUD OVERLAY
        # -------------------------------------------------------------
        vis_without_hud = vis.copy()
        try:
            vis = renderer.render(vis, metadata)
            vis_for_disk = vis.copy()
        except Exception as e:
            print(f"[WARN] Renderer failed on frame {frame_idx}: {e}. Falling back to unrendered frame.")
            # Optionally fallback to CurrentRenderer or just continue with unrendered vis

        cycles.write(vis_for_disk)
        last_vis = vis_without_hud
        last_metadata = metadata

        if yield_mode:
            yield {
                "state": sm_state,
                "progress": ui_progress,
                "models": ui_models,
                "inspection_progress": {
                    "socket_current": prog.socket_current,
                    "socket_required": prog.socket_required,
                    "tube_current": prog.tube_current,
                    "tube_required": prog.tube_required
                } if prog else None,
                "frame": vis_without_hud,
                "socket_box": sock_hit["bbox"] if sock_hit else None,
                "invisible_roi": invisible_roi,
                "socket_hit": sock_hit,
                "hand_hit": hand_in_roi,
                "detected_seq": ui_detected_seq,
                "anomaly_ratio": vote_counter.anomaly_votes / max(vote_counter.total, 1) if vote_counter else 0,
                "normal_votes": vote_counter.normal_votes,
                "anomaly_votes": vote_counter.anomaly_votes,
                "final_verdict": peak_order if peak_order != "N/A" else "UNKNOWN",
                "cycle_no": cycles.cycle_no if cycles else 0,
                "active_cycle": cycles.active if cycles else False,
                "passed": cycles.passed if cycles else 0,
                "failed": cycles.failed if cycles else 0,
                "aborted": cycles.aborted if cycles else 0,
                "frame_idx": frame_idx,
                "fps": float(fps_ema),
                "frame_ms": float(frame_ms),
                "frame_ms_avg": float(frame_ms_ema),
                "socket_ms": float(socket_ms),
                "tube_ms": float(tube_ms),
            }            
        if SHOW_PREVIEW and not yield_mode:
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
        _finalize_cycle(aborted=aborted)

    if cap:
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
    from app.services.excel_sync_service import ExcelSyncService
    # In legacy standalone contexts, batch_id and video_run_id may not be known.
    # The sync service accepts None for these.
    ExcelSyncService.get_instance().queue_row(
        batch_id=None,
        video_run_id=None,
        excel_dir=excel_dir,
        run_metrics=run_metrics
    )


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
    if USE_HALF:
        import torch
        gpu_name = torch.cuda.get_device_name(0).lower()
        if any(x in gpu_name for x in ["1050", "1060", "1070", "1080", "1650", "1660"]):
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
    ckpt = torch.load(seg_model_path, map_location=DEVICE, weights_only=False)
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
                    "into each other")
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