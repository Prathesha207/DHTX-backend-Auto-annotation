#!/usr/bin/env python3
"""
Diagnostic Inference Engine — v50_MergedSingleVideo (Modularized)
==================================================================
Base inference engine : v45 (live inference every frame, no mask freeze,
                         FIX-I1..FIX-I8 boundary/threshold/morphology fixes)
Tube-order logic       : v49 FIX-I18 — Nearest-Pixel Angular Gate
Cycle/production logic : v49 — CycleManager, production status bar,
                         cycle-based Excel logging, single-video / multi-cycle
                         processing loop

Architecture:
- Modular Configuration System (PipelineConfig & component configs)
- Dedicated Model Execution Runners:
    1. run_model_socket (Socket Detection via YOLO / YOLO-OBB)
    2. run_hand_model (Operator Hand in ROI Detection via Optical Flow + YOLO Pose)
    3. run_model_tube_segmentation (Tube Segmentation via UNet++ with EMA,
       Identity Lock, Conflict-Resolved Morphology, and ROI Gating)
- Modular ROI Subsystem (Socket ROI, Perf ROI, Shape Masks, Exclusion Zones, Composite Gates)
- Multi-cycle processing loop with real-time HUD overlays & Excel logging
"""

import os
import sys
import time
import platform
import argparse
import math
import shutil
import zipfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import albumentations as A
# pyrefly: ignore [missing-import]
from albumentations.pytorch import ToTensorV2
# pyrefly: ignore [missing-import]
import segmentation_models_pytorch as smp

# [FIX-I52] Ensure OpenCV's internal thread pool is properly initialized.
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 4))
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

if platform.system() == "Windows":
    _DEFAULT_BASE     = r"C:\Users\sonar\Desktop\autodistill-grounded-sam-2"
    _DEFAULT_OUT_BASE = r"C:\Users\sonar\Desktop\UNet++_results"
else:
    _DEFAULT_BASE     = "/mnt/c/Users/sonar/Desktop/autodistill-grounded-sam-2"
    _DEFAULT_OUT_BASE = "/mnt/c/Users/sonar/Desktop/UNet++_results"


@dataclass
class ModelPathsConfig:
    """Paths to models, input video, and output directories."""
    video_path: str = os.path.join(_DEFAULT_BASE, "input_videos", "shift_recording.mp4")
    seg_model_path: str = os.path.join(
        _DEFAULT_BASE, "outputs", "finetune_augmentation_v2",
        "run_20260623_140144", "best_model_finetuned_v2.pth"
    )
    socket_model_path: str = ""
    hand_pose_path: str = "yolov8n-pose.pt"
    out_base: str = _DEFAULT_OUT_BASE


@dataclass
class DeviceConfig:
    """Device, precision, and GPU benchmark settings."""
    device: str = "cpu"
    require_gpu: bool = True
    use_half: bool = True
    gpu_warmup_hw: Tuple[int, int] = (720, 1280)
    opt_flow_downscale: float = 0.5


@dataclass
class SocketConfig:
    """Socket detection parameters."""
    conf_thr: float = 0.35
    cls_no_socket: int = 0
    cls_socket: int = 1
    reset_grace_frames: int = 45
    box_fill_alpha: float = 0.28


@dataclass
class HandConfig:
    """Hand / pose detection and motion gate parameters."""
    pose_conf_thr: float = 0.40
    motion_thr: float = 3.5
    motion_frac: float = 0.06


@dataclass
class ROIConfig:
    """Socket ROI, Perf ROI, and Mask ROI Gate configuration."""
    # Socket Bounding ROI Padding
    pad_x: int = 180
    pad_y: int = 160

    # Perf ROI (Bounds expensive CPU morphology/flow work)
    perf_roi_enabled: bool = True
    perf_roi_pad: int = 550

    # Mask ROI Keep-Region Gate
    mask_roi_classes: Tuple[int, ...] = (2, 3, 4)
    mask_roi_shape: str = "quad_ellipse"
    mask_roi_radius: int = 160
    mask_roi_radius_x: int = 160
    mask_roi_radius_y: int = 330
    mask_roi_offset_x: int = 0
    mask_roi_offset_y: int = -120

    mask_roi_radius_up: int = 330
    mask_roi_radius_down: int = 120
    mask_roi_radius_left: int = 160
    mask_roi_radius_right: int = 220

    mask_roi_auto_scale: bool = True
    mask_roi_mult_up: float = 1.3
    mask_roi_mult_down: float = 0.65
    mask_roi_mult_left: float = 0.75
    mask_roi_mult_right: float = 1.05
    mask_roi_polygon: Optional[List[Tuple[int, int]]] = None

    # Hard Mask Exclusion Zone (Keep-Out)
    mask_exclude_enabled: bool = True
    mask_exclude_classes: Tuple[int, ...] = (2, 3, 4)
    mask_exclude_auto_scale: bool = True
    mask_exclude_offset_mult_x: float = -1.3
    mask_exclude_offset_mult_y: float = -0.9
    mask_exclude_radius_mult_x: float = 1.1
    mask_exclude_radius_mult_y: float = 1.1
    mask_exclude_offset_x: int = -250
    mask_exclude_offset_y: int = -150
    mask_exclude_radius_x: int = 200
    mask_exclude_radius_y: int = 200


@dataclass
class SegmentationConfig:
    """Tube segmentation parameters, thresholds, and post-processing."""
    num_classes: int = 5
    img_size: Tuple[int, int] = (512, 512)
    overlay_alpha: float = 0.90
    show_mask_overlay: bool = True
    ema_alpha: float = 0.20
    boundary_sharpening: bool = False
    sharpen_temp: float = 0.85

    class_conf_thr: Dict[int, float] = field(default_factory=lambda: {
        1: 0.35,  # device
        2: 0.25,  # tube_blue (yellow overlay)
        3: 0.25,  # trans_mid_tube (blue overlay)
        4: 0.25,  # trans_end_tube (pink overlay)
    })

    warmup_class_conf_thr: Dict[int, float] = field(default_factory=lambda: {
        1: 0.25,
        2: 0.20,
        3: 0.20,
        4: 0.20,
    })

    # Identity Hysteresis Lock + Flow Tracking
    identity_hysteresis_enabled: bool = True
    identity_ema_alpha: float = 0.15
    tube_identity_margin: float = 0.06
    tube_present_thr: float = 0.15
    lock_engage_mode: str = "post_warmup"
    locked_pixel_min_conf: float = 0.08

    # Morphology and Dilation
    mask_dilate_enabled: bool = True
    mask_dilate_sz: int = 6
    mask_dilate_classes: Tuple[int, ...] = (2, 3, 4)
    mask_close_enabled: bool = True
    mask_close_sz: int = 6

    min_tube_px: int = 40
    min_area_px: int = 120
    morph_kernel_sz: int = 5
    persist_frames: int = 1

    # Channels
    in_channels: int = 3
    use_radial_channel: bool = False

    # Mask Freeze
    mask_freeze_enabled: bool = True
    mask_freeze_motion_thr: float = 2.5
    mask_freeze_motion_frac: float = 0.04

    # Debug & HSV
    mask_debug: bool = False
    use_hsv_gate: bool = False
    hsv_gates: Dict[int, Tuple[int, int, int, bool]] = field(default_factory=lambda: {
        2: (80,  105, 55,  False),
        3: (100, 135, 45,  False),
        4: (140, 180, 45,  True),
    })
    hsv_ema_decay: float = 0.40


@dataclass
class InspectionConfig:
    """Warmup, sequencing, stability, and verdict evaluation settings."""
    warmup_frames: int = 20
    max_warmup_retries: int = 3
    verdict_thr: float = 0.50
    verdict_hold_sec: float = 2.0
    min_seq_stable: int = 3
    expected_seq: List[int] = field(default_factory=lambda: [2, 3, 4])
    nearest_search_radius: int = 350
    n_anomaly_confirm: int = 4
    latch_frames: int = 10


@dataclass
class UIConfig:
    """Display, HUD, and logging settings."""
    show_preview: bool = True
    max_excel_retries: int = 5
    frame_ms_warn_threshold: float = 150.0
    frame_ms_ema_alpha: float = 0.12


@dataclass
class PipelineConfig:
    """Master composite configuration holding all sub-configs."""
    paths: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    socket: SocketConfig = field(default_factory=SocketConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Loads configuration from a YAML file."""
        import yaml
        cfg = cls()
        if os.path.isfile(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg.update_from_dict(data)
            print(f"[CONFIG] Loaded configuration from {yaml_path}")
        else:
            print(f"[WARN] Configuration file not found: {yaml_path}, using defaults.")
        return cfg

    def update_from_dict(self, d: Dict[str, Any]):
        """Updates nested configuration attributes from a dictionary."""
        if not isinstance(d, dict):
            return
        for section, sub_dict in d.items():
            if hasattr(self, section) and isinstance(sub_dict, dict):
                target_obj = getattr(self, section)
                for k, v in sub_dict.items():
                    if hasattr(target_obj, k):
                        # Handle type conversions if needed
                        current_val = getattr(target_obj, k)
                        if isinstance(current_val, tuple) and isinstance(v, list):
                            v = tuple(v)
                        setattr(target_obj, k, v)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS & DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════

CLASS_INFO = {
    0: ("background",     (0,   0,   0),   False),
    1: ("device",         (0,   128, 0),   False),
    2: ("tube_blue",      (0,   255, 255), True),   # Yellow overlay
    3: ("trans_mid_tube", (255, 0,   0),   True),   # Blue overlay
    4: ("trans_end_tube", (255, 0,   255), True),   # Pink overlay
}
TUBE_LABELS = {2: "Yellow  tube_blue", 3: "Blue    mid-tube", 4: "Pink    end-tube"}
TUBE_SHORT  = {2: "Yel", 3: "Blu", 4: "Pnk"}

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

_ROI_COL_AMBER = (0, 165, 255)
_NORM_MEAN = None
_NORM_STD  = None
_YOLO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo")


# ══════════════════════════════════════════════════════════════════════════════
#  2. ROI SUBSYSTEM (SEPARATE & MODULAR FUNCTIONS)
# ══════════════════════════════════════════════════════════════════════════════

def build_socket_roi(frame_shape: Tuple[int, ...], bbox: Tuple[int, int, int, int],
                     pad_x: int = 180, pad_y: int = 160) -> Tuple[int, int, int, int]:
    """
    Builds an expanded bounding ROI rectangle around the detected socket bbox.
    Clamps coordinates strictly within frame dimensions.
    """
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(W - 1, x2 + pad_x), min(H - 1, y2 + pad_y))


def compute_socket_roi_geometry(sock_hit: Optional[Dict[str, Any]],
                                last_centre: Optional[Tuple[float, float]] = None,
                                last_bbox_size: Optional[Tuple[int, int]] = None
                                ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]]]:
    """
    Computes socket center (cx, cy) and bbox dimensions (w, h) from sock_hit or fallback history.
    """
    if sock_hit is not None and "bbox" in sock_hit:
        x1, y1, x2, y2 = sock_hit["bbox"]
        centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        bbox_size = (x2 - x1, y2 - y1)
        return centre, bbox_size
    return last_centre, last_bbox_size


def build_perf_roi(center: Optional[Tuple[float, float]],
                   frame_shape: Tuple[int, ...],
                   pad: int = 550,
                   enabled: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    [FIX-I45] Computes a performance bounding crop ROI around the socket center
    to accelerate expensive CPU morphology and optical flow operations.
    """
    if not enabled or center is None:
        return None
    h, w = frame_shape[:2]
    cx, cy = center
    x1 = max(0, int(cx - pad)); x2 = min(w, int(cx + pad))
    y1 = max(0, int(cy - pad)); y2 = min(h, int(cy + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def create_quad_ellipse_roi_mask(frame_shape: Tuple[int, ...],
                                 center: Tuple[float, float],
                                 radius_up: float,
                                 radius_down: float,
                                 radius_left: float,
                                 radius_right: float) -> np.ndarray:
    """Creates a boolean mask for asymmetric quad-ellipse ROI."""
    h, w = frame_shape[:2]
    cx, cy = center
    rx_max = max(radius_left, radius_right)
    ry_max = max(radius_up, radius_down)
    x1 = max(0, int(cx - rx_max)); x2 = min(w, int(cx + rx_max) + 1)
    y1 = max(0, int(cy - ry_max)); y2 = min(h, int(cy + ry_max) + 1)

    roi_mask = np.zeros((h, w), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return roi_mask

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dx = xx - cx
    dy = yy - cy
    rx = np.where(dx >= 0, radius_right, radius_left).astype(np.float32)
    ry = np.where(dy >= 0, radius_down, radius_up).astype(np.float32)
    norm2 = (dx / np.maximum(rx, 1e-6)) ** 2 + (dy / np.maximum(ry, 1e-6)) ** 2
    roi_mask[y1:y2, x1:x2] = norm2 <= 1.0
    return roi_mask


def create_ellipse_roi_mask(frame_shape: Tuple[int, ...],
                            center: Tuple[float, float],
                            radius_x: float,
                            radius_y: float) -> np.ndarray:
    """Creates a boolean mask for symmetric ellipse ROI."""
    h, w = frame_shape[:2]
    cx, cy = center
    x1 = max(0, int(cx - radius_x)); x2 = min(w, int(cx + radius_x) + 1)
    y1 = max(0, int(cy - radius_y)); y2 = min(h, int(cy + radius_y) + 1)

    roi_mask = np.zeros((h, w), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return roi_mask

    yy, xx = np.mgrid[y1:y2, x1:x2]
    norm2 = ((xx - cx) / max(radius_x, 1e-6)) ** 2 + ((yy - cy) / max(radius_y, 1e-6)) ** 2
    roi_mask[y1:y2, x1:x2] = norm2 <= 1.0
    return roi_mask


def create_circle_roi_mask(frame_shape: Tuple[int, ...],
                           center: Tuple[float, float],
                           radius: float) -> np.ndarray:
    """Creates a boolean mask for circular ROI."""
    h, w = frame_shape[:2]
    cx, cy = center
    x1 = max(0, int(cx - radius)); x2 = min(w, int(cx + radius) + 1)
    y1 = max(0, int(cy - radius)); y2 = min(h, int(cy + radius) + 1)

    roi_mask = np.zeros((h, w), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return roi_mask

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    roi_mask[y1:y2, x1:x2] = dist2 <= (radius ** 2)
    return roi_mask


def create_polygon_roi_mask(frame_shape: Tuple[int, ...],
                            center: Tuple[float, float],
                            polygon_pts: List[Tuple[int, int]]) -> np.ndarray:
    """Creates a boolean mask from polygon vertices relative to socket center."""
    h, w = frame_shape[:2]
    cx, cy = center
    if not polygon_pts:
        return np.ones((h, w), dtype=bool)
    pts = np.array([[cx + dx, cy + dy] for dx, dy in polygon_pts], dtype=np.int32)
    roi_mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_u8, [pts], 1)
    return roi_mask_u8.astype(bool)


def restrict_mask_to_socket_roi(pred_map: np.ndarray,
                                center: Optional[Tuple[float, float]],
                                bbox_size: Optional[Tuple[int, int]] = None,
                                roi_config: Optional[ROIConfig] = None,
                                **kwargs) -> np.ndarray:
    """
    [FIX-I30] Confines segmentation prediction classes to the socket-adjacent keep-region ROI.
    Supports quad_ellipse, ellipse, circle, square, and polygon shapes with auto-scaling.
    """
    if center is None:
        return pred_map

    cfg = roi_config or ROIConfig()

    shape        = kwargs.get("shape", cfg.mask_roi_shape)
    classes      = kwargs.get("classes", cfg.mask_roi_classes)
    radius       = kwargs.get("radius", cfg.mask_roi_radius)
    radius_x     = kwargs.get("radius_x", cfg.mask_roi_radius_x)
    radius_y     = kwargs.get("radius_y", cfg.mask_roi_radius_y)
    offset_x     = kwargs.get("offset_x", cfg.mask_roi_offset_x)
    offset_y     = kwargs.get("offset_y", cfg.mask_roi_offset_y)
    polygon      = kwargs.get("polygon", cfg.mask_roi_polygon)
    radius_up    = kwargs.get("radius_up", cfg.mask_roi_radius_up)
    radius_down  = kwargs.get("radius_down", cfg.mask_roi_radius_down)
    radius_left  = kwargs.get("radius_left", cfg.mask_roi_radius_left)
    radius_right = kwargs.get("radius_right", cfg.mask_roi_radius_right)
    auto_scale   = kwargs.get("auto_scale", cfg.mask_roi_auto_scale)

    if shape == "quad_ellipse" and auto_scale and bbox_size is not None:
        bbox_w, bbox_h = bbox_size
        if bbox_w > 0 and bbox_h > 0:
            radius_up    = cfg.mask_roi_mult_up    * bbox_h
            radius_down  = cfg.mask_roi_mult_down  * bbox_h
            radius_left  = cfg.mask_roi_mult_left  * bbox_w
            radius_right = cfg.mask_roi_mult_right * bbox_w

    cx, cy = center
    eff_center = (cx + offset_x, cy + offset_y)
    h, w = pred_map.shape[:2]

    if shape == "quad_ellipse":
        roi_mask = create_quad_ellipse_roi_mask((h, w), eff_center, radius_up, radius_down, radius_left, radius_right)
    elif shape == "polygon":
        roi_mask = create_polygon_roi_mask((h, w), eff_center, polygon)
    elif shape == "ellipse":
        roi_mask = create_ellipse_roi_mask((h, w), eff_center, radius_x, radius_y)
    elif shape == "circle":
        roi_mask = create_circle_roi_mask((h, w), eff_center, radius)
    else:  # "square"
        x1 = max(0, int(eff_center[0] - radius)); x2 = min(w, int(eff_center[0] + radius))
        y1 = max(0, int(eff_center[1] - radius)); y2 = min(h, int(eff_center[1] + radius))
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y1:y2, x1:x2] = True

    for ci in classes:
        outside = (pred_map == ci) & (~roi_mask)
        if outside.any():
            pred_map[outside] = 0

    return pred_map


def apply_exclusion_zone(pred_map: np.ndarray,
                         center: Optional[Tuple[float, float]],
                         bbox_size: Optional[Tuple[int, int]] = None,
                         roi_config: Optional[ROIConfig] = None,
                         **kwargs) -> np.ndarray:
    """
    [FIX-I36] Suppresses tube pixels inside a designated hard exclusion keep-out zone.
    """
    cfg = roi_config or ROIConfig()
    enabled = kwargs.get("enabled", cfg.mask_exclude_enabled)
    if not enabled or center is None:
        return pred_map

    classes    = kwargs.get("classes", cfg.mask_exclude_classes)
    offset_x   = kwargs.get("offset_x", cfg.mask_exclude_offset_x)
    offset_y   = kwargs.get("offset_y", cfg.mask_exclude_offset_y)
    radius_x   = kwargs.get("radius_x", cfg.mask_exclude_radius_x)
    radius_y   = kwargs.get("radius_y", cfg.mask_exclude_radius_y)
    auto_scale = kwargs.get("auto_scale", cfg.mask_exclude_auto_scale)

    if auto_scale and bbox_size is not None:
        bbox_w, bbox_h = bbox_size
        if bbox_w > 0 and bbox_h > 0:
            offset_x = cfg.mask_exclude_offset_mult_x * bbox_w
            offset_y = cfg.mask_exclude_offset_mult_y * bbox_h
            radius_x = cfg.mask_exclude_radius_mult_x * bbox_w
            radius_y = cfg.mask_exclude_radius_mult_y * bbox_h

    h, w = pred_map.shape[:2]
    cx = center[0] + offset_x
    cy = center[1] + offset_y

    x1 = max(0, int(cx - radius_x)); x2 = min(w, int(cx + radius_x) + 1)
    y1 = max(0, int(cy - radius_y)); y2 = min(h, int(cy + radius_y) + 1)
    if x2 <= x1 or y2 <= y1:
        return pred_map

    yy, xx = np.mgrid[y1:y2, x1:x2]
    norm2  = ((xx - cx) / max(radius_x, 1e-6)) ** 2 + ((yy - cy) / max(radius_y, 1e-6)) ** 2
    exclude_mask = np.zeros((h, w), dtype=bool)
    exclude_mask[y1:y2, x1:x2] = norm2 <= 1.0

    for ci in classes:
        inside = (pred_map == ci) & exclude_mask
        if inside.any():
            pred_map[inside] = 0

    return pred_map


def apply_all_roi_gates(pred_map: np.ndarray,
                        center: Optional[Tuple[float, float]],
                        bbox_size: Optional[Tuple[int, int]] = None,
                        roi_config: Optional[ROIConfig] = None) -> np.ndarray:
    """
    Composite ROI helper: Applies socket-adjacent keep-region followed by hard exclusion zone.
    """
    if center is None:
        return pred_map
    pred_map = restrict_mask_to_socket_roi(pred_map, center, bbox_size=bbox_size, roi_config=roi_config)
    pred_map = apply_exclusion_zone(pred_map, center, bbox_size=bbox_size, roi_config=roi_config)
    return pred_map


# ══════════════════════════════════════════════════════════════════════════════
#  3. HARDWARE, MODEL LOADERS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_cyclic_match(detected: List[int], expected: List[int]) -> bool:
    """Checks if detected list is a cyclic permutation of expected list."""
    if len(detected) != len(expected):
        return False
    n = len(expected)
    for i in range(n):
        if detected == expected[i:] + expected[:i]:
            return True
    return False


def make_radial_channel_np(h: int, w: int, cx: Optional[float] = None, cy: Optional[float] = None) -> np.ndarray:
    """Generates normalized radial distance channel (0.0 to 1.0) centered at (cx, cy)."""
    cx = cx if cx is not None else w / 2.0
    cy = cy if cy is not None else h / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
    return np.clip(dist / max(math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2), 1.0), 0.0, 1.0)


def detect_in_channels_from_ckpt(ckpt_path: str) -> int:
    """Detects in_channels (3 or 4) from first conv weights in model checkpoint."""
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


def _farneback_flow_downscaled(img_ref: np.ndarray, img_now: np.ndarray,
                               downscale: Optional[float] = 0.5) -> np.ndarray:
    """
    [FIX-I49] Computes Farneback dense optical flow on downscaled grayscale images.
    """
    downscale = 0.5 if downscale is None else downscale
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


def _gpu_sanity_benchmark(device: str):
    """[FIX-I40] FP16 matmul benchmark to verify GPU compute capability."""
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
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FIX-I40] [WARN] GPU sanity benchmark failed (non-fatal): {e}")


def resolve_device(require_gpu: bool = True) -> str:
    """
    [FIX-I38/I39] Configures CUDA / CPU device and initializes TF32 and normalization tensors.
    """
    global _NORM_MEAN, _NORM_STD
    print("\n" + "=" * 78)
    print("  [FIX-I38] GPU / DEVICE DIAGNOSTICS")
    print("=" * 78)
    print(f"  torch version         : {torch.__version__}")
    print(f"  torch.version.cuda     : {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_ok}")

    if cuda_ok:
        try:
            idx  = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap  = torch.cuda.get_device_capability(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            print(f"  Selected GPU index      : {idx}")
            print(f"  Selected GPU name       : {name} (sm_{cap[0]}{cap[1]})")
            print(f"  GPU memory free/total   : {free_b/1e9:.2f} GB / {total_b/1e9:.2f} GB")
        except Exception as e:
            print(f"  [WARN] Could not read full CUDA device info: {e}")
            cuda_ok = False

    if not cuda_ok:
        print("  [WARN] Running on CPU.")
        print("=" * 78 + "\n")
        if require_gpu:
            raise RuntimeError("CUDA is not available to torch. Pass --no_require_gpu to allow CPU execution.")
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
    print("=" * 78 + "\n")
    return device


def _log_model_device(label: str, device_str: str):
    print(f"[GPU-CHECK] {label:<22} -> device = {device_str}")


def load_yolo(path: str, label: str, device: str = "cpu", use_half: bool = False,
              warmup_hw: Tuple[int, int] = (720, 1280)):
    """Loads and warms up a YOLO model on the target device."""
    if not path:
        return None
    try:
        # pyrefly: ignore [missing-import]
        from ultralytics import YOLO
        m = YOLO(path)
        if device is not None:
            m.to(device)
            try:
                wh, ww = warmup_hw
                dummy = np.zeros((wh, ww, 3), dtype=np.uint8)
                m.predict(dummy, device=device, half=use_half, verbose=False)
            except Exception as warm_e:
                print(f"[WARN] YOLO {label} warmup inference failed: {warm_e}")
        task = getattr(m, "task", "unknown")
        print(f"[OK ] YOLO {label}: {path}  half={use_half}  task={task}")
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
#  4. MODEL RUNNER FUNCTIONS (DISTINCT & CALLABLE WITH CONFIG)
# ══════════════════════════════════════════════════════════════════════════════

def run_model_socket(socket_model: Any,
                     frame: np.ndarray,
                     conf_thr: Optional[float] = None,
                     config: Optional[PipelineConfig] = None) -> Optional[Dict[str, Any]]:
    """
    [FUNCTION 1] Socket Detection Runner.
    Runs YOLO socket model (supporting standard bounding box or OBB).
    Returns dict: {'bbox': (x1,y1,x2,y2), 'class': cls_id, 'conf': float, 'obb_points': list/None}
    """
    if socket_model is None:
        return None

    c_thr = conf_thr if conf_thr is not None else (config.socket.conf_thr if config else 0.35)
    dev   = config.device.device if config else "cpu"
    half  = config.device.use_half if config else False

    res = socket_model(frame, verbose=False, device=dev, half=half)[0]
    best, best_c = None, -1.0

    if getattr(res, "boxes", None) is not None:
        for box in res.boxes:
            c = float(box.conf[0])
            if c >= c_thr and c > best_c:
                best_c = c
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                best = {"bbox": (x1, y1, x2, y2),
                        "class": int(box.cls[0]), "conf": c,
                        "obb_points": None}

    elif getattr(res, "obb", None) is not None and len(res.obb) > 0:
        obb = res.obb
        for i in range(len(obb)):
            c = float(obb.conf[i])
            if c >= c_thr and c > best_c:
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


def run_hand_model(pose_model: Any,
                   frame_bgr: np.ndarray,
                   roi: Tuple[int, int, int, int],
                   prev_gray_ref: Optional[List[Optional[np.ndarray]]] = None,
                   pose_conf_thr: Optional[float] = None,
                   config: Optional[PipelineConfig] = None) -> bool:
    """
    [FUNCTION 2] Hand in ROI Detection Runner.
    Evaluates:
      1. Optical Flow motion in ROI against reference frame.
      2. YOLO Pose Keypoints (wrists/hands) in ROI.
      3. YOLO person bounding box overlap in ROI.
    Returns True if hand is detected in ROI, False otherwise.
    """
    rx1, ry1, rx2, ry2 = roi
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    mot_thr  = config.hand.motion_thr if config else 3.5
    mot_frac = config.hand.motion_frac if config else 0.06
    p_thr    = pose_conf_thr if pose_conf_thr is not None else (config.hand.pose_conf_thr if config else 0.40)
    dev      = config.device.device if config else "cpu"
    half     = config.device.use_half if config else False

    motion = False
    if prev_gray_ref is not None and prev_gray_ref[0] is not None:
        rc = gray[ry1:ry2, rx1:rx2]
        rp = prev_gray_ref[0][ry1:ry2, rx1:rx2]
        if rc.shape == rp.shape and rc.size > 0:
            flow = _farneback_flow_downscaled(rp, rc, downscale=config.device.opt_flow_downscale if config else 0.5)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if float((mag > mot_thr).sum()) / roi_area >= mot_frac:
                motion = True

    if prev_gray_ref is not None:
        prev_gray_ref[0] = gray

    if motion:
        return True

    if pose_model is None:
        return False

    res = pose_model(frame_bgr, verbose=False, device=dev, half=half)[0]
    if hasattr(res, "keypoints") and res.keypoints is not None:
        for kpts in res.keypoints.data:
            if kpts.shape[0] < 11:
                continue
            for idx in (5, 6, 7, 8, 9, 10):  # Shoulders, elbows, wrists
                kx, ky, kc = kpts[idx].tolist()
                if (kc >= p_thr and rx1 <= int(kx) <= rx2 and ry1 <= int(ky) <= ry2):
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
#  5. SEGMENTATION CONFLICT RESOLUTION & ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_tube_conflicts(masks: Dict[int, np.ndarray], dists: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    [FIX-I54] Resolves contested pixels claimed by multiple tube dilations
    to the nearest original pixel class via distance transform.
    """
    classes = list(masks.keys())
    stack = np.stack([masks[ci] for ci in classes], axis=0)
    claim_count = stack.sum(axis=0)
    contested = claim_count > 1
    if not contested.any():
        return {ci: masks[ci] for ci in classes}

    dist_stack = np.stack([dists[ci] for ci in classes], axis=0).astype(np.float32)
    if 3 in classes and 4 in classes:
        idx_4 = classes.index(4)
        dist_stack[idx_4] += 3.0  # Class 3 precedence on contested boundary

    dist_eligible = np.where(stack == 1, dist_stack, np.inf)
    winner_idx = np.argmin(dist_eligible, axis=0)

    out = {}
    for i, ci in enumerate(classes):
        m = masks[ci].copy()
        lose = contested & (winner_idx != i)
        m[lose] = 0
        out[ci] = m
    return out


def _dilate_multiclass_protected(work: np.ndarray, classes: Tuple[int, ...], kernel: np.ndarray) -> Dict[int, np.ndarray]:
    """[FIX-I50 + FIX-I54] Batched dilation with non-tube protection & Voronoi conflict resolution."""
    originals = {ci: (work == ci).astype(np.uint8) for ci in classes}
    if not any(o.any() for o in originals.values()):
        return {ci: originals[ci] for ci in classes}

    stack = np.stack([originals[ci] for ci in classes], axis=-1)
    dilated = cv2.dilate(stack, kernel)
    if dilated.ndim == 2:
        dilated = dilated[..., None]

    masks = {}
    for i, ci in enumerate(classes):
        d = dilated[..., i].copy()
        other = ((work != 0) & (work != ci)).astype(np.uint8)
        d[other == 1] = 0
        masks[ci] = d

    dists = {
        ci: cv2.distanceTransform((originals[ci] == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for ci in classes
    }
    return _resolve_tube_conflicts(masks, dists)


def _close_multiclass_protected(work: np.ndarray, classes: Tuple[int, ...], kernel: np.ndarray) -> Dict[int, np.ndarray]:
    """[FIX-I50 + FIX-I54] Batched morphological close with Voronoi conflict resolution."""
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
        c = np.maximum(eroded[..., i], originals[ci])
        other = ((work != 0) & (work != ci)).astype(np.uint8)
        c = c.copy()
        c[other == 1] = 0
        closed[ci] = c

    dists = {
        ci: cv2.distanceTransform((originals[ci] == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for ci in classes
    }
    return _resolve_tube_conflicts(closed, dists)


# Transform Pipeline
_tf_pipeline = A.Compose([
    A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101),
    ToTensorV2(),
])


@torch.no_grad()
def raw_infer(seg_model: torch.nn.Module,
              frame_bgr: np.ndarray,
              socket_centre: Optional[Tuple[float, float]] = None,
              config: Optional[PipelineConfig] = None) -> np.ndarray:
    """Executes UNet++ neural network forward pass with GPU resizing."""
    cfg_dev = config.device if config else DeviceConfig()
    cfg_seg = config.seg if config else SegmentationConfig()

    oh, ow = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t_u8   = _tf_pipeline(image=rgb)["image"]

    dev           = cfg_dev.device
    compute_dtype = torch.float16 if (cfg_dev.use_half and dev.startswith("cuda")) else torch.float32

    t = t_u8.unsqueeze(0).to(dev, non_blocking=True).to(compute_dtype)
    t = t.div_(255.0).sub_(_NORM_MEAN.to(compute_dtype)).div_(_NORM_STD.to(compute_dtype))

    if cfg_seg.use_radial_channel:
        scale = cfg_seg.img_size[0] / max(oh, ow)
        new_h = int(oh * scale)
        new_w = int(ow * scale)
        pad_y = (cfg_seg.img_size[0] - new_h) // 2
        pad_x = (cfg_seg.img_size[1] - new_w) // 2
        scx   = socket_centre[0] * scale + pad_x if socket_centre else cfg_seg.img_size[1] / 2.0
        scy   = socket_centre[1] * scale + pad_y if socket_centre else cfg_seg.img_size[0] / 2.0
        rad   = make_radial_channel_np(cfg_seg.img_size[0], cfg_seg.img_size[1], cx=scx, cy=scy)
        rad_t = torch.from_numpy(rad).to(dev, non_blocking=True).to(compute_dtype)
        t     = torch.cat([t, rad_t.unsqueeze(0).unsqueeze(0)], dim=1)

    with torch.amp.autocast("cuda", enabled=(dev.startswith("cuda")),
                             dtype=torch.float16 if cfg_dev.use_half else torch.float32):
        logits = seg_model(t)

    probs = F.softmax(logits.float(), dim=1)
    scale = cfg_seg.img_size[0] / max(oh, ow)
    new_h = int(oh * scale)
    new_w = int(ow * scale)
    pad_y = (cfg_seg.img_size[0] - new_h) // 2
    pad_x = (cfg_seg.img_size[1] - new_w) // 2
    crop  = probs[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    resized = F.interpolate(crop, size=(oh, ow), mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()


class SegmentationEngine:
    """
    Stateful Segmentation Engine managing:
    - Temporal EMA probability smoothing
    - Flow-tracked 3-way Tube Identity Hysteresis Lock (FIX-I56)
    - Joint 3-way boundary sharpening
    - Batched conflict-resolved dilation & gap-bridging close
    - Connected component filtering & persistence
    """

    _TUBE_CLASSES = (2, 3, 4)

    def __init__(self, model: torch.nn.Module, config: Optional[PipelineConfig] = None):
        self.model            = model
        self.config           = config or PipelineConfig()
        self.ema_probs        = None
        self._persist         = {c: 0 for c in range(1, self.config.seg.num_classes)}
        self._morph_k         = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.seg.morph_kernel_sz, self.config.seg.morph_kernel_sz))
        self._dilate_k        = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.seg.mask_dilate_sz, self.config.seg.mask_dilate_sz))
        self._close_k         = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.seg.mask_close_sz, self.config.seg.mask_close_sz))
        self.last_raw_pred    = None
        self._dbg_frame_ct    = 0
        self.identity_ema     = None
        self.identity_lock    = None
        self._prev_gray_track = None

    def reset_dilate_kernel(self):
        self._dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.seg.mask_dilate_sz, self.config.seg.mask_dilate_sz))
        self._close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.seg.mask_close_sz, self.config.seg.mask_close_sz))

    def reset(self):
        self.ema_probs         = None
        self.last_raw_pred     = None
        self._persist          = {c: 0 for c in range(1, self.config.seg.num_classes)}
        self.identity_ema      = None
        self.identity_lock     = None
        self._prev_gray_track  = None

    def _track_lock_with_flow(self, frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray_track is None or self.identity_lock is None or self.identity_ema is None:
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
        flow = _farneback_flow_downscaled(prev_crop, gray_crop, downscale=self.config.device.opt_flow_downscale)

        hc, wc = gray_crop.shape
        gx, gy = np.meshgrid(np.arange(wc), np.arange(hc))
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)

        lock_crop = self.identity_lock[y1:y2, x1:x2]
        self.identity_lock[y1:y2, x1:x2] = cv2.remap(
            lock_crop, map_x, map_y,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        for i in range(self.identity_ema.shape[0]):
            ema_crop = self.identity_ema[i, y1:y2, x1:x2]
            self.identity_ema[i, y1:y2, x1:x2] = cv2.remap(
                ema_crop, map_x, map_y,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        self._prev_gray_track = gray

    def _apply_identity_hysteresis(self, probs: np.ndarray, pred: np.ndarray,
                                   enforce: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        h, w = probs.shape[1], probs.shape[2]
        if self.identity_ema is None:
            self.identity_ema  = np.zeros((3, h, w), dtype=np.float32)
            self.identity_lock = np.zeros((h, w), dtype=np.int8)

        class_probs = np.stack([probs[c] for c in self._TUBE_CLASSES], axis=0)
        tube_mass   = class_probs.sum(axis=0)
        present     = tube_mass >= self.config.seg.tube_present_thr

        alpha = self.config.seg.identity_ema_alpha
        self.identity_ema[:, present] = (
            alpha * class_probs[:, present] + (1.0 - alpha) * self.identity_ema[:, present]
        )

        self.identity_lock[~present] = 0

        best_idx = np.argmax(self.identity_ema, axis=0)
        best_val = np.take_along_axis(self.identity_ema, best_idx[None, :, :], axis=0)[0]

        lock_idx = np.clip(self.identity_lock.astype(np.int32) - 2, 0, 2)
        cur_val  = np.take_along_axis(self.identity_ema, lock_idx[None, :, :], axis=0)[0]

        unset = present & (self.identity_lock == 0)
        self.identity_lock[unset] = (best_idx[unset] + 2).astype(np.int8)

        locked = present & (self.identity_lock != 0) & (~unset)
        flip   = locked & (best_idx != lock_idx) & (best_val > cur_val + self.config.seg.tube_identity_margin)
        self.identity_lock[flip] = (best_idx[flip] + 2).astype(np.int8)

        override = present & np.isin(pred, self._TUBE_CLASSES) & (self.identity_lock != 0)
        if enforce:
            pred[override] = self.identity_lock[override]
        else:
            override = np.zeros_like(override)

        return pred, override

    def infer(self, frame_bgr: np.ndarray,
              socket_centre: Optional[Tuple[float, float]] = None,
              apply_identity_lock: bool = True) -> np.ndarray:
        """Runs the full segmentation, smoothing, and morphological pipeline."""
        self._dbg_frame_ct += 1
        cfg_seg = self.config.seg
        cfg_roi = self.config.roi

        perf_roi = build_perf_roi(socket_centre, frame_bgr.shape, pad=cfg_roi.perf_roi_pad,
                                  enabled=cfg_roi.perf_roi_enabled)
        self._track_lock_with_flow(frame_bgr, roi=perf_roi)

        probs_raw = raw_infer(self.model, frame_bgr, socket_centre=socket_centre, config=self.config)

        self.ema_probs = (
            probs_raw.copy() if self.ema_probs is None
            else cfg_seg.ema_alpha * probs_raw + (1 - cfg_seg.ema_alpha) * self.ema_probs
        )
        probs = self.ema_probs.copy()

        if cfg_seg.boundary_sharpening:
            tube_stack  = np.stack([probs[2], probs[3], probs[4]], axis=0)
            tube_stack -= tube_stack.max(axis=0, keepdims=True)
            tube_stack /= cfg_seg.sharpen_temp
            exp_t        = np.exp(tube_stack)
            tube_softmax = exp_t / (exp_t.sum(axis=0, keepdims=True) + 1e-7)
            tube_mass    = probs[2] + probs[3] + probs[4]
            probs[2]     = tube_softmax[0] * tube_mass
            probs[3]     = tube_softmax[1] * tube_mass
            probs[4]     = tube_softmax[2] * tube_mass

        pred = probs.argmax(axis=0).astype(np.uint8)
        self.last_raw_pred = pred.copy().astype(np.int32)

        lock_override_mask = None
        if cfg_seg.identity_hysteresis_enabled:
            pred, lock_override_mask = self._apply_identity_hysteresis(
                probs, pred, enforce=apply_identity_lock)

        active_conf_thr = cfg_seg.class_conf_thr if apply_identity_lock else cfg_seg.warmup_class_conf_thr

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
                low_locked = probs[ci][locked_here] < cfg_seg.locked_pixel_min_conf
                if low_locked.any():
                    yx = np.where(locked_here)
                    pred[yx[0][low_locked], yx[1][low_locked]] = 0

        if perf_roi is not None:
            px1, py1, px2, py2 = perf_roi
            if px1 > 0: pred[:, :px1] = 0
            if px2 < pred.shape[1]: pred[:, px2:] = 0
            if py1 > 0: pred[:py1, :] = 0
            if py2 < pred.shape[0]: pred[py2:, :] = 0
            work = pred[py1:py2, px1:px2]
        else:
            work = pred

        if cfg_seg.mask_dilate_enabled:
            any_tube = any((work == ci).any() for ci in cfg_seg.mask_dilate_classes)
            if any_tube:
                dilated_by_class = _dilate_multiclass_protected(work, cfg_seg.mask_dilate_classes, self._dilate_k)
                for ci in cfg_seg.mask_dilate_classes: work[work == ci] = 0
                for ci in cfg_seg.mask_dilate_classes: work[dilated_by_class[ci] == 1] = ci

            if cfg_seg.mask_close_enabled and cfg_seg.mask_close_sz > 0:
                any_tube2 = any((work == ci).any() for ci in cfg_seg.mask_dilate_classes)
                if any_tube2:
                    closed_by_class = _close_multiclass_protected(work, cfg_seg.mask_dilate_classes, self._close_k)
                    for ci in cfg_seg.mask_dilate_classes: work[work == ci] = 0
                    for ci in cfg_seg.mask_dilate_classes: work[closed_by_class[ci] == 1] = ci

        # Final morphological clean pass with conflict resolution
        originals_morph = {ci: (work == ci).astype(np.uint8) for ci in (2, 3, 4)}
        candidate_morph = {}
        for ci in range(1, cfg_seg.num_classes):
            bm = (work == ci).astype(np.uint8)
            bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, self._morph_k)
            if ci not in (3, 4):
                bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, self._morph_k)
            candidate_morph[ci] = bm

        dists_morph = {
            ci: cv2.distanceTransform((originals_morph[ci] == 0).astype(np.uint8), cv2.DIST_L2, 3)
            for ci in (2, 3, 4)
        }
        resolved_morph = _resolve_tube_conflicts({ci: candidate_morph[ci] for ci in (2, 3, 4)}, dists_morph)
        for ci in (2, 3, 4):
            work[work == ci] = 0
            work[resolved_morph[ci] == 1] = ci

        # Connected component filtering
        for ci in range(1, cfg_seg.num_classes):
            bm = (work == ci).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bm)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < cfg_seg.min_area_px:
                    work[labels == i] = 0

        # Temporal persistence
        for ci in range(1, cfg_seg.num_classes):
            self._persist[ci] = self._persist[ci] + 1 if (pred == ci).any() else 0
            if self._persist[ci] < cfg_seg.persist_frames:
                pred[pred == ci] = 0

        return pred.astype(np.int32)


def run_model_tube_segmentation(seg_engine: SegmentationEngine,
                                frame: np.ndarray,
                                socket_centre: Optional[Tuple[float, float]] = None,
                                socket_bbox_size: Optional[Tuple[int, int]] = None,
                                apply_identity_lock: bool = True,
                                apply_roi_gates: bool = True,
                                config: Optional[PipelineConfig] = None) -> np.ndarray:
    """
    [FUNCTION 3] Tube Segmentation Model Runner.
    Runs segmentation inference via SegmentationEngine and optionally applies
    all socket ROI restriction gates and exclusion zones.
    """
    cfg_roi = config.roi if config else (seg_engine.config.roi if seg_engine else ROIConfig())

    pred_map = seg_engine.infer(frame, socket_centre=socket_centre,
                                apply_identity_lock=apply_identity_lock)
    if apply_roi_gates and socket_centre is not None:
        pred_map = apply_all_roi_gates(pred_map, socket_centre, bbox_size=socket_bbox_size, roi_config=cfg_roi)

    return pred_map


# ══════════════════════════════════════════════════════════════════════════════
#  6. TUBE ORDER & GATING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_tube_order(pred_map: np.ndarray,
                        socket_bbox: Optional[Tuple[int, int, int, int]] = None,
                        search_radius: int = 350,
                        min_tube_px: int = 40,
                        expected_seq: Optional[List[int]] = None,
                        debug: bool = False) -> Tuple[Dict[int, str], str, List[int], Dict[str, Any]]:
    """
    [FIX-I18] Nearest-Pixel Angular Gate for Tube Order Evaluation.
    Calculates tube angles relative to socket center and validates cyclic sequence order.
    """
    expected = expected_seq or [2, 3, 4]
    if socket_bbox is not None:
        x1, y1, x2, y2 = socket_bbox
        scx, scy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    else:
        ys, xs = np.where(pred_map == 1)
        scx    = float(xs.mean()) if len(xs) else pred_map.shape[1] / 2.0
        scy    = float(ys.mean()) if len(ys) else pred_map.shape[0] / 2.0

    H, W = pred_map.shape[:2]
    R    = search_radius

    rx1 = max(0, int(scx - R)); ry1 = max(0, int(scy - R))
    rx2 = min(W - 1, int(scx + R)); ry2 = min(H - 1, int(scy + R))

    rect_mask = np.zeros((H, W), dtype=np.uint8)
    rect_mask[ry1:ry2 + 1, rx1:rx2 + 1] = 1
    pred_roi = pred_map * rect_mask

    status    = {2: "Absent", 3: "Absent", 4: "Absent"}
    angles    = {}
    anchors   = {}
    nearest_d = {}

    for ci in (2, 3, 4):
        ty, tx = np.where(pred_roi == ci)
        if len(tx) < min_tube_px:
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

    if len(visible) < len(expected):
        if debug: dbg["result"] = "PARTIAL"
        return status, "PARTIAL", visible, dbg

    sorted_by_angle = sorted(visible, key=lambda c: angles[c])
    TWO_PI = 2.0 * math.pi
    n      = len(sorted_by_angle)
    raw_a  = [angles[c] for c in sorted_by_angle]
    gaps   = [(raw_a[(i + 1) % n] - raw_a[i]) % TWO_PI for i in range(n)]
    start  = (int(np.argmax(gaps)) + 1) % n
    seq    = sorted_by_angle[start:] + sorted_by_angle[:start]
    order_result = "OK" if is_cyclic_match(seq, expected) else "ANOMALY"

    if debug:
        dbg["gaps_deg"]    = [(sorted_by_angle[i], sorted_by_angle[(i + 1) % n],
                                 math.degrees(gaps[i])) for i in range(n)]
        dbg["max_gap_idx"] = int(np.argmax(gaps))
        dbg["start_ci"]    = sorted_by_angle[start]
        dbg["seq"]         = seq
        dbg["result"]      = order_result

    return status, order_result, seq, dbg


class VoteCounter:
    """Maintains voting ratio between NORMAL and ANOMALY frames in a cycle."""
    def __init__(self, threshold: float = 0.50):
        self.threshold     = threshold
        self.anomaly_votes = 0
        self.normal_votes  = 0

    def reset(self):
        self.anomaly_votes = 0
        self.normal_votes  = 0

    def record(self, gate_result: str):
        if gate_result == "ANOMALY":
            self.anomaly_votes += 1
        elif gate_result == "OK":
            self.normal_votes  += 1

    @property
    def total(self) -> int:
        return self.anomaly_votes + self.normal_votes

    def final_verdict(self) -> str:
        if self.total == 0:
            return "UNKNOWN"
        return "ANOMALY" if (self.anomaly_votes / self.total) > self.threshold else "NORMAL"


class SequenceStabilityGate:
    """[FIX-I6] Requires detected tube sequence to be stable for N frames."""
    def __init__(self, min_stable: int = 3):
        self.min_stable = min_stable
        self._prev_seq  = None
        self._stable_ct = 0

    def reset(self):
        self._prev_seq  = None
        self._stable_ct = 0

    def update(self, raw_order: str, detected_seq: List[int]) -> str:
        seq_key = tuple(detected_seq)
        if seq_key == self._prev_seq:
            self._stable_ct += 1
        else:
            self._stable_ct = 1
            self._prev_seq  = seq_key
        if self._stable_ct >= self.min_stable:
            return raw_order
        return "PARTIAL"


class AnomalyConfirmGate:
    """Latches ANOMALY verdict once consecutive anomaly count reaches threshold."""
    def __init__(self, n: int = 4):
        self.n        = n
        self._count   = 0
        self._latched = False

    def reset(self):
        self._count   = 0
        self._latched = False

    def update(self, order_result: str) -> str:
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
    """Locks final verdict once result is held for N consecutive frames."""
    def __init__(self, n: int = 10):
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

    def update(self, gate_result: str) -> bool:
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


def mask_has_moved(gray_now: np.ndarray, gray_ref: np.ndarray,
                   thr: float = 2.5, frac: float = 0.04,
                   downscale: float = 0.5) -> bool:
    """[FIX-I37] Optical flow motion test to trigger mask refresh in mask-freeze mode."""
    if gray_now.shape != gray_ref.shape or gray_now.size == 0:
        return True
    flow = _farneback_flow_downscaled(gray_ref, gray_now, downscale=downscale)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moved_frac = float((mag > thr).sum()) / mag.size
    return moved_frac >= frac


# ══════════════════════════════════════════════════════════════════════════════
#  7. CYCLE MANAGER & EXCEL LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def get_verdict_dir(out_dir: str, verdict: str) -> Tuple[str, str]:
    folder = verdict if verdict in ("NORMAL", "ANOMALY", "UNKNOWN") else "UNKNOWN"
    d      = os.path.join(out_dir, folder)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d, folder


class CycleManager:
    """Manages video recording and cycle lifecycle per socket attach-detach span."""
    def __init__(self, resolved_output_dir: str, video_stem: str, fps_src: float,
                 frame_size: Tuple[int, int], hold_sec: float = 2.0):
        self.resolved_output_dir = resolved_output_dir
        self.video_stem          = video_stem
        self.fps_src             = fps_src
        self.frame_size          = frame_size
        self.hold_sec            = hold_sec

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
    def total_cycles(self) -> int:
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

    def write(self, frame: np.ndarray):
        if self.active and self.writer is not None:
            self.writer.write(frame)

    def hold_final_frame(self, frame: np.ndarray, seconds: Optional[float] = None):
        if not self.active or self.writer is None:
            return
        sec = self.hold_sec if seconds is None else seconds
        hold = max(1, int(self.fps_src * sec))
        for _ in range(hold):
            self.writer.write(frame)

    def end_cycle(self, verdict: str, extra_metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.active:
            return None

        if self.writer is not None:
            self.writer.release()
            self.writer = None
            time.sleep(0.15)
        self.active = False

        dest_dir, folder_name = get_verdict_dir(self.resolved_output_dir, verdict)
        final_name = f"{self.video_stem}_{folder_name}_cycle{self.cycle_no:03d}.mp4"
        final_path = os.path.join(dest_dir, final_name)

        for _attempt in range(5):
            try:
                if Path(final_path).exists():
                    try:
                        Path(final_path).unlink()
                    except Exception:
                        pass
                shutil.move(self.temp_path, final_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.3)
                if _attempt == 4 and Path(self.temp_path).exists():
                    try:
                        shutil.copy2(self.temp_path, final_path)
                        os.remove(self.temp_path)
                    except Exception:
                        pass

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
        print("\n" + "=" * W)
        print("  FINAL CYCLE REPORT")
        print("=" * W)
        print(f"  TOTAL CYCLES      : {self.total_cycles}")
        print(f"  PASSED (NORMAL)   : {self.passed}")
        print(f"  FAILED (ANOMALY)  : {self.failed}")
        print(f"  UNKNOWN           : {self.unknown}")
        print("=" * W + "\n")


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


def append_to_excel(run_metrics: Dict[str, Any], excel_dir: str, max_retries: int = 5):
    """[FIX-I55] Appends cycle metrics to Excel log with persistent Sr No. and Timestamp."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[WARN] openpyxl not installed — skipping Excel."); return

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
            cell.fill      = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border    = bdr
        ws.row_dimensions[1].height = 32
        ws.freeze_panes = "A2"
        return wb, ws

    if os.path.exists(excel_path):
        try:
            wb = openpyxl.load_workbook(excel_path)
            ws = wb.active
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
            broken_path = os.path.join(
                excel_dir, f"inspection_log.broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            try: shutil.move(excel_path, broken_path)
            except OSError: pass
            wb, ws = _create_new_workbook()
    else:
        wb, ws = _create_new_workbook()

    existing_data_rows = max(0, ws.max_row - 1)
    sr_no = existing_data_rows + 1

    row_vals = []
    for key, _ in _EXCEL_COLUMNS:
        if key == "sr_no": row_vals.append(sr_no)
        elif key == "cycle_no": row_vals.append(int(run_metrics.get(key, 0)))
        elif key == "timestamp": row_vals.append(run_metrics.get(key, "N/A"))
        else: row_vals.append(run_metrics.get(key, "N/A"))
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
        cell.alignment = Alignment(horizontal="center" if key in center_keys else "left", vertical="center")
        val = str(cell.value or "")
        if key in verdict_keys:
            hx        = _VERDICT_FILLS.get(val, "F2F2F2")
            cell.fill = PatternFill(start_color=hx, end_color=hx, fill_type="solid")
            cell.font = Font(name="Arial", size=10, bold=True)

    for ci in range(1, len(_EXCEL_COLUMNS) + 1):
        cl = get_column_letter(ci)
        mx = max(len(str(ws.cell(row=r, column=ci).value or "")) for r in range(1, ws.max_row + 1))
        ws.column_dimensions[cl].width = max(mx + 4, 14)

    for attempt in range(1, max_retries + 1):
        try:
            wb.save(excel_path)
            print(f"[EXCEL] Sr#{sr_no}  Cycle #{run_metrics.get('cycle_no','-')} → {excel_path}")
            return
        except PermissionError:
            if attempt == max_retries: raise
            print(f"[WARN] Excel locked, retry {attempt}/{max_retries} in 5s…")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  8. RENDERING & HUD OVERLAYS
# ══════════════════════════════════════════════════════════════════════════════

def draw_seg_overlay(frame: np.ndarray, pred_map: np.ndarray, alpha: float = 0.90) -> np.ndarray:
    """Renders semi-transparent tube segmentation mask overlays on frame."""
    out = frame.copy()
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show: continue
        mask = pred_map == ci
        if not mask.any(): continue
        layer       = np.zeros_like(frame)
        layer[mask] = bgr
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0)
    return out


def draw_raw_argmax_fallback(frame: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    """Fallback renderer when filtered mask is blank."""
    out = frame.copy()
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show: continue
        mask = raw_pred == ci
        if not mask.any(): continue
        layer       = np.zeros_like(frame)
        layer[mask] = tuple(int(v * 0.40) for v in bgr)
        out = cv2.addWeighted(out, 1.0, layer, 0.50, 0)
    H, _ = out.shape[:2]
    cv2.putText(out, "RAW ARGMAX (pre-filter)", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 200), 1, cv2.LINE_AA)
    return out


def draw_socket_box(frame: np.ndarray, hit: Optional[Dict[str, Any]], fill_alpha: float = 0.28) -> np.ndarray:
    """[FIX-I47/I53] Draws socket / no-socket filled bounding box (supports standard box & OBB)."""
    if hit is None:
        return frame

    x1, y1, x2, y2 = hit["bbox"]
    is_p  = hit["class"] == 1

    col      = (0, 220, 100) if is_p else (50, 50, 230)
    fill_col = (150, 255, 190) if is_p else (140, 140, 255)
    label    = f"{'Socket' if is_p else 'No Socket'}  {hit['conf']*100:.0f}%"
    obb_pts  = hit.get("obb_points")

    overlay = frame.copy()
    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], fill_col)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_col, -1)
    frame = cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0)

    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
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


def draw_final_verdict_overlay(frame: np.ndarray, verdict: str,
                               cycle_no: Optional[int] = None,
                               stats: Optional[Dict[str, Any]] = None,
                               config: Optional[PipelineConfig] = None) -> np.ndarray:
    """Renders high-visibility end-of-cycle summary card overlay."""
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
    cv2.putText(out, label, (tx + 3, ty + 3), cv2.FONT_HERSHEY_DUPLEX, fs_big, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, fs_big, tc, 3, cv2.LINE_AA)
    fs_sub = max(0.8, fs_big * 0.42)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 2)
    sy = ty + th + max(20, int(H * 0.04))
    cv2.putText(out, sub, ((W - sw) // 2, sy), cv2.FONT_HERSHEY_SIMPLEX, fs_sub, tc, 2, cv2.LINE_AA)
    if stats:
        in_ch  = config.seg.in_channels if config else 3
        use_rc = config.seg.use_radial_channel if config else False
        lines = [
            f"Cycle frames  : {stats.get('total_frames', '-')}",
            f"Warmup frames : {stats.get('warmup_frames', '-')}",
            f"Infer frames  : {stats.get('infer_frames', '-')}",
            f"Normal votes  : {stats.get('normal_votes', '-')}",
            f"Anomaly votes : {stats.get('anomaly_votes', '-')}",
            f"Anomaly ratio : {stats.get('anomaly_ratio', 0):.1%}",
            f"Channels      : {in_ch}  radial={use_rc}",
        ]
        fs_s = max(0.55, fs_big * 0.32)
        rh   = max(24, int(H * 0.038))
        bw   = max(300, int(W * 0.30))
        bh   = rh * len(lines) + 24
        bx   = (W - bw) // 2
        by   = sy + max(30, int(H * 0.05))
        cv2.rectangle(out, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), (30, 30, 30), -1)
        cv2.rectangle(out, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), tc, 1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (bx, by + (i + 1) * rh),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_s, (210, 210, 210), 1, cv2.LINE_AA)
    return out


_STATE_STYLE = {
    STATE_IDLE:    {"chip_bg": (45, 45, 45),   "chip_fg": (150, 150, 150), "border": (80, 80, 80)},
    STATE_WARMUP:  {"chip_bg": (100, 80, 10),  "chip_fg": (255, 220, 50),  "border": (180, 140, 20)},
    STATE_HAND:    {"chip_bg": (0, 120, 210),  "chip_fg": (255, 255, 255), "border": (0, 165, 255)},
    STATE_INSPECT: {"chip_bg": (20, 90, 170),  "chip_fg": (255, 255, 255), "border": (40, 130, 210)},
    STATE_NORMAL:  {"chip_bg": (10, 140, 40),  "chip_fg": (255, 255, 255), "border": (30, 200, 60)},
    STATE_ANOMALY: {"chip_bg": (15, 15, 210),  "chip_fg": (255, 255, 255), "border": (30, 30, 240)},
    STATE_PARTIAL: {"chip_bg": (20, 130, 160), "chip_fg": (255, 255, 255), "border": (40, 175, 195)},
}


def _put(img: np.ndarray, text: str, x: int, y: int, fs: float, col: Tuple[int, int, int], thick: int = 1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick, cv2.LINE_AA)


def draw_production_status_bar(frame: np.ndarray, state: str, cycle_no: int,
                               passed: int, failed: int, unknown: int) -> np.ndarray:
    """Renders the top production summary header bar."""
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
    _put(out, seg1, cx, ty, fs_main, (255, 255, 255), 2); cx += w1 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg2, cx, ty, fs_main, sty["chip_fg"] if state not in (STATE_NORMAL, STATE_ANOMALY)
         else (60, 230, 60) if state == STATE_NORMAL else (60, 60, 255), 2)
    cx += w2 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg3, cx, ty, fs_main, (60, 230, 60), 2); cx += w3 + gap
    cv2.line(out, (cx - gap // 2, by1 + 6), (cx - gap // 2, by2 - 6), (90, 90, 90), 1)
    _put(out, seg4, cx, ty, fs_main, (60, 60, 255), 2)

    if unknown:
        sub = f"unknown: {unknown}"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 1)
        _put(out, sub, (W - sw) // 2, by2 + int(18 * S), fs_sub, (150, 150, 150), 1)

    return out


def draw_hud(frame: np.ndarray, fps: float, frame_idx: int, state: str,
             socket_hit: Optional[Dict[str, Any]], status_dict: Dict[int, str],
             order_status: str, detected_seq: List[int],
             anomaly_counter: int = 0, hand_in_roi: bool = False,
             warmup_frame: int = 0, warmup_retry: int = 0,
             vote_counter: Optional[VoteCounter] = None, infer_frames: int = 0,
             seq_stable_ctr: int = 0, cycle_no: int = 0, frame_ms: float = 0.0,
             frame_ms_avg: float = 0.0, mask_is_locked: bool = False,
             config: Optional[PipelineConfig] = None,
             total_frames: int = 0, **kwargs) -> np.ndarray:
    """Renders comprehensive side HUD panels for diagnostic metrics."""
    out  = frame.copy()
    H, W = out.shape[:2]
    S    = W / 1280.0
    PAD  = max(10, int(12 * S))
    FS_XS = max(0.40, 0.42 * S); FS_SM = max(0.48, 0.52 * S)
    FS_MD = max(0.58, 0.62 * S); FS_LG = max(0.70, 0.76 * S)
    TK1   = max(1, int(S));       TK2   = max(1, int(2 * S))
    ROW   = max(26, int(28 * S)); DOT   = max(5, int(6 * S))

    TOP_OFFSET = max(70, int(78 * S))

    LP_W  = max(280, int(300 * S)); LP_H = PAD * 2 + ROW * 10 + 8
    LP_X, LP_Y = 10, 10 + TOP_OFFSET
    RP_W  = max(300, int(325 * S)); RP_H = PAD * 2 + ROW * 9 + 20
    RP_X  = W - RP_W - 10; RP_Y = 10 + TOP_OFFSET

    ovl = out.copy()
    for (px, py, pw, ph) in [(LP_X, LP_Y, LP_W, LP_H), (RP_X, RP_Y, RP_W, RP_H)]:
        cv2.rectangle(ovl, (px, py), (px + pw, py + ph), (14, 14, 14), -1)
    cv2.addWeighted(ovl, 0.72, out, 0.28, 0, out)

    sty = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
    for (px, py, pw, ph), bc in [
        ((LP_X, LP_Y, LP_W, LP_H), sty["border"]),
        ((RP_X, RP_Y, RP_W, RP_H), (70, 70, 70))
    ]:
        cv2.rectangle(out, (px, py), (px + pw, py + ph), bc, 1)

    lx = LP_X + PAD; ly = LP_Y + PAD + ROW - 4; vx = lx + max(60, int(64 * S))
    _put(out, "FPS",   lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{fps:5.1f}", vx, ly, FS_LG, (220, 220, 220), TK2); ly += ROW + 4
    _put(out, "FRAME", lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_idx:06d}", vx, ly, FS_MD, (200, 200, 200), TK1); ly += ROW + 4

    warn_thr = config.ui.frame_ms_warn_threshold if config else 150.0
    ms_col = (60, 60, 255) if frame_ms_avg >= warn_thr else (200, 200, 200)
    _put(out, "MS",    lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_ms:5.1f} (avg {frame_ms_avg:5.1f})", vx, ly, FS_SM, ms_col, TK1); ly += ROW + 6

    chip = _CHIP_LABEL.get(state, state)
    (cw, ch), bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, FS_SM, TK1)
    cpx, cpy = max(10, int(11 * S)), max(6, int(7 * S))
    cx1, cy1 = lx, ly; cx2, cy2 = cx1 + cw + cpx * 2, cy1 + ch + bl + cpy * 2
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["chip_bg"], -1)
    cv2.rectangle(out, (cx1, cy1), (cx2, cy2), sty["border"], 1)
    _put(out, chip, cx1 + cpx, cy1 + cpy + ch, FS_SM, sty["chip_fg"], TK1)
    n_anom_thr = config.inspection.n_anomaly_confirm if config else 4
    if 0 < anomaly_counter < n_anom_thr:
        _put(out, f"({anomaly_counter}/{n_anom_thr})", cx2 + 6, cy1 + cpy + ch, FS_XS, (160, 80, 80), TK1)
    ly = cy2 + 6

    warmup_total = config.inspection.warmup_frames if config else 20
    max_w_retries = config.inspection.max_warmup_retries if config else 3
    if state == STATE_WARMUP and warmup_total > 0:
        bw   = cx2 - cx1; bh = max(6, int(7 * S))
        prog = min(warmup_frame / (warmup_total * (warmup_retry + 1)), 1.0)
        cv2.rectangle(out, (cx1, ly), (cx1 + bw, ly + bh), (60, 60, 60), -1)
        cv2.rectangle(out, (cx1, ly), (cx1 + int(bw * prog), ly + bh), (180, 140, 20), -1); ly += bh + 4
        if warmup_retry > 0:
            _put(out, f"retry {warmup_retry}/{max_w_retries}", cx1, ly + ROW - 6, FS_XS, (160, 120, 40), TK1); ly += ROW

    if state == STATE_HAND:
        _put(out, "DETECTIONS PAUSED (hand in ROI)", lx, ly + ROW - 6, FS_XS, _ROI_COL_AMBER, TK1); ly += ROW
    else:
        _put(out, f"LIVE INFER   [{infer_frames}f]", lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        if mask_is_locked:
            _put(out, "MASK: LOCKED", lx + 190, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        else:
            _put(out, "MASK: REFRESHED", lx + 190, ly + ROW - 6, FS_XS, (0, 165, 255), TK1)
        ly += ROW

        min_seq_req = config.inspection.min_seq_stable if config else 3
        if seq_stable_ctr > 0:
            sc_col = (60, 220, 60) if seq_stable_ctr >= min_seq_req else (160, 160, 40)
            _put(out, f"SEQ STABLE  {seq_stable_ctr}/{min_seq_req}", lx, ly + ROW - 6, FS_XS, sc_col, TK1); ly += ROW

    if vote_counter is not None and vote_counter.total > 0:
        ly += 2
        _put(out, f"OK : {vote_counter.normal_votes}", lx, ly + ROW - 6, FS_XS, (60, 220, 60), TK1); ly += ROW
        _put(out, f"AN : {vote_counter.anomaly_votes}", lx, ly + ROW - 6, FS_XS, (80, 80, 230), TK1); ly += ROW

    rx = RP_X + PAD; ry = RP_Y + PAD
    _put(out, "INSPECTION STATUS", rx, ry + ROW - 6, FS_XS, (100, 100, 100), TK1)
    ry += ROW + 4
    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1); ry += 8

    if hand_in_roi:
        hc = _ROI_COL_AMBER; ht = "HAND     IN ROI"
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, hc, -1)
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT + 2, (255, 255, 255), 1)
    else:
        hc = (70, 70, 70); ht = "HAND     CLEAR"
        cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, hc, -1)
    _put(out, ht, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, hc, TK1); ry += ROW + 4

    s_col, s_txt = (
        ((0, 210, 100), "SOCKET   PRESENT")
        if socket_hit and socket_hit["class"] == 1
        else ((60, 60, 220), "SOCKET   ABSENT")
    )
    cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, s_col, -1)
    _put(out, s_txt, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, s_col, TK1); ry += ROW + 4
    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1); ry += 8

    if hand_in_roi:
        for ci in (2, 3, 4):
            dc = (50, 50, 50); tc3 = (140, 140, 140)
            cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   PAUSED", rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1); ry += ROW + 4
    else:
        for ci in (2, 3, 4):
            pres = status_dict.get(ci, "Absent") == "Present"
            dc   = CLASS_INFO[ci][1] if pres else (50, 50, 50)
            tc3  = (180, 255, 180) if pres else (90, 90, 90)
            cv2.circle(out, (rx + DOT, ry + ROW // 2 - 3), DOT, dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   {'PRESENT' if pres else 'ABSENT'}",
                 rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1); ry += ROW + 4

    cv2.line(out, (rx, ry), (RP_X + RP_W - PAD, ry), (45, 45, 45), 1); ry += 8

    if hand_in_roi:
        seq_str = "-"
    else:
        seq_str = " > ".join(TUBE_SHORT[c] for c in detected_seq) if detected_seq else "-"
    _put(out, f"SEQ  {seq_str}", rx, ry + ROW - 6, FS_SM, (160, 160, 160), TK1); ry += ROW + 6

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


def draw_debug_overlay(frame: np.ndarray, dbg: Dict[str, Any]) -> np.ndarray:
    """Renders detailed nearest-pixel angular geometry debug lines on frame."""
    if not dbg:
        return frame
    out = frame.copy()
    scx = int(dbg.get("scx", 0));  scy = int(dbg.get("scy", 0))
    rx1 = int(dbg.get("rx1", scx)); ry1 = int(dbg.get("ry1", scy))
    rx2 = int(dbg.get("rx2", scx)); ry2 = int(dbg.get("ry2", scy))
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (scx, scy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    anchors    = dbg.get("anchors", {})
    angles_deg = dbg.get("angles_deg", {})
    nearest_d  = dbg.get("nearest_dist", {})
    gaps       = dbg.get("gaps_deg", [])
    seq        = dbg.get("seq", [])
    result     = dbg.get("result", "?")
    max_gap    = max((g for _, _, g in gaps), default=0)

    for fc, tc2, gd in gaps:
        col = (0, 0, 220) if abs(gd - max_gap) < 0.01 else (100, 100, 100)
        af  = anchors.get(fc); at = anchors.get(tc2)
        if af and at:
            mx, my = int((af[0] + at[0]) / 2), int((af[1] + at[1]) / 2)
            cv2.putText(out, f"{gd:.0f}°", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    for ci, (ax, ay) in anchors.items():
        iax, iay = int(ax), int(ay)
        col = CLASS_INFO[ci][1]
        cv2.line(out, (scx, scy), (iax, iay), col, 2, cv2.LINE_AA)
        cv2.circle(out, (iax, iay), 6, col, -1)
        cv2.circle(out, (iax, iay), 6, (255, 255, 255), 1)
        dtxt = f" d={nearest_d.get(ci, 0):.0f}px" if ci in nearest_d else ""
        cv2.putText(out, f"{TUBE_SHORT[ci]} {angles_deg.get(ci, 0):.1f}°{dtxt}",
                    (iax + 8, iay + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

    H, _    = out.shape[:2]
    seq_str = " > ".join(TUBE_SHORT[c] for c in seq) if seq else "?"
    col_b   = ((0, 200, 0) if result == "OK" else (0, 0, 220) if result == "ANOMALY" else (50, 170, 200))
    banner  = f"[DBG] NEAREST-PX SEQ: {seq_str}  |  RESULT: {result}"
    (bw, bh), bl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    by = H - 14
    cv2.rectangle(out, (6, by - bh - 6), (6 + bw + 12, by + bl + 2), (20, 20, 20), -1)
    cv2.putText(out, banner, (12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_b, 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  9. CORE PROCESSOR & SINGLE VIDEO MULTI-CYCLE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_video_cycles(video_path: str,
                         resolved_output_dir: str,
                         seg_net: torch.nn.Module,
                         yolo_socket: Any,
                         yolo_pose: Any,
                         config: PipelineConfig,
                         print_summary: bool = False,
                         enable_debug: bool = False) -> Optional[CycleManager]:
    """
    Executes the multi-cycle video inference pipeline using modular model runners
    (run_model_socket, run_hand_model, run_model_tube_segmentation) and config settings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}"); return None

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] {Path(video_path).name} — {src_w}x{src_h} @ {fps_src:.1f} fps ({total_frames} frames)")

    is_short_clip = (0 < total_frames < 250)
    effective_warmup_frames = 2 if is_short_clip else config.inspection.warmup_frames
    effective_anomaly_confirm = 2 if is_short_clip else config.inspection.n_anomaly_confirm
    effective_max_retries = 0 if is_short_clip else config.inspection.max_warmup_retries

    if is_short_clip:
        print(f"[AUTO-DETECT] Short single-cycle clip detected ({total_frames} frames). "
              f"Configured warmup={effective_warmup_frames} frames, "
              f"anomaly_confirm={effective_anomaly_confirm} frames.")

    video_stem = Path(video_path).stem
    cycles     = CycleManager(resolved_output_dir, video_stem, fps_src, (src_w, src_h),
                              hold_sec=config.inspection.verdict_hold_sec)

    seg_engine   = SegmentationEngine(seg_net, config=config)
    anomaly_gate = AnomalyConfirmGate(effective_anomaly_confirm)
    latch_gate   = ResultLatchGate(config.inspection.latch_frames)
    vote_counter = VoteCounter(config.inspection.verdict_thr)
    seq_gate     = SequenceStabilityGate(config.inspection.min_seq_stable)

    show_preview = config.ui.show_preview
    WIN = f"v50_Merged | {os.path.basename(video_path)} | Q=quit D=debug F=fs M=maskdbg"
    if show_preview:
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, min(src_w, 1280), min(src_h, 720))
        except (cv2.error, Exception) as gui_err:
            show_preview = False
            print(f"[WARN] OpenCV GUI preview unavailable ({gui_err}). Running in headless mode.")

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

    last_socket_centre    = None
    last_socket_bbox_size = None

    mask_frozen_pred     = None
    mask_frozen_gray_roi = None
    mask_is_locked       = False

    frame_ms_ema = 0.0
    status_dict  = EMPTY_STAT.copy()
    detected_seq = []
    order_status = "N/A"
    last_vis     = None

    peak_status = EMPTY_STAT.copy()
    peak_seq    = []
    peak_order  = "N/A"

    prev_gray_ref: List[Optional[np.ndarray]] = [None]

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
                last_vis, final_verdict, cycle_no=cycles.cycle_no, stats=stats_card, config=config)
            cycles.hold_final_frame(card)
            if show_preview:
                try:
                    cv2.imshow(WIN, card)
                    cv2.waitKey(int(config.inspection.verdict_hold_sec * 1000))
                except Exception:
                    pass

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
            "ema_alpha":         config.seg.ema_alpha,
            "sharpening":        "ON" if config.seg.boundary_sharpening else "OFF",
            "seq_stable_min":    config.inspection.min_seq_stable,
            "channels":          config.seg.in_channels,
            "radial_channel":    "ON" if config.seg.use_radial_channel else "OFF",
            "debug_mode":        "ON" if enable_debug else "OFF",
        }
        summary = cycles.end_cycle(final_verdict, extra_metrics=extra)
        if summary and print_summary:
            print("\n" + "=" * 62 + f"\n CYCLE #{summary['cycle_no']:03d} SUMMARY\n" + "=" * 62)
            for k, v in summary.items():
                print(f"  {k:<26}: {v}")
            print("=" * 62 + "\n")
        if summary:
            append_to_excel(summary, resolved_output_dir, max_retries=config.ui.max_excel_retries)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0  = time.perf_counter()
        vis = frame.copy()

        # ── Step 1: Concurrent Model Execution for Socket & Hand Detection ──
        sock_future = _YOLO_EXECUTOR.submit(
            run_model_socket, yolo_socket, frame, config.socket.conf_thr, config)
        hand_future = None
        if invisible_roi is not None:
            hand_future = _YOLO_EXECUTOR.submit(
                run_hand_model, yolo_pose, frame, invisible_roi,
                prev_gray_ref, config.hand.pose_conf_thr, config)

        sock_hit      = sock_future.result()
        socket_now    = sock_hit is not None and sock_hit["class"] == config.socket.cls_socket
        no_socket_now = sock_hit is not None and sock_hit["class"] == config.socket.cls_no_socket

        if socket_now:
            last_socket_centre, last_socket_bbox_size = compute_socket_roi_geometry(
                sock_hit, last_socket_centre, last_socket_bbox_size)

        was_latched = socket_latched

        if socket_now:
            if socket_drop_frames > 0:
                print(f"[INFO] flicker absorbed: socket reappeared after "
                      f"{socket_drop_frames} 'no socket' frame(s) — "
                      f"still cycle #{cycles.cycle_no:03d}")
            invisible_roi      = build_socket_roi(frame.shape, sock_hit["bbox"],
                                                  pad_x=config.roi.pad_x, pad_y=config.roi.pad_y)
            socket_latched     = True
            socket_drop_frames = 0
        elif no_socket_now:
            socket_drop_frames += 1
            if socket_drop_frames >= config.socket.reset_grace_frames:
                print(f"[INFO] removal confirmed after {socket_drop_frames} consecutive "
                      f"'no socket' frames — closing cycle #{cycles.cycle_no:03d}")
                invisible_roi         = None
                socket_latched        = False
                warmup_frame_count    = 0
                warmup_retry          = 0
                warmup_done           = False
                infer_frames          = 0
                cycle_total_frames    = 0
                last_socket_centre    = None
                last_socket_bbox_size = None
                mask_frozen_pred      = None
                mask_frozen_gray_roi  = None
                mask_is_locked        = False
                seg_engine.reset()
                anomaly_gate.reset()
                latch_gate.reset()
                seq_gate.reset()
                prev_gray_ref[0] = None
                status_dict  = EMPTY_STAT.copy()
                order_status = "N/A"
                detected_seq = []
                peak_status  = EMPTY_STAT.copy()
                peak_seq     = []
                peak_order   = "N/A"

        if socket_latched and not was_latched:
            cycles.start_cycle()
            vote_counter.reset()
            print(f"[INFO] vote counter reset for cycle #{cycles.cycle_no:03d}")
        if was_latched and not socket_latched:
            _finalize_cycle()

        hand_in_roi = hand_future.result() if hand_future is not None else False
        current_dbg = {}

        # ── State Machine & Step 2/3 Execution ────────────────────────────
        if not socket_latched or invisible_roi is None:
            state          = STATE_IDLE
            pred_map       = ZERO_PRED
            status_dict    = EMPTY_STAT
            order_status   = "N/A"
            detected_seq   = []
            mask_is_locked = False
            anomaly_gate.reset()
            latch_gate.reset()
            vis = draw_socket_box(vis, sock_hit, fill_alpha=config.socket.box_fill_alpha)

        elif hand_in_roi:
            # HAND state: tube inference paused, mask never shown
            state          = STATE_HAND
            pred_map       = ZERO_PRED
            status_dict    = EMPTY_STAT
            order_status   = "N/A"
            detected_seq   = []
            current_dbg    = {}
            mask_frozen_pred     = None
            mask_frozen_gray_roi = None
            mask_is_locked       = False
            vis = draw_socket_box(vis, sock_hit, fill_alpha=config.socket.box_fill_alpha)

        else:
            # LIVE INFERENCE / MOTION-GATED MASK FREEZE
            cycle_total_frames += 1
            lock_engage = (
                True if config.seg.lock_engage_mode == "immediate" else warmup_done
            )

            roi_center, roi_bbox_size = compute_socket_roi_geometry(
                sock_hit, last_socket_centre, last_socket_bbox_size)

            def _run_fresh_inference():
                return run_model_tube_segmentation(
                    seg_engine, frame,
                    socket_centre=roi_center,
                    socket_bbox_size=roi_bbox_size,
                    apply_identity_lock=lock_engage,
                    apply_roi_gates=True,
                    config=config
                )

            if not warmup_done or not config.seg.mask_freeze_enabled:
                pred_map = _run_fresh_inference()
                mask_is_locked = False
            else:
                gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rx1, ry1, rx2, ry2 = invisible_roi
                crop_now = gray_now[ry1:ry2, rx1:rx2]

                need_refresh = (
                    mask_frozen_pred is None
                    or mask_frozen_gray_roi is None
                    or mask_has_moved(crop_now, mask_frozen_gray_roi,
                                      thr=config.seg.mask_freeze_motion_thr,
                                      frac=config.seg.mask_freeze_motion_frac,
                                      downscale=config.device.opt_flow_downscale)
                )

                if need_refresh:
                    pred_map = _run_fresh_inference()
                    mask_frozen_pred     = pred_map.copy()
                    mask_frozen_gray_roi = crop_now.copy()
                    mask_is_locked       = False
                else:
                    pred_map       = mask_frozen_pred
                    mask_is_locked = True

            socket_bbox = sock_hit["bbox"] if sock_hit else None
            status_dict, raw_order, detected_seq, current_dbg = evaluate_tube_order(
                pred_map, socket_bbox,
                search_radius=config.inspection.nearest_search_radius,
                min_tube_px=config.seg.min_tube_px,
                expected_seq=config.inspection.expected_seq,
                debug=enable_debug
            )

            stable_order = seq_gate.update(raw_order, detected_seq)
            gate_result  = anomaly_gate.update(stable_order)
            order_status = gate_result
            latch_gate.update(gate_result)

            has_tubes = any((pred_map == ci).any() for ci in (2, 3, 4))

            if not warmup_done:
                warmup_frame_count += 1
                state = STATE_WARMUP
                target = effective_warmup_frames * (warmup_retry + 1)
                if warmup_frame_count >= target:
                    if not has_tubes and warmup_retry < effective_max_retries:
                        warmup_retry += 1
                        print(f"[WARN] f{frame_idx:05d}: warmup blank → retry {warmup_retry}/{effective_max_retries}")
                    else:
                        warmup_done = True
                        # [FIX-I57] Reset gates and temporal state at WARMUP->INSPECT boundary
                        anomaly_gate.reset()
                        seq_gate.reset()
                        latch_gate.reset()
                        seg_engine.identity_ema  = None
                        seg_engine.identity_lock = None
                        seg_engine.ema_probs     = None
                        print(f"[WARMUP DONE] cycle#{cycles.cycle_no:03d} f{frame_idx:05d}")

            else:
                infer_frames += 1
                if not mask_is_locked:
                    vote_counter.record(gate_result)

                state = {"OK":      STATE_NORMAL,
                         "ANOMALY": STATE_ANOMALY,
                         "PARTIAL": STATE_PARTIAL}.get(order_status, STATE_INSPECT)

                if any(v == "Present" for v in status_dict.values()) and order_status != "N/A":
                    peak_status = dict(status_dict)
                    peak_seq    = list(detected_seq)
                    peak_order  = order_status

            if config.seg.show_mask_overlay:
                if has_tubes:
                    vis = draw_seg_overlay(vis, pred_map, alpha=config.seg.overlay_alpha)
                elif seg_engine.last_raw_pred is not None:
                    vis = draw_raw_argmax_fallback(vis, seg_engine.last_raw_pred)

            vis = draw_socket_box(vis, sock_hit, fill_alpha=config.socket.box_fill_alpha)

        if enable_debug and current_dbg:
            vis = draw_debug_overlay(vis, current_dbg)

        frame_ms = (time.perf_counter() - t0) * 1000.0
        frame_ms_ema = (
            frame_ms if frame_idx == 1
            else config.ui.frame_ms_ema_alpha * frame_ms + (1 - config.ui.frame_ms_ema_alpha) * frame_ms_ema
        )
        if frame_ms > config.ui.frame_ms_warn_threshold:
            print(f"[WARN] f{frame_idx:05d}: slow frame {frame_ms:.1f}ms (avg {frame_ms_ema:.1f}ms)")

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
            config           = config,
            total_frames     = total_frames
        )

        cycles.write(vis)
        last_vis = vis

        if show_preview:
            try:
                cv2.imshow(WIN, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit"); break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key == ord("d"):
                    enable_debug = not enable_debug
                    print(f"[INFO] Debug {'ON' if enable_debug else 'OFF'}")
                if key == ord("m"):
                    config.seg.mask_debug = not config.seg.mask_debug
                    print(f"[INFO] Mask debug {'ON' if config.seg.mask_debug else 'OFF'}")
            except Exception:
                show_preview = False

    if cycles.active:
        _finalize_cycle()

    cap.release()
    if show_preview:
        try:
            cv2.destroyWindow(WIN)
        except Exception:
            pass

    return cycles


# ══════════════════════════════════════════════════════════════════════════════
#  10. RUN MANAGER & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_single_video(config: PipelineConfig,
                     print_summary: bool = False,
                     enable_debug: bool = False,
                     forced_channels: Optional[int] = None):
    """Initializes models from configuration and launches the inspection loop."""
    config.device.device = resolve_device(require_gpu=config.device.require_gpu)
    config.device.use_half = bool(config.device.use_half) and config.device.device.startswith("cuda")

    print(f"[INFO] Device              : {config.device.device}")
    print(f"[INFO] [FIX-I39] FP16 half : {config.device.use_half}")
    print(f"[INFO] Input video         : {config.paths.video_path}")
    print(f"[INFO] Warmup              : {config.inspection.warmup_frames} frames")
    print(f"[INFO] Dilate (batched)    : enabled={config.seg.mask_dilate_enabled} size={config.seg.mask_dilate_sz}px")
    print(f"[INFO] Close (batched)     : enabled={config.seg.mask_close_enabled} size={config.seg.mask_close_sz}px")
    print(f"[INFO] Identity Lock       : enabled={config.seg.identity_hysteresis_enabled}")
    print(f"[INFO] Mask ROI Gate       : shape={config.roi.mask_roi_shape} auto_scale={config.roi.mask_roi_auto_scale}")
    print(f"[INFO] Mask Exclusion Zone : enabled={config.roi.mask_exclude_enabled}")
    print(f"[INFO] Mask Freeze         : enabled={config.seg.mask_freeze_enabled}")

    if forced_channels is not None:
        config.seg.in_channels = forced_channels
        print(f"[INFO] in_channels forced  : {config.seg.in_channels}")
    else:
        config.seg.in_channels = detect_in_channels_from_ckpt(config.paths.seg_model_path)

    config.seg.use_radial_channel = (config.seg.in_channels == 4)
    print(f"[INFO] in_channels         : {config.seg.in_channels}  radial={config.seg.use_radial_channel}")

    seg_net = smp.UnetPlusPlus(
        encoder_name="tu-hrnet_w18",
        encoder_weights=None,
        in_channels=config.seg.in_channels,
        classes=config.seg.num_classes,
        activation=None,
    ).to(config.device.device)

    ckpt = torch.load(config.paths.seg_model_path, map_location=config.device.device)
    seg_net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    seg_net.eval()
    if config.device.use_half:
        seg_net = seg_net.half()

    warmup_dtype = torch.float16 if config.device.use_half else torch.float32
    with torch.no_grad():
        seg_net(torch.zeros(1, config.seg.in_channels, *config.seg.img_size,
                            device=config.device.device, dtype=warmup_dtype))
    print("[INFO] GPU warmup done.")
    _log_model_device("Segmentation UNet++", f"{next(seg_net.parameters()).device}  half={config.device.use_half}")

    yolo_socket = load_yolo(config.paths.socket_model_path, "Socket",
                            device=config.device.device, use_half=config.device.use_half,
                            warmup_hw=config.device.gpu_warmup_hw)
    yolo_pose   = load_yolo(config.paths.hand_pose_path, "Pose",
                            device=config.device.device, use_half=config.device.use_half,
                            warmup_hw=config.device.gpu_warmup_hw)

    if not os.path.isfile(config.paths.video_path):
        raise FileNotFoundError(f"Input video not found: {config.paths.video_path}")

    current_date        = datetime.now().strftime("%Y-%m-%d")
    resolved_output_dir = os.path.join(config.paths.out_base, current_date)
    for sub in ("NORMAL", "ANOMALY", "UNKNOWN"):
        Path(os.path.join(resolved_output_dir, sub)).mkdir(parents=True, exist_ok=True)

    cycles = process_video_cycles(
        video_path=config.paths.video_path,
        resolved_output_dir=resolved_output_dir,
        seg_net=seg_net,
        yolo_socket=yolo_socket,
        yolo_pose=yolo_pose,
        config=config,
        print_summary=print_summary,
        enable_debug=enable_debug
    )

    if cycles is not None:
        cycles.final_report()


def parse_cli_args() -> Tuple[PipelineConfig, argparse.Namespace]:
    """Parses command line arguments and populates the structured PipelineConfig."""
    ap = argparse.ArgumentParser(
        description="Diagnostic Inference Engine v50_MergedSingleVideo (Modularized)")
    ap.add_argument("--config",        default="config.yaml" if os.path.isfile("config.yaml") else None,
                    help="Path to YAML configuration file (default: config.yaml if exists).")
    ap.add_argument("--video",         default=None, help="Path to input video.")
    ap.add_argument("--model",         default=None, help="Path to segmentation model weights.")
    ap.add_argument("--out_base", "--out", dest="out_base", default=None,
                    help="Path to output root directory.")
    ap.add_argument("--yolo",          default=None,
                    help="Path to socket YOLO / YOLO-OBB weights.")
    ap.add_argument("--hand_yolo",     default=None, help="Path to YOLO pose weights.")
    ap.add_argument("--channels",      type=int,   default=None,
                    help="Force in_channels (3 or 4). Auto-detected if omitted.")
    ap.add_argument("--warmup",        type=int,   default=None)
    ap.add_argument("--verdict_thr",   type=float, default=None)
    ap.add_argument("--seq_stable",    type=int,   default=None)
    ap.add_argument("--no_sharpening", action="store_true")
    ap.add_argument("--ema_alpha",     type=float, default=None)
    ap.add_argument("--sharpen_temp",  type=float, default=None)
    ap.add_argument("--conf_thr_mid",  type=float, default=None)
    ap.add_argument("--conf_thr_end",  type=float, default=None)
    ap.add_argument("--min_tube_px",   type=int,   default=None)
    ap.add_argument("--min_area_px",   type=int,   default=None)
    ap.add_argument("--search_radius", type=int,   default=None)
    ap.add_argument("--mask_roi_radius", type=int, default=None)
    ap.add_argument("--mask_roi_shape",
                    choices=["circle", "square", "ellipse", "quad_ellipse", "polygon"],
                    default=None)
    ap.add_argument("--mask_roi_up", type=int, default=None)
    ap.add_argument("--mask_roi_down", type=int, default=None)
    ap.add_argument("--mask_roi_left", type=int, default=None)
    ap.add_argument("--mask_roi_right", type=int, default=None)
    ap.add_argument("--no_roi_auto_scale", action="store_true")
    ap.add_argument("--mask_roi_mult_up", type=float, default=None)
    ap.add_argument("--mask_roi_mult_down", type=float, default=None)
    ap.add_argument("--mask_roi_mult_left", type=float, default=None)
    ap.add_argument("--mask_roi_mult_right", type=float, default=None)
    ap.add_argument("--no_mask_exclude", action="store_true")
    ap.add_argument("--no_exclude_auto_scale", action="store_true")
    ap.add_argument("--mask_exclude_mult_offset_x", type=float, default=None)
    ap.add_argument("--mask_exclude_mult_offset_y", type=float, default=None)
    ap.add_argument("--mask_exclude_mult_rx", type=float, default=None)
    ap.add_argument("--mask_exclude_mult_ry", type=float, default=None)
    ap.add_argument("--mask_exclude_offset_x", type=int, default=None)
    ap.add_argument("--mask_exclude_offset_y", type=int, default=None)
    ap.add_argument("--mask_exclude_rx", type=int, default=None)
    ap.add_argument("--mask_exclude_ry", type=int, default=None)
    ap.add_argument("--no_mask_freeze", action="store_true")
    ap.add_argument("--mask_freeze_motion_thr", type=float, default=None)
    ap.add_argument("--mask_freeze_motion_frac", type=float, default=None)
    ap.add_argument("--opt_flow_downscale", type=float, default=None)
    ap.add_argument("--mask_roi_polygon_file", type=str, default=None)
    ap.add_argument("--mask_roi_rx", type=int, default=None)
    ap.add_argument("--mask_roi_ry", type=int, default=None)
    ap.add_argument("--mask_roi_offset_x", type=int, default=None)
    ap.add_argument("--mask_roi_offset_y", type=int, default=None)
    ap.add_argument("--socket_grace",  type=int,   default=None)
    ap.add_argument("--dilate_sz",     type=int,   default=None)
    ap.add_argument("--close_sz",      type=int,   default=None)
    ap.add_argument("--overlay_alpha", type=float, default=None)
    ap.add_argument("--socket_fill_alpha", type=float, default=None)
    ap.add_argument("--no_dilate",     action="store_true")
    ap.add_argument("--identity_margin", type=float, default=None)
    ap.add_argument("--no_identity_lock", action="store_true")
    ap.add_argument("--lock_engage_mode", choices=["post_warmup", "immediate"], default=None)
    ap.add_argument("--locked_conf_floor", type=float, default=None)
    ap.add_argument("--frame_ms_warn", type=float, default=None)
    ap.add_argument("--perf_roi_pad", type=int, default=None)
    ap.add_argument("--no_perf_roi", action="store_true")
    ap.add_argument("--hsv_gate",      action="store_true")
    ap.add_argument("--print_summary", action="store_true")
    ap.add_argument("--no_preview",    action="store_true")
    ap.add_argument("--debug",         action="store_true")
    ap.add_argument("--mask_debug",    action="store_true")
    ap.add_argument("--no_require_gpu", action="store_true")
    ap.add_argument("--half",          dest="half", action="store_true", default=None)
    ap.add_argument("--no_half",       dest="half", action="store_false")
    ap.add_argument("--gpu_warmup_hw", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    args = ap.parse_args()

    # 1. Base Configuration: load from YAML if specified/available, otherwise use defaults
    if args.config and os.path.isfile(args.config):
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # 2. CLI Overrides (applied only if explicitly provided)
    if args.video is not None:
        config.paths.video_path = args.video
    if args.model is not None:
        config.paths.seg_model_path = args.model
    if args.yolo is not None:
        config.paths.socket_model_path = args.yolo
    if args.hand_yolo is not None:
        config.paths.hand_pose_path = args.hand_yolo
    if args.out_base is not None:
        config.paths.out_base = args.out_base

    if args.no_require_gpu:
        config.device.require_gpu = False
    if args.half is not None:
        config.device.use_half = args.half
    if args.opt_flow_downscale is not None:
        config.device.opt_flow_downscale = args.opt_flow_downscale
    if args.gpu_warmup_hw:
        config.device.gpu_warmup_hw = tuple(args.gpu_warmup_hw)

    if args.socket_grace is not None:
        config.socket.reset_grace_frames = args.socket_grace
    if args.socket_fill_alpha is not None:
        config.socket.box_fill_alpha = args.socket_fill_alpha

    if args.mask_roi_radius is not None: config.roi.mask_roi_radius = args.mask_roi_radius
    if args.mask_roi_shape is not None: config.roi.mask_roi_shape = args.mask_roi_shape
    if args.mask_roi_rx is not None: config.roi.mask_roi_radius_x = args.mask_roi_rx
    if args.mask_roi_ry is not None: config.roi.mask_roi_radius_y = args.mask_roi_ry
    if args.mask_roi_up is not None: config.roi.mask_roi_radius_up = args.mask_roi_up
    if args.mask_roi_down is not None: config.roi.mask_roi_radius_down = args.mask_roi_down
    if args.mask_roi_left is not None: config.roi.mask_roi_radius_left = args.mask_roi_left
    if args.mask_roi_right is not None: config.roi.mask_roi_radius_right = args.mask_roi_right
    if args.no_roi_auto_scale: config.roi.mask_roi_auto_scale = False
    if args.mask_roi_mult_up is not None: config.roi.mask_roi_mult_up = args.mask_roi_mult_up
    if args.mask_roi_mult_down is not None: config.roi.mask_roi_mult_down = args.mask_roi_mult_down
    if args.mask_roi_mult_left is not None: config.roi.mask_roi_mult_left = args.mask_roi_mult_left
    if args.mask_roi_mult_right is not None: config.roi.mask_roi_mult_right = args.mask_roi_mult_right
    if args.mask_roi_offset_x is not None: config.roi.mask_roi_offset_x = args.mask_roi_offset_x
    if args.mask_roi_offset_y is not None: config.roi.mask_roi_offset_y = args.mask_roi_offset_y

    if args.no_mask_exclude: config.roi.mask_exclude_enabled = False
    if args.no_exclude_auto_scale: config.roi.mask_exclude_auto_scale = False
    if args.mask_exclude_mult_offset_x is not None: config.roi.mask_exclude_offset_mult_x = args.mask_exclude_mult_offset_x
    if args.mask_exclude_mult_offset_y is not None: config.roi.mask_exclude_offset_mult_y = args.mask_exclude_mult_offset_y
    if args.mask_exclude_mult_rx is not None: config.roi.mask_exclude_radius_mult_x = args.mask_exclude_mult_rx
    if args.mask_exclude_mult_ry is not None: config.roi.mask_exclude_radius_mult_y = args.mask_exclude_mult_ry
    if args.mask_exclude_offset_x is not None: config.roi.mask_exclude_offset_x = args.mask_exclude_offset_x
    if args.mask_exclude_offset_y is not None: config.roi.mask_exclude_offset_y = args.mask_exclude_offset_y
    if args.mask_exclude_rx is not None: config.roi.mask_exclude_radius_x = args.mask_exclude_rx
    if args.mask_exclude_ry is not None: config.roi.mask_exclude_radius_y = args.mask_exclude_ry

    if args.perf_roi_pad is not None: config.roi.perf_roi_pad = args.perf_roi_pad
    if args.no_perf_roi: config.roi.perf_roi_enabled = False

    if args.mask_roi_polygon_file:
        with open(args.mask_roi_polygon_file, "r") as f:
            pts = json.load(f)
        config.roi.mask_roi_polygon = [(int(p[0]), int(p[1])) for p in pts]

    if args.ema_alpha is not None: config.seg.ema_alpha = args.ema_alpha
    if args.no_sharpening: config.seg.boundary_sharpening = False
    if args.sharpen_temp is not None: config.seg.sharpen_temp = args.sharpen_temp
    if args.conf_thr_mid is not None: config.seg.class_conf_thr[3] = args.conf_thr_mid
    if args.conf_thr_end is not None: config.seg.class_conf_thr[4] = args.conf_thr_end
    if args.min_tube_px is not None: config.seg.min_tube_px = args.min_tube_px
    if args.min_area_px is not None: config.seg.min_area_px = args.min_area_px
    if args.dilate_sz is not None: config.seg.mask_dilate_sz = args.dilate_sz
    if args.close_sz is not None:
        config.seg.mask_close_sz = args.close_sz
        config.seg.mask_close_enabled = (args.close_sz > 0)
    if args.no_dilate: config.seg.mask_dilate_enabled = False
    if args.overlay_alpha is not None: config.seg.overlay_alpha = args.overlay_alpha
    if args.identity_margin is not None: config.seg.tube_identity_margin = args.identity_margin
    if args.no_identity_lock: config.seg.identity_hysteresis_enabled = False
    if args.lock_engage_mode is not None: config.seg.lock_engage_mode = args.lock_engage_mode
    if args.locked_conf_floor is not None: config.seg.locked_pixel_min_conf = args.locked_conf_floor
    if args.no_mask_freeze: config.seg.mask_freeze_enabled = False
    if args.mask_freeze_motion_thr is not None: config.seg.mask_freeze_motion_thr = args.mask_freeze_motion_thr
    if args.mask_freeze_motion_frac is not None: config.seg.mask_freeze_motion_frac = args.mask_freeze_motion_frac
    if args.mask_debug: config.seg.mask_debug = True
    if args.hsv_gate: config.seg.use_hsv_gate = True

    if args.warmup is not None: config.inspection.warmup_frames = args.warmup
    if args.verdict_thr is not None: config.inspection.verdict_thr = args.verdict_thr
    if args.seq_stable is not None: config.inspection.min_seq_stable = args.seq_stable
    if args.search_radius is not None: config.inspection.nearest_search_radius = args.search_radius

    if args.no_preview: config.ui.show_preview = False
    if args.frame_ms_warn is not None: config.ui.frame_ms_warn_threshold = args.frame_ms_warn

    return config, args


if __name__ == "__main__":
    cfg, parsed_args = parse_cli_args()
    run_single_video(
        config          = cfg,
        print_summary   = parsed_args.print_summary,
        enable_debug    = parsed_args.debug,
        forced_channels = parsed_args.channels,
    )