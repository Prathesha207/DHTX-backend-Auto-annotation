import os
import torch
import segmentation_models_pytorch as smp

import sys
from pathlib import Path

# Ensure the models module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.services.ml_adapter import (
    load_yolo, 
    detect_in_channels_from_ckpt, 
    NUM_CLASSES, 
    IMG_SIZE
)

class ModelManager:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg_net = None
        self.yolo_socket = None
        self.yolo_pose = None
        self.loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self, seg_model_path: str, yolo_socket_path: str, pose_model_path: str):
        if self.loaded:
            return

        print(f"[ModelManager] Selected device: {self.device}")
        if self.device == "cuda":
            print(f"[ModelManager] GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"[ModelManager] CUDA Version: {torch.version.cuda}")
        
        # ── Critical: Initialize all GPU globals in the engine module properly ──
        # This calls resolve_device() which sets DEVICE, USE_HALF, _NORM_MEAN, _NORM_STD,
        # enables cudnn.benchmark, TF32 matmul — same path as the standalone script.
        import app.services.ml_adapter as inf_mod
        engine = inf_mod.engine

        # Check Pascal GPU (GTX 10xx) — FP16 is slower than FP32 on sm_6x
        use_half = True
        if self.device == "cuda":
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 7:
                print(f"  [ModelManager] GPU sm_{cap[0]}{cap[1]} (Pascal or older): disabling FP16 for speed.")
                use_half = False

        # Use the engine's resolve_device to set DEVICE + all GPU norm tensors properly
        resolved_device = engine.resolve_device(require_gpu=False)
        engine.DEVICE = resolved_device
        engine.USE_HALF = use_half and resolved_device.startswith("cuda")
        self.device = resolved_device  # keep in sync
        print(f"  [ModelManager] Resolved device: {resolved_device}, USE_HALF: {engine.USE_HALF}")

        # Force-init _NORM_MEAN / _NORM_STD on the correct device NOW
        # (raw_infer lazy-inits them on first call — this pre-empts that with the right device)
        engine._NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=resolved_device).view(1, 3, 1, 1)
        engine._NORM_STD  = torch.tensor([0.229, 0.224, 0.225], device=resolved_device).view(1, 3, 1, 1)

        print("[ModelManager] Loading SegNet...")
        in_channels = detect_in_channels_from_ckpt(seg_model_path)
        self.seg_net = smp.UnetPlusPlus(
            encoder_name="tu-hrnet_w18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=NUM_CLASSES,
            activation=None,
        ).to(self.device)
        ckpt = torch.load(seg_model_path, map_location=self.device, weights_only=False)
        self.seg_net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.seg_net.eval()

        # Warm up: run one dummy forward pass so cuDNN autotunes kernels before real inference
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *IMG_SIZE, device=self.device)
            if engine.USE_HALF:
                dummy = dummy.half()
            self.seg_net(dummy)
        print(f"[ModelManager] SegNet loaded on: {next(self.seg_net.parameters()).device}")
            
        print("[ModelManager] Loading YOLO Socket...")
        self.yolo_socket = load_yolo(yolo_socket_path, "Socket", device=self.device)
        if self.yolo_socket:
            print(f"[ModelManager] YOLO Socket loaded on: {self.yolo_socket.device}")
            
        print("[ModelManager] Loading YOLO Pose...")
        self.yolo_pose = load_yolo(pose_model_path, "Pose", device=self.device)
        if self.yolo_pose:
            print(f"[ModelManager] YOLO Pose loaded on: {self.yolo_pose.device}")
        
        self.loaded = True
        print("[ModelManager] All models loaded and GPU-ready.")

    def get_models(self):
        if not self.loaded:
            BASE_DIR = Path(__file__).resolve().parents[2]
            MODEL_DIR = BASE_DIR / "models" / "ml"
            self.load_models(
                seg_model_path=str(MODEL_DIR / "best_model_finetuned_manual.pth"),
                yolo_socket_path=str(MODEL_DIR / "best.pt"),
                pose_model_path=str(MODEL_DIR / "yolov8n-pose.pt")
            )
        return self.seg_net, self.yolo_socket, self.yolo_pose
