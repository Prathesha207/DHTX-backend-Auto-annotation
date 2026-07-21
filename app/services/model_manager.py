import os
import torch
import segmentation_models_pytorch as smp

import sys
from pathlib import Path

# Ensure the models module can be imported
sys.path.append(str(Path(__file__).resolve().parents[3]))

from models.inference_video_full_detection import (
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
        
        print("[ModelManager] Loading SegNet...")
        
        # Override the global DEVICE in inference module to ensure it uses the Manager's device
        import models.inference_video_full_detection as inf_mod
        inf_mod.DEVICE = self.device
        
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
        with torch.no_grad():
            self.seg_net(torch.zeros(1, in_channels, *IMG_SIZE, device=self.device))
        print(f"[ModelManager] SegNet loaded and assigned to: {next(self.seg_net.parameters()).device}")
            
        print("[ModelManager] Loading YOLO Socket...")
        self.yolo_socket = load_yolo(yolo_socket_path, "Socket")
        if self.yolo_socket:
            self.yolo_socket.to(self.device)
            print(f"[ModelManager] YOLO Socket loaded and assigned to: {self.yolo_socket.device}")
            
        print("[ModelManager] Loading YOLO Pose...")
        self.yolo_pose = load_yolo(pose_model_path, "Pose")
        if self.yolo_pose:
            self.yolo_pose.to(self.device)
            print(f"[ModelManager] YOLO Pose loaded and assigned to: {self.yolo_pose.device}")
        
        self.loaded = True
        print("[ModelManager] Models loaded successfully and cached.")

    def get_models(self):
        if not self.loaded:
            raise RuntimeError("Models are not loaded yet.")
        return self.seg_net, self.yolo_socket, self.yolo_pose
