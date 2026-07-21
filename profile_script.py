import sys
import time

sys.path.append('.')

from models.inference_video_full_detection import (
    load_yolo, 
    detect_in_channels_from_ckpt,
    NUM_CLASSES, IMG_SIZE, DEVICE,
    detect_socket, detect_hand_in_roi,
    raw_infer
)

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    print("DEVICE:", DEVICE)
    print("PyTorch CUDA available:", torch.cuda.is_available())
    
    t0 = time.time()
    
    # 1. Load models
    print("Loading models...")
    yolo_socket = load_yolo("models/ml/best.pt", "Socket")
    yolo_pose = load_yolo("models/ml/yolov8n-pose.pt", "Pose")
    
    IN_CHANNELS = 3
    seg_net = smp.UnetPlusPlus(
        encoder_name="tu-hrnet_w18",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    ).to(DEVICE)
    ckpt = torch.load("models/ml/best_model_finetuned_manual.pth", map_location=DEVICE, weights_only=False)
    seg_net.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    seg_net.eval()
    
    t1 = time.time()
    print(f"Model Loading Time: {t1 - t0:.3f}s")
    
    cap = cv2.VideoCapture("uploads/56fdc956607d40c98de1f12a757bca02.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Could not read video")
        return
        
    t2 = time.time()
    print(f"Video Decoding (first frame): {t2 - t1:.3f}s")
    
    for i in range(10):
        t_start = time.time()
        sock_hit = detect_socket(yolo_socket, frame, 0.5)
        t_sock = time.time()
        
        hand = detect_hand_in_roi(yolo_pose, frame, (0, 0, 100, 100), 0.5)
        t_hand = time.time()
        
        raw_pred = raw_infer(seg_net, frame)
        t_seg = time.time()
        
        print(f"Frame {i}: YOLO Socket={t_sock-t_start:.3f}s, YOLO Pose={t_hand-t_sock:.3f}s, Seg={t_seg-t_hand:.3f}s, Total={t_seg-t_start:.3f}s")

if __name__ == '__main__':
    main()
