import os
import sys
from app.core.config import config_manager
from sqlalchemy import text
from app.database.database import engine

def validate_startup():
    errors = []

    # 1. Validate Models Exist
    from app.core.config import SEGMENTATION_MODEL_NAME, POSE_MODEL_NAME, SOCKET_MODEL_NAME
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    best_pt_path = os.path.join(models_dir, "ml", SEGMENTATION_MODEL_NAME)
    yolo_pt_path = os.path.join(models_dir, "ml", POSE_MODEL_NAME)
    
    if not os.path.exists(best_pt_path):
        raise RuntimeError(f"Segmentation model not found: {SEGMENTATION_MODEL_NAME}")
    if not os.path.exists(yolo_pt_path):
        errors.append(f"Missing required model: {yolo_pt_path}")

    # 2. Validate DB Schema/Connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        errors.append(f"Database validation failed: {e}")

    # 3. Output directories
    paths = config_manager.get("paths", {})
    output_dir = paths.get("output_dir", "outputs")
    logs_dir = paths.get("logs_dir", "logs")

    for d in [output_dir, logs_dir]:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), d)
        try:
            os.makedirs(full_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create directory {full_path}: {e}")

    # 4. GPU/CPU initialization
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[STARTUP] GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("[STARTUP] No GPU detected, using CPU fallback.")
    except Exception as e:
        errors.append(f"PyTorch initialization failed: {e}")

    if errors:
        print("\n==============================================")
        print("         STARTUP VALIDATION FAILED            ")
        print("==============================================")
        for err in errors:
            print(f"- {err}")
        print("==============================================\n")
        return False
        
    return True

if __name__ == "__main__":
    if not validate_startup():
        sys.exit(1)
