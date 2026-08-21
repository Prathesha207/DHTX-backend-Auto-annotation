from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from pathlib import Path
import yaml
import os

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

router = APIRouter(tags=["ML Models"])

class MLModel(BaseModel):
    id: str
    name: str
    path: str
    description: str
    is_active: bool

class ModelsResponse(BaseModel):
    status: str
    models: List[MLModel]

def get_models_from_config() -> List[MLModel]:
    """
    Dynamically reads the config.yaml to ensure the API always matches 
    the actual ML pipeline's source of truth.
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            
        paths = config.get("paths", {})
        
        return [
            MLModel(
                id="unet_segmentation",
                name="Unet++ Tube Segmentation",
                path=paths.get("seg_model_path", "unknown"),
                description="Unet++ model with HRNet-W18 encoder for tube segmentation.",
                is_active=True
            ),
            MLModel(
                id="yolo_socket",
                name="YOLO Socket Detection",
                path=paths.get("socket_model_path", "unknown"),
                description="YOLOv8 Oriented Bounding Box model for socket and cap detection.",
                is_active=True
            ),
            MLModel(
                id="yolo_pose",
                name="YOLO Hand Pose",
                path=paths.get("hand_pose_path", "unknown"),
                description="YOLOv8 Nano Pose model for hand and wrist keypoint detection.",
                is_active=True
            )
        ]
    except Exception as e:
        # Fallback if config is missing
        return []

@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    Returns a structured list of all ML models currently active in the config.
    """
    return ModelsResponse(
        status="success",
        models=get_models_from_config()
    )
