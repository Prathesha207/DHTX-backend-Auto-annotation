import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml
from fastapi import APIRouter, status, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import database
from crud import crud

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

START_TIME = time.monotonic()

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]


CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

MIN_FREE_DISK_GB = float(os.getenv("DHTX_MIN_FREE_DISK_GB", "2.0"))
MIN_FREE_GPU_VRAM_GB = float(os.getenv("DHTX_MIN_FREE_GPU_VRAM_GB", "1.0"))

# ============================================================================
# Response Models
# ============================================================================

class HardwareHealth(BaseModel):
    gpu: str | None
    gpu_available: bool
    gpu_vram_total_gb: float | None
    gpu_vram_free_gb: float | None
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_free_gb: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    model_files: dict[str, bool]
    uploads: bool
    outputs: bool
    hardware: HardwareHealth

# ============================================================================
# GPU
# ============================================================================

def get_gpu_health() -> tuple[bool, str | None, float | None, float | None]:
    gpu_available = False
    gpu_name = None
    free_vram_gb = None
    total_vram_gb = None

    # First try NVIDIA SMI because it is a lightweight diagnostic.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=True,
        )

        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]

        if len(parts) >= 3:
            gpu_name = parts[0]
            free_mb = float(parts[1])
            total_mb = float(parts[2])
            free_vram_gb = round(free_mb / 1024, 2)
            total_vram_gb = round(total_mb / 1024, 2)
            gpu_available = True
            return (gpu_available, gpu_name, free_vram_gb, total_vram_gb)

    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        pass

    # PyTorch fallback.
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_vram_gb = round(free_bytes / (1024 ** 3), 2)
            total_vram_gb = round(total_bytes / (1024 ** 3), 2)
    except Exception:
        pass

    return (gpu_available, gpu_name, free_vram_gb, total_vram_gb)

# ============================================================================
# Model Files
# ============================================================================

def check_model_files() -> dict[str, bool]:
    """Checks whether the required model files exist dynamically from config.yaml."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        paths = config.get("paths", {})
        
        # Resolve paths against BASE_DIR to get absolute paths
        return {
            "unet_segmentation": (BASE_DIR / paths.get("seg_model_path", "missing")).is_file(),
            "yolo_socket": (BASE_DIR / paths.get("socket_model_path", "missing")).is_file(),
            "yolo_pose": (BASE_DIR / paths.get("hand_pose_path", "missing")).is_file()
        }
    except Exception:
        return {
            "unet_segmentation": False,
            "yolo_socket": False,
            "yolo_pose": False
        }

# ============================================================================
# Storage
# ============================================================================

def check_storage(db: Session) -> tuple[bool, bool, float]:
    """Checks upload/output directories and available disk space."""
    storage = crud.get_active_storage(db)
    if storage:
        storage_path = storage.root_path
    else:
        storage_path = str(BASE_DIR)
        
    uploads_dir = Path(storage_path) / "raw"
    outputs_dir = Path(storage_path) / "inferences"
    
    uploads_ok = uploads_dir.is_dir()
    outputs_ok = outputs_dir.is_dir()

    try:
        _, _, free_bytes = shutil.disk_usage(storage_path)
        free_gb = round(free_bytes / (1024 ** 3), 2)
    except OSError:
        free_gb = 0.0

    return (uploads_ok, outputs_ok, free_gb)

# ============================================================================
# Health Endpoint
# ============================================================================

@router.get("", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health(db: Session = Depends(database.get_db)) -> HealthResponse:
    """
    DHTX production health/readiness endpoint.
    Checks required model files, GPU availability, CPU usage, RAM usage, 
    disk space, and IO directories.
    """
    # Hardware (non-blocking CPU check)
    (gpu_available, gpu_name, gpu_vram_free_gb, gpu_vram_total_gb) = get_gpu_health()
    cpu_usage = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    memory_available_gb = round(memory.available / (1024 ** 3), 2)

    # ML Models
    model_files = check_model_files()
    all_model_files_present = all(model_files.values())

    # Infrastructure
    uploads_ok, outputs_ok, disk_free_gb = check_storage(db)

    # Overall readiness
    hardware_ok = (
        gpu_available
        and gpu_vram_free_gb is not None
        and gpu_vram_free_gb >= MIN_FREE_GPU_VRAM_GB
        and disk_free_gb >= MIN_FREE_DISK_GB
    )

    is_ready = all([
        all_model_files_present,
        uploads_ok,
        outputs_ok,
        hardware_ok,
    ])

    system_status = "healthy" if is_ready else "unhealthy"

    return HealthResponse(
        status=system_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(time.monotonic() - START_TIME, 2),
        model_files=model_files,
        uploads=uploads_ok,
        outputs=outputs_ok,
        hardware=HardwareHealth(
            gpu=gpu_name,
            gpu_available=gpu_available,
            gpu_vram_total_gb=gpu_vram_total_gb,
            gpu_vram_free_gb=gpu_vram_free_gb,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory_available_gb,
            disk_free_gb=disk_free_gb,
        ),
    )
