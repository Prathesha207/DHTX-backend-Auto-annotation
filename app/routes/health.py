import time
import psutil
from fastapi import APIRouter
from app.services.job_queue import job_queue
from app.database.database import SessionLocal
from app.models.batch import Batch

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()

@router.get("")
def get_health():
    import os
    from pathlib import Path
    
    # 1. Hardware/GPU Check
    import psutil
    gpu_available = False
    gpu_name = "None"
    gpu_memory = "0MB"
    try:
        import subprocess
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        )
        if smi_output.strip():
            parts = smi_output.strip().split(',')
            gpu_name = parts[0].strip()
            mem_mb = int(parts[1].strip())
            gpu_memory = f"{round(mem_mb / 1024, 1)}GB"
            gpu_available = True
    except Exception:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory = f"{round(total_memory_bytes / (1024**3), 1)}GB"
        except Exception:
            pass
        
    ram_gb = f"{round(psutil.virtual_memory().total / (1024**3), 1)}GB"

    # 2. Database Check
    db_ok = False
    try:
        from sqlalchemy import text
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    # 3. Models Loaded Check
    from app.services.model_manager import ModelManager
    models_loaded = ModelManager.get_instance().is_ready.is_set()

    # 4. WebSocket Check
    from app.services.websocket_manager import manager
    ws_ok = manager.loop is not None and manager.loop.is_running()

    # 5. Folders Check
    BASE_DIR = Path(__file__).resolve().parents[2]
    uploads_ok = (BASE_DIR / 'uploads').exists()
    outputs_ok = (BASE_DIR / 'outputs').exists()

    # Calculate final status
    is_ready = db_ok and models_loaded and ws_ok and uploads_ok and outputs_ok
    status = "READY" if is_ready else "INITIALIZING"
    if job_queue.is_unhealthy:
        status = "UNHEALTHY"

    return {
        "status": status,
        "database": db_ok,
        "models_loaded": models_loaded,
        "websocket": ws_ok,
        "gpu": gpu_available,
        "hardware": {
            "gpu": gpu_name,
            "gpu_memory": gpu_memory,
            "ram": ram_gb
        },
        "uploads": uploads_ok,
        "outputs": outputs_ok,
        "version": "2.0.0",
        "uptime": time.time() - START_TIME
    }
