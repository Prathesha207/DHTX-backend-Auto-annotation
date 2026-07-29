import sys

# [FIX] Ensure stdout/stderr use UTF-8 regardless of the console/terminal
# encoding. This prevents UnicodeEncodeError when the ML pipeline prints
# box-drawing characters (═, █, →, etc.) on a Windows cp1252 console.
# errors='replace' ensures a bad character becomes '?' rather than crashing.
# This complements the PYTHONIOENCODING=utf-8 set in start_backend.bat and
# the Electron spawn env — either fix alone is sufficient, both together
# guarantee coverage across all deployment modes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager
import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database.database import Base, engine

# Import models so SQLAlchemy creates all tables
from app.models import (
    batch,
    video_run,
    cycle,
    log,
    inference_config,
)

# Import routers
from app.routes import (
    batches,
    video_runs,
    cycles,
    logs,
    upload,
    ws,
    settings,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    # Seed inference configuration defaults
    from app.crud.inference_config import get_config
    from app.services.recovery_manager import RecoveryManager
    from app.database.database import SessionLocal
    with SessionLocal() as db:
        get_config(db)
        RecoveryManager.recover_orphaned_jobs(db)

    from app.services.websocket_manager import manager
    manager.set_loop(asyncio.get_running_loop())   # ADD — fixes the crash

    from app.services.model_manager import ModelManager
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_DIR = BASE_DIR / "models" / "ml"
    
    # Initialize and load models once at startup
    model_manager = ModelManager.get_instance()
    model_manager.load_models(
        seg_model_path=str(MODEL_DIR / "best_model_finetuned_manual.pth"),
        yolo_socket_path=str(MODEL_DIR / "best.pt"),
        pose_model_path=str(MODEL_DIR / "yolov8n-pose.pt")
    )


    print("=" * 60)
    print("SQLite database initialized.")
    
    # Initialize background job queue
    from app.services.job_queue import job_queue
    
    yield

app = FastAPI(
    title="DHTX Auto Annotation API",
    description="Desktop AI Inference Application Metadata API",
    version="1.0.0",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    for component in openapi_schema.get("components", {}).get("schemas", {}).values():
        if "properties" not in component:
            continue
        for prop in component["properties"].values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop["format"] = "binary"
                del prop["contentMediaType"]
            elif prop.get("type") == "array" and isinstance(prop.get("items"), dict):
                items = prop["items"]
                if items.get("contentMediaType") == "application/octet-stream":
                    items["format"] = "binary"
                    del items["contentMediaType"]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
import time
from app.services.api_logger import api_logger

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = f"{process_time:.2f}ms"
    
    # Log the API request details
    api_logger.info(f"[API REQUEST] {request.method} {request.url.path} | Status: {response.status_code} | Time: {formatted_process_time}")
    
    return response


# ===========================
# API Routers
# ===========================

app.include_router(batches.router)
app.include_router(video_runs.router)
app.include_router(cycles.router)
app.include_router(logs.router)
app.include_router(upload.router)
app.include_router(ws.router)
app.include_router(settings.router)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def read_root():
    return {
        "message": "DHTX Auto Annotation API is running..."
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )