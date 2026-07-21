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
)

# Import routers
from app.routes import (
    batches,
    video_runs,
    cycles,
    logs,
    upload,
    ws,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# API Routers
# ===========================

app.include_router(batches.router)
app.include_router(video_runs.router)
app.include_router(cycles.router)
app.include_router(logs.router)
app.include_router(upload.router)
app.include_router(ws.router)

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