from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, models, upload, storage, inference, settings, live, batch
from services.frame_streamer import streamer
from database import database
from database import models as db_models
from pathlib import Path
import asyncio
from services.queue_service import inference_worker, recover_interrupted_jobs

# Create the database tables & ensure schema
db_models.Base.metadata.create_all(bind=database.engine)
database.ensure_db_schema()

inference_task = None
app = FastAPI(title="DHTX Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_seed():
    db = database.SessionLocal()
    from utils.storage_discovery import get_writable_default_location
    try:
        setting = db.query(db_models.StorageSetting).first()
        if not setting:
            default_path = get_writable_default_location()
            if default_path:
                new_setting = db_models.StorageSetting(
                    root_path=default_path,
                    is_active=True
                )
                db.add(new_setting)
                db.commit()
                
        # Ensure default ModelSetting exists
        val_setting = db.query(db_models.ModelSetting).first()
        if not val_setting:
            new_val = db_models.ModelSetting(is_active=True)
            db.add(new_val)
            db.commit()
    finally:
        db.close()
    
    # Phase 9: Crash recovery — reset any PROCESSING → QUEUED before starting worker
    recover_interrupted_jobs()

    # Start the sequential background worker
    global inference_task
    inference_task = asyncio.create_task(inference_worker())
    
    # Attach the active event loop to the frame streamer
    streamer.attach_loop(asyncio.get_running_loop())

# Include the routes from our api folder
app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(storage.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(batch.router, prefix="/api")

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "DHTX Backend is running"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content="", media_type="image/x-icon", status_code=204)

@app.on_event("shutdown")
async def shutdown_event():
    from services.inference_service import request_global_shutdown
    from services.queue_service import recover_interrupted_jobs
    request_global_shutdown()
    print("🛑 FastAPI shutdown: terminating ML inference threads...")
    # Clean up any jobs that were processing when stopped
    recover_interrupted_jobs(is_shutdown=True)
