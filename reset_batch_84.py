from app.database.database import SessionLocal
from app.models.batch import Batch
from app.models.video_run import VideoRun
from app.models.cycle import Cycle
from app.models.log import Log

db = SessionLocal()
b = db.query(Batch).filter(Batch.id == 84).first()
if b:
    b.status = "queued"
    for v in b.video_runs:
        v.status = "queued"
    db.commit()
    print("Batch 84 reset to queued.")
else:
    print("Batch 84 not found.")
