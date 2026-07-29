from app.database.database import SessionLocal
from app.models.batch import Batch
from app.services.job_queue import job_queue

db = SessionLocal()
b = db.query(Batch).filter(Batch.id == 80).first()
if b:
    b.status = "queued"
    for v in b.video_runs:
        v.status = "queued"
    db.commit()
    print("Batch 80 reset to queued.")
    
    # Send it to the queue manually for testing
    job_queue.enqueue_batch(80)
    print("Batch 80 enqueued to worker.")
else:
    print("Batch 80 not found.")
