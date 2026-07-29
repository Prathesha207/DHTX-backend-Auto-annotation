import app.main
from app.database.database import SessionLocal
from app.models.batch import Batch
from app.services.ml_runner import MLRunner

db = SessionLocal()
b = db.query(Batch).filter(Batch.id == 80).first()
b.status = "queued"
for v in b.video_runs:
    v.status = "queued"
db.commit()
print("Reset to queued. Starting MLRunner directly.")
runner = MLRunner()
runner.run_batch(80, False)
