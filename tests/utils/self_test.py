import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.database.database import Base
from app.models.batch import Batch
from app.models.video_run import VideoRun
from tests.utils.db_verifier import DBVerifier
from tests.utils.filesystem_verifier import FilesystemVerifier
from tests.utils.leak_detector import LeakDetector
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestUtilities")

def test_db_verifier():
    logger.info("--- Testing DB Verifier ---")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    # Create valid batch
    b = Batch(id=1, batch_name="Test", input_type="folder", total_videos=2, completed_videos=1)
    db.add(b)
    db.add(VideoRun(id=1, batch_id=1, status="completed"))
    db.add(VideoRun(id=2, batch_id=1, status="running"))
    db.commit()
    
    verifier = DBVerifier(db)
    assert verifier.verify_batch(1) == True, "Should pass valid batch"
    
    # Corrupt total_videos
    b.total_videos = 3
    db.commit()
    assert verifier.verify_batch(1) == False, "Should fail when total_videos is incorrect"
    
    # Corrupt completed_videos
    b.total_videos = 2
    b.completed_videos = 2
    db.commit()
    assert verifier.verify_batch(1) == False, "Should fail when completed_videos is incorrect"
    
    # Orphan rows
    db.add(VideoRun(id=3, batch_id=99, status="queued"))
    db.commit()
    assert verifier.check_orphan_rows() == False, "Should detect orphan rows"
    logger.info("DB Verifier Self-Test PASSED.")

def test_filesystem_verifier():
    logger.info("--- Testing Filesystem Verifier ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        batch_name = "test_fs_batch"
        batch_dir = os.path.join(tmp, batch_name)
        os.makedirs(batch_dir)
        
        # Valid state
        with open(os.path.join(batch_dir, f"{batch_name}_summary.txt"), "w") as f: f.write("ok")
        video_dir = os.path.join(batch_dir, "vid1")
        os.makedirs(video_dir)
        with open(os.path.join(video_dir, "vid1_output.mp4"), "w") as f: f.write("ok")
        with open(os.path.join(video_dir, "vid1_detection_report.xlsx"), "w") as f: f.write("ok")
        
        verifier = FilesystemVerifier(tmp)
        assert verifier.verify_artifacts(batch_name, ["vid1.mp4"]) == True, "Should pass valid filesystem"
        
        # Missing file
        os.remove(os.path.join(video_dir, "vid1_output.mp4"))
        assert verifier.verify_artifacts(batch_name, ["vid1.mp4"]) == False, "Should detect missing file"
        
        # Orphan folder
        with open(os.path.join(video_dir, "vid1_output.mp4"), "w") as f: f.write("ok")
        os.makedirs(os.path.join(batch_dir, "orphan_folder"))
        assert verifier.verify_artifacts(batch_name, ["vid1.mp4"]) == False, "Should detect orphan folder"
        logger.info("Filesystem Verifier Self-Test PASSED.")

def test_leak_detector():
    logger.info("--- Testing Leak Detector ---")
    detector = LeakDetector()
    detector.capture_baseline()
    
    # Valid
    assert detector.check_leaks() == True, "Should pass with no leaks"
    
    # GPU leak
    t = None
    if torch.cuda.is_available():
        t = torch.zeros((1000, 1000, 100), device="cuda")
        # Leak is currently > 100MB threshold, 1000x1000x100x4 bytes = 400MB
        assert detector.check_leaks() == False, "Should detect GPU leak"
        del t
        torch.cuda.empty_cache()
    
    # File leak
    f = open("leak_test_dummy.txt", "w")
    assert detector.check_leaks() == False, "Should detect file leak"
    f.close()
    os.remove("leak_test_dummy.txt")
    
    logger.info("Leak Detector Self-Test PASSED.")

if __name__ == "__main__":
    test_db_verifier()
    test_filesystem_verifier()
    test_leak_detector()
