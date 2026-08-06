import os
import hashlib
import logging

logger = logging.getLogger("FilesystemVerifier")

class FilesystemVerifier:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def calculate_checksum(self, filepath):
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {filepath}: {e}")
            return None

    def verify_artifacts(self, batch_name, expected_video_names):
        batch_dir = os.path.join(self.output_dir, batch_name)
        if not os.path.exists(batch_dir):
            logger.error(f"Batch folder {batch_dir} does not exist.")
            return False
            
        success = True
        
        # Check summary exists
        summary_path = os.path.join(batch_dir, "batch_summary.txt")
        if not os.path.exists(summary_path):
            logger.error(f"Summary file missing: {summary_path}")
            success = False
            
        for name in expected_video_names:
            base_name = os.path.splitext(name)[0]
            
            # The backend puts videos and excels directly in the batch folder
            out_video = os.path.join(batch_dir, f"{base_name}_processed.mp4")
            if not os.path.exists(out_video):
                logger.error(f"Output video missing: {out_video}")
                success = False
                
            excel = os.path.join(batch_dir, f"{base_name}_detection_report.xlsx")
            if not os.path.exists(excel):
                logger.error(f"Excel report missing: {excel}")
                success = False
                
        return success
