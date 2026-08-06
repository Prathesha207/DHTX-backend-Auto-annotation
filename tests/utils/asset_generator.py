import os
import shutil
import logging

logger = logging.getLogger("AssetGenerator")

class AssetGenerator:
    def __init__(self, golden_dir=None, temp_dir=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if golden_dir is None:
            golden_dir = os.path.join(base, "golden")
        if temp_dir is None:
            temp_dir = os.path.join(base, "test_assets")
        self.golden_dir = os.path.abspath(golden_dir)
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_golden_file(self, name):
        return os.path.join(self.golden_dir, name)

    def generate_batch(self, batch_name: str, count: int, source_type="normal_short.mp4"):
        """Generates a folder of `count` videos by copying from golden dataset."""
        batch_folder = os.path.join(self.temp_dir, batch_name)
        os.makedirs(batch_folder, exist_ok=True)
        
        source_path = self.get_golden_file(source_type)
        if not os.path.exists(source_path):
            logger.error(f"Golden source not found: {source_path}")
            return None
            
        generated_paths = []
        for i in range(count):
            dest_name = f"video_{i:04d}.mp4"
            dest_path = os.path.join(batch_folder, dest_name)
            # Use hardlink if possible for speed, otherwise copy
            try:
                os.link(source_path, dest_path)
            except OSError:
                shutil.copy2(source_path, dest_path)
            generated_paths.append(dest_path)
            
        logger.info(f"Generated {count} files in {batch_folder} based on {source_type}")
        return batch_folder, generated_paths

    def clear_temp(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)
            logger.info("Cleared temporary test assets.")
