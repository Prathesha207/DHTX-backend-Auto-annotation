import os
import json

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")

DEFAULT_CONFIG = {
    "config_version": 1,
    "gpu_options": {
        "vram_sweep_frequency": 10,
        "sweep_on_oom": True
    },
    "queue_settings": {
        "max_concurrent_batches": 1,
        "max_retries_recoverable": 3
    },
    "paths": {
        "output_dir": "outputs",
        "temp_dir": "temp",
        "logs_dir": "logs"
    },
    "logging": {
        "level": "INFO",
        "log_frames": False
    }
}

# Single source of truth for model filenames
SEGMENTATION_MODEL_NAME = "best_model_optimized.pth"
POSE_MODEL_NAME = "yolov8n-pose.pt"
SOCKET_MODEL_NAME = "best.pt"

class ConfigManager:
    _instance = None

    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def load_config(self):
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Basic merge to ensure new keys in DEFAULT_CONFIG exist in loaded config
                for k, v in DEFAULT_CONFIG.items():
                    if k not in data:
                        data[k] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_k not in data[k]:
                                data[k][sub_k] = sub_v
                                
                self.config = data
            except Exception as e:
                print(f"[CONFIG ERROR] Failed to load config.json: {e}")
        else:
            self.save_config()

    def save_config(self):
        try:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to save config.json: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

config_manager = ConfigManager.get_instance()
