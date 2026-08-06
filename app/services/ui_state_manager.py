from typing import Dict, Any

class UIStateManager:
    """
    Pure formatter.
    Never tracks frames or makes decisions.
    Simply passes the native ML dictionaries to the UI.
    """
    def __init__(self, config=None):
        pass
        
    def reset(self):
        pass

    def update(self, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the native state and UI dictionaries directly 
        from inference_video_full_detection.py and passes them through.
        """
        # The ML script now natively yields "state", "progress", and "models"
        state = ml_result.get("state", "WARMUP")
        progress = ml_result.get("progress", {"current": 0, "target": 1, "label": "Waiting"})
        models = ml_result.get("models", {
            "Socket Detector": {"state_label": "IDLE", "result": "-", "running": False, "progress_pct": 0},
            "Tube Detector": {"state_label": "IDLE", "result": "-", "running": False, "progress_pct": 0}
        })
        
        # We rename the keys to match what the frontend expects
        # The ML script outputs "socket" and "tube"
        socket_ms = int(ml_result.get("socket_ms", 0))
        tube_ms = int(ml_result.get("tube_ms", 0))

        if "socket" in models:
            models["Socket Detector"] = {
                "state_label": models["socket"]["state"],
                "result": models["socket"]["result"],
                "running": models["socket"]["state"] == "RUNNING",
                "progress_pct": models["socket"]["progress"],
                "elapsed_ms": socket_ms
            }
            del models["socket"]
            
        if "tube" in models:
            models["Tube Detector"] = {
                "state_label": models["tube"]["state"],
                "result": models["tube"]["result"],
                "running": models["tube"]["state"] == "RUNNING",
                "progress_pct": models["tube"]["progress"],
                "elapsed_ms": tube_ms
            }
            del models["tube"]
            
        return {
            "state": state,
            "progress": progress,
            "models": models
        }
