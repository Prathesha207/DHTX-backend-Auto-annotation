from abc import ABC, abstractmethod
import numpy as np

class BaseMLPlugin(ABC):
    @abstractmethod
    def initialize(self, config: dict):
        """
        Load models and set up the plugin.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray, **kwargs) -> dict:
        """
        Process a single video frame.
        Must return a structured dictionary containing:
        {
            "status": str,
            "verdict": str,
            "detections": list,
            "tracking": dict,
            "metrics": dict
        }
        """
        pass

    @abstractmethod
    def shutdown(self, aborted: bool = True):
        """
        Free memory and cleanup.
        """
        pass
