from dataclasses import dataclass


@dataclass
class InferenceSettings:
    """
    Global inference settings.

    Every optimization in the ML pipeline should read from here.
    """

    # ==========================================================
    # Integration Configuration
    # ==========================================================
    # Toggle between "legacy" (inference_video_full_detection.py)
    # and "new" (inference_video_full_detection_new.py)
    ml_pipeline: str = "new"

    # ==========================================================
    # Frame Processing
    # ==========================================================

    # Run inference every Nth frame
    frame_skip: int = 1

    # Ignore first N frames
    warmup_frames: int = 20

    # Wait N frames after anomaly before checking again
    cooldown_frames: int = 15

    # ==========================================================
    # Feature Toggles
    # ==========================================================

    # Skip expensive models if no hand is detected
    enable_hand_skip: bool = True

    # Stop inference after anomaly until cycle finishes
    stop_after_anomaly: bool = True

    # Enable majority voting
    enable_majority_vote: bool = True

    # Enable confidence smoothing
    enable_confidence_buffer: bool = True

    # Enable ROI optimization
    enable_roi_crop: bool = False

    # ==========================================================
    # Majority Voting
    # ==========================================================

    majority_vote_window: int = 5

    anomaly_votes_required: int = 3

    # ==========================================================
    # Confidence
    # ==========================================================

    anomaly_confidence_threshold: float = 0.60

    normal_confidence_threshold: float = 0.60

    # ==========================================================
    # Performance
    # ==========================================================

    measure_model_time: bool = True

    measure_fps: bool = True

    save_processed_video: bool = True

    save_excel: bool = True

    save_logs: bool = True

    # ==========================================================
    # Debug
    # ==========================================================

    verbose_logging: bool = True


# Global singleton used by the backend
settings = InferenceSettings()