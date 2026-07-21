import os
import sys
import subprocess
from pathlib import Path

from app.services.inference_settings import settings


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "ml"

SCRIPT_PATH = BASE_DIR / "models" / "inference_video_full_detection.py"

YOLO_MODEL = MODEL_DIR / "best.pt"

POSE_MODEL = MODEL_DIR / "yolov8n-pose.pt"

SEQUENCE_MODEL = MODEL_DIR / "best_model_finetuned_manual.pth"


class InferenceEngine:

    @staticmethod
    def build_command(
        *,
        video_path: str,
        output_dir: str,
    ) -> list[str]:

        command = [
            sys.executable,
            str(SCRIPT_PATH),

            "--video",
            video_path,

            "--model",
            str(SEQUENCE_MODEL),

            "--out_base",
            output_dir,

            "--yolo",
            str(YOLO_MODEL),

            "--hand_yolo",
            str(POSE_MODEL),

            "--print_summary",

            "--render_mode",
            "frontend",
        ]

        # Future configurable arguments are passed via environment variables in build_environment()

        return command

    @staticmethod
    def build_environment():

        env = os.environ.copy()

        env["HEADLESS"] = "1"

        env["FRAME_SKIP"] = str(settings.frame_skip)
        env["WARMUP_FRAMES"] = str(settings.warmup_frames)
        env["COOLDOWN_FRAMES"] = str(settings.cooldown_frames)

        env["ENABLE_HAND_SKIP"] = str(settings.enable_hand_skip)
        env["STOP_AFTER_ANOMALY"] = str(settings.stop_after_anomaly)
        env["ENABLE_MAJORITY_VOTE"] = str(settings.enable_majority_vote)

        return env

    @staticmethod
    def start(
        *,
        video_path: str,
        output_dir: str,
    ):

        command = InferenceEngine.build_command(
            video_path=video_path,
            output_dir=output_dir,
        )

        env = InferenceEngine.build_environment()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        return process