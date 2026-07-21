import json
import re

from app.services.websocket_manager import manager


_STATUS_PATTERN = re.compile(r"\[STATUS\]\s*(\{.*\})\s*$")


class StatusParser:
    """
    Parses the [STATUS] {...json...} lines emitted by
    inference_video_full_detection.py (one per ~5 frames, plus on every
    state change) and broadcasts them to the frontend over the batch's
    websocket as a 'status' message, so the UI can render the inspection
    HUD (cycle/verdict/tube presence/etc.) as real components instead of
    relying on the pixels baked into the output video.

    Example input line:

        [STATUS] {"frame": 95, "fps": 0.9, "state": "NORMAL", ...}
    """

    @staticmethod
    def parse(
        *,
        line: str,
        batch_id: int,
        video_id: int,
    ) -> bool:

        match = _STATUS_PATTERN.search(line)

        if not match:
            return False

        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return False

        manager.send_threadsafe(
            batch_id,
            {
                "type": "status",
                "video_id": video_id,
                **payload,
            },
        )

        return True
