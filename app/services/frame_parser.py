import re
from app.services.websocket_manager import manager

_FRAME_PATTERN = re.compile(r"^\[FRAME\]\s*(.*)$")

class FrameParser:
    @staticmethod
    def parse(
        *,
        line: str,
        batch_id: int,
        video_id: int,
    ) -> bool:
        match = _FRAME_PATTERN.search(line)
        if not match:
            return False

        b64_data = match.group(1).strip()
        manager.send_threadsafe(
            batch_id,
            {
                "type": "frame",
                "video_id": video_id,
                "base64": b64_data,
            },
        )
        return True
