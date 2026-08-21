import asyncio
import base64
import logging
import queue
import threading

import cv2

logger = logging.getLogger(__name__)

class FrameStreamer:
    """
    A dedicated background thread that consumes un-annotated frames and telemetry
    from the ML pipeline, JPEG-encodes them, and pushes them to an asyncio Queue
    for WebSocket delivery.

    Also maintains a per-video latest-snapshot dict so that newly connecting
    browsers can immediately receive the current frame + telemetry without
    waiting for the next live push.
    """
    def __init__(self):
        # Bounded queue from ML thread to encoder thread (prevents ML blocking)
        self.input_queue = queue.Queue(maxsize=5)
        
        # Bounded queue from encoder thread to WebSocket route
        self.ws_queue = None
        self.loop = None

        # Per-video snapshots: { video_id: { "image": ..., "telemetry": ... } }
        # Used by live.py to replay current state on WebSocket connect.
        self._snapshots: dict[str, dict] = {}
        self._snapshot_lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True, name="FrameStreamerThread")
        self._thread.start()

    def attach_loop(self, loop: asyncio.AbstractEventLoop):
        """Called by the FastAPI startup event to provide the running event loop."""
        self.loop = loop
        # We must initialize the asyncio.Queue inside the active event loop
        self.ws_queue = asyncio.Queue(maxsize=5)

    def push(self, frame, telemetry):
        """Called synchronously by the ML thread inside hooked_draw_hud."""
        # We don't want to block the ML thread, and we always want the newest frame!
        # If full, drop the oldest frame before putting the new one.
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
                
        try:
            self.input_queue.put_nowait((frame, telemetry))
        except queue.Full:
            pass

    def _run(self):
        """Background thread loop."""
        while True:
            try:
                frame, telemetry = self.input_queue.get()
                
                # Heavy JPEG encoding happens here, OFF the ML thread!
                # Downscale by 50% for the live view to drastically reduce bandwidth
                # and allow the frontend React app to render at 30+ FPS smoothly.
                height, width = frame.shape[:2]
                small_frame = cv2.resize(frame, (width // 2, height // 2))
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                success, buffer = cv2.imencode('.jpg', small_frame, encode_param)
                
                if success and self.loop:
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    payload = {
                        "image": b64_str,
                        "telemetry": telemetry
                    }

                    # Store snapshot keyed by video_id so reconnecting
                    # browsers can get the current state immediately.
                    vid = telemetry.get("video_id") if telemetry else None
                    if vid:
                        with self._snapshot_lock:
                            self._snapshots[vid] = payload

                    # Safely push to the asyncio event loop
                    self.loop.call_soon_threadsafe(self._push_to_ws_queue, payload)
            except Exception as e:
                logger.error("FrameStreamer encoding error: %s", e, exc_info=True)

    def _push_to_ws_queue(self, payload):
        """Executes inside the asyncio event loop."""
        if not self.ws_queue:
            return
            
        if self.ws_queue.full():
            # Drop the oldest frame to maintain realtime latency
            try:
                self.ws_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        
        try:
            self.ws_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def get_latest_snapshot(self, video_id: str) -> dict | None:
        """Return the snapshot for a specific video_id.

        Returns None if no snapshot exists for that video.
        No fallback — caller must know which video they want.
        """
        with self._snapshot_lock:
            return self._snapshots.get(video_id)

    def clear_snapshot(self, video_id: str) -> None:
        """Remove only the snapshot for the finished video.

        Does NOT affect snapshots belonging to other videos.
        """
        with self._snapshot_lock:
            self._snapshots.pop(video_id, None)

# Global instance to be imported by the hooks and the websocket route
streamer = FrameStreamer()
