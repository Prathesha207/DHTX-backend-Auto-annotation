from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.frame_streamer import streamer
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket, video_id: str | None = None):
        """Atomic connect sequence:
        1. Accept the WebSocket connection
        2. Register in active_connections
        3. Send requested video's snapshot (if available)
        4. Start/continue the broadcaster

        The snapshot is sent BEFORE the broadcaster can race with it,
        ensuring the client's first message is always the current state.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        # Replay current state for the requested video on connect.
        # This fires before the broadcaster loop can send the next live
        # frame, so the client always sees the snapshot first.
        if video_id:
            snapshot = streamer.get_latest_snapshot(video_id)
            if snapshot:
                try:
                    await websocket.send_json(snapshot)
                except Exception:
                    pass  # client disconnected immediately

        # Start broadcaster if this is the first client
        if not self.broadcast_task:
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if not self.active_connections and self.broadcast_task:
            self.broadcast_task.cancel()
            self.broadcast_task = None

    async def _broadcast_loop(self):
        while True:
            try:
                # ws_queue is None until attach_loop() is called at startup
                if streamer.ws_queue is None:
                    await asyncio.sleep(0.1)
                    continue

                # Wait for the next fully encoded payload from the streamer
                payload = await streamer.ws_queue.get()
                
                # Send to all connected clients concurrently with a timeout to prevent hanging
                async def send_to_client(connection):
                    try:
                        # 2-second timeout so a dead/frozen TCP connection won't block the loop forever
                        await asyncio.wait_for(connection.send_json(payload), timeout=2.0)
                        return None
                    except Exception:
                        return connection

                # Fire all sends concurrently
                results = await asyncio.gather(*(send_to_client(c) for c in self.active_connections))
                
                # Any connection that returned itself had an error/timeout and is dead
                for dead in results:
                    if dead:
                        self.disconnect(dead)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Broadcast loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.05)  # brief back-off, then retry — task never dies

manager = ConnectionManager()

@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    # Frontend sends: ws://host/api/live?video_id=xxx
    video_id = websocket.query_params.get("video_id")

    await manager.connect(websocket, video_id=video_id)
    try:
        while True:
            # Keep connection alive, wait for client to disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
