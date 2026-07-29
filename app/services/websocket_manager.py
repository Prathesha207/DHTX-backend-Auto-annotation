import asyncio
from collections import defaultdict
from typing import Dict, List
from fastapi import WebSocket
from app.services.job_session_manager import job_session_mgr
from app.services.heartbeat_manager import heartbeat_mgr


class WebSocketManager:

    def __init__(self):
        self.connections: Dict[int, List[WebSocket]] = defaultdict(list)
        self.loop: asyncio.AbstractEventLoop | None = None   # set once at startup

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        if loop is not None:
            heartbeat_mgr.start_monitor(self, loop)

    async def connect(self, batch_id: int, websocket: WebSocket):
        await websocket.accept()
        self.connections[batch_id].append(websocket)
        heartbeat_mgr.register(websocket, batch_id)
        job_session_mgr.update_client_count(batch_id, len(self.connections[batch_id]))
        print(f"[WebSocket] Client Connected to Batch {batch_id}. Total clients: {len(self.connections[batch_id])}")

    def disconnect(self, batch_id: int, websocket: WebSocket):
        heartbeat_mgr.unregister(websocket)
        if websocket in self.connections[batch_id]:
            self.connections[batch_id].remove(websocket)
            job_session_mgr.update_client_count(batch_id, len(self.connections[batch_id]))
            print(f"[WebSocket] Client Disconnected from Batch {batch_id}. Remaining clients: {len(self.connections[batch_id])}")

    def has_clients(self, batch_id: int) -> bool:
        return len(self.connections[batch_id]) > 0

    async def send(self, batch_id: int, data: dict):
        dead = []
        for ws in self.connections[batch_id]:
            if heartbeat_mgr.is_timed_out(ws, timeout_seconds=15.0):
                print(f"[Heartbeat] Heartbeat Timeout for client on Batch {batch_id}. Removing client.")
                dead.append(ws)
                continue
            try:
                await ws.send_json(data)
            except Exception as e:
                print(f"[WebSocket] Frame Send Failure on Batch {batch_id} ({e}). Removing client.")
                dead.append(ws)
        for ws in dead:
            self.disconnect(batch_id, ws)

    def send_threadsafe(self, batch_id: int, data: dict):
        """
        THIS is what every service should call instead of asyncio.create_task().
        Safe to call from anywhere — the main event loop, a background task, or
        a plain threading.Thread. It updates the in-memory JobSession first,
        then hands the coroutine back to the loop that owns the WebSocket connections.
        """
        # Always update the in-memory JobSession so state is preserved even if no sockets are connected
        try:
            job_session_mgr.update_from_payload(batch_id, data)
        except Exception as e:
            print(f"[JobSession] Error updating session from payload: {e}")

        if self.loop is None:
            return   # server hasn't finished starting yet
        try:
            if not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.send(batch_id, data), self.loop)
        except RuntimeError:
            pass # Event loop is closed, ignore


manager = WebSocketManager()