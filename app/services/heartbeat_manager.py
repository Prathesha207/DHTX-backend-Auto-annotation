import time
import asyncio
from typing import Dict, Any
from fastapi import WebSocket

class HeartbeatManager:
    _instance = None

    def __init__(self):
        self.last_seen: Dict[WebSocket, float] = {}
        self.batch_map: Dict[WebSocket, int] = {}
        self.monitor_task: asyncio.Task | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, websocket: WebSocket, batch_id: int):
        self.last_seen[websocket] = time.time()
        self.batch_map[websocket] = batch_id

    def record_heartbeat(self, websocket: WebSocket):
        if websocket in self.last_seen:
            self.last_seen[websocket] = time.time()

    def unregister(self, websocket: WebSocket):
        self.last_seen.pop(websocket, None)
        self.batch_map.pop(websocket, None)

    def is_timed_out(self, websocket: WebSocket, timeout_seconds: float = 15.0) -> bool:
        last = self.last_seen.get(websocket)
        if last is None:
            return False
        return (time.time() - last) > timeout_seconds

    def start_monitor(self, websocket_manager: Any, loop: asyncio.AbstractEventLoop):
        if self.monitor_task is None or self.monitor_task.done():
            self.monitor_task = loop.create_task(self._monitor_loop(websocket_manager))

    async def _monitor_loop(self, websocket_manager: Any):
        while True:
            try:
                await asyncio.sleep(5.0)
                now = time.time()
                timed_out_ws = []
                for ws, last_time in list(self.last_seen.items()):
                    if (now - last_time) > 15.0:
                        timed_out_ws.append(ws)

                for ws in timed_out_ws:
                    batch_id = self.batch_map.get(ws)
                    print(f"[Heartbeat] Heartbeat Timeout for client on Batch {batch_id}. Removing client.")
                    if batch_id is not None:
                        websocket_manager.disconnect(batch_id, ws)
                    else:
                        self.unregister(ws)
                    try:
                        await ws.close()
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Heartbeat] Monitor loop error: {e}")

heartbeat_mgr = HeartbeatManager.get_instance()
