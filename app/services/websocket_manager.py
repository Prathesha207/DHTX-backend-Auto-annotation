import asyncio
from collections import defaultdict
from fastapi import WebSocket


class WebSocketManager:

    def __init__(self):
        self.connections = defaultdict(list)
        self.loop: asyncio.AbstractEventLoop | None = None   # set once at startup

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def connect(self, batch_id: int, websocket: WebSocket):
        await websocket.accept()
        self.connections[batch_id].append(websocket)

    def disconnect(self, batch_id: int, websocket: WebSocket):
        if websocket in self.connections[batch_id]:
            self.connections[batch_id].remove(websocket)

    async def send(self, batch_id: int, data: dict):
        dead = []
        for ws in self.connections[batch_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(batch_id, ws)

    def send_threadsafe(self, batch_id: int, data: dict):
        """
        THIS is what every service should call instead of asyncio.create_task().
        Safe to call from anywhere — the main event loop, a background task, or
        a plain threading.Thread. It hands the coroutine back to the loop that
        actually owns the WebSocket connections, instead of trying (and failing)
        to create a task on a thread that has no loop at all.
        """
        if self.loop is None:
            return   # server hasn't finished starting yet
        asyncio.run_coroutine_threadsafe(self.send(batch_id, data), self.loop)


manager = WebSocketManager()