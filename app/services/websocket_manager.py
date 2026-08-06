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
        self.control_queues: Dict[int, asyncio.Queue] = {}
        self.frame_queues: Dict[int, asyncio.Queue] = {}
        self.broadcaster_tasks: Dict[int, asyncio.Task] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        if loop is not None:
            heartbeat_mgr.start_monitor(self, loop)

    def _ensure_broadcaster(self, batch_id: int):
        if batch_id not in self.control_queues and self.loop is not None and not self.loop.is_closed():
            self.control_queues[batch_id] = asyncio.Queue() # Unbounded for critical messages
            self.frame_queues[batch_id] = asyncio.Queue(maxsize=1) # Strict latest-frame strategy
            self.broadcaster_tasks[batch_id] = self.loop.create_task(self._broadcaster_loop(batch_id))

    async def _broadcaster_loop(self, batch_id: int):
        ctrl_q = self.control_queues[batch_id]
        frame_q = self.frame_queues[batch_id]
        
        # 1. Drain control queue first if any exist initially
        while not ctrl_q.empty():
            data = ctrl_q.get_nowait()
            await self.send(batch_id, data)
            ctrl_q.task_done()

        # Create tasks once outside the loop
        ctrl_task = asyncio.create_task(ctrl_q.get())
        frame_task = asyncio.create_task(frame_q.get())
        
        while True:
            try:
                # 2. Wait for EITHER a control message OR a frame
                done, pending = await asyncio.wait(
                    [ctrl_task, frame_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        data = task.result()
                        await self.send(batch_id, data)
                        if task == ctrl_task:
                            ctrl_q.task_done()
                            ctrl_task = asyncio.create_task(ctrl_q.get())
                        else:
                            frame_q.task_done()
                            frame_task = asyncio.create_task(frame_q.get())
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        print(f"[WebSocket] Broadcaster send error for Batch {batch_id}: {e}")
                        # Recreate task if it failed
                        if task == ctrl_task:
                            ctrl_task = asyncio.create_task(ctrl_q.get())
                        else:
                            frame_task = asyncio.create_task(frame_q.get())
                    
            except asyncio.CancelledError:
                ctrl_task.cancel()
                frame_task.cancel()
                break
            except Exception as e:
                print(f"[WebSocket] Broadcaster loop error for Batch {batch_id}: {e}")
                await asyncio.sleep(1)

    async def connect(self, batch_id: int, websocket: WebSocket):
        await websocket.accept()
        self.connections[batch_id].append(websocket)
        heartbeat_mgr.register(websocket, batch_id)
        job_session_mgr.update_client_count(batch_id, len(self.connections[batch_id]))
        self._ensure_broadcaster(batch_id)
        print(f"[WebSocket] Client Connected to Batch {batch_id}. Total clients: {len(self.connections[batch_id])}")

    def disconnect(self, batch_id: int, websocket: WebSocket):
        heartbeat_mgr.unregister(websocket)
        if websocket in self.connections[batch_id]:
            self.connections[batch_id].remove(websocket)
            job_session_mgr.update_client_count(batch_id, len(self.connections[batch_id]))
            print(f"[WebSocket] Client Disconnected from Batch {batch_id}. Remaining clients: {len(self.connections[batch_id])}")
            # If no clients left, we could clean up the queue/task, but it's okay to leave it alive for reconnects.

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
            
        # Inject WebSocket Metadata (Session UUID, Sequence Number, Timestamp)
        import time
        session = job_session_mgr.get_session(batch_id)
        if session:
            session.ws_sequence_number += 1
            data["session_uuid"] = session.session_uuid
            data["sequence"] = session.ws_sequence_number
            data["batch_id"] = batch_id
            if "timestamp" not in data:
                data["timestamp"] = time.time()

        if self.loop is None or self.loop.is_closed():
            return   # server hasn't finished starting yet

        # Ensure broadcaster exists for this batch
        if batch_id not in self.control_queues:
            self.loop.call_soon_threadsafe(self._ensure_broadcaster, batch_id)
            
        def _enqueue():
            if batch_id in self.control_queues:
                msg_type = data.get("type")
                if msg_type in ("frame", "progress"):
                    try:
                        self.frame_queues[batch_id].put_nowait(data)
                    except asyncio.QueueFull:
                        # Drop the older frame completely to guarantee zero streaming latency
                        try:
                            self.frame_queues[batch_id].get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self.frame_queues[batch_id].put_nowait(data)
                else:
                    self.control_queues[batch_id].put_nowait(data)

        self.loop.call_soon_threadsafe(_enqueue)


manager = WebSocketManager()