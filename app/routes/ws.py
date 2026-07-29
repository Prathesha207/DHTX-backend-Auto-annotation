import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import manager
from app.services.job_session_manager import job_session_mgr
from app.services.heartbeat_manager import heartbeat_mgr
from app.database.database import SessionLocal

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/{batch_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    batch_id: int,
):
    await manager.connect(batch_id, websocket)

    # Immediately send complete JobSession snapshot upon connection/reconnection
    try:
        with SessionLocal() as db:
            snapshot = job_session_mgr.get_snapshot_dict(batch_id, db)
            await websocket.send_json(snapshot)
    except Exception as e:
        print(f"[WebSocket] Error sending initial snapshot on Batch {batch_id}: {e}")

    try:
        while True:
            data_text = await websocket.receive_text()
            heartbeat_mgr.record_heartbeat(websocket)
            try:
                data_json = json.loads(data_text)
                if isinstance(data_json, dict) and str(data_json.get("type")).upper() == "PING":
                    await websocket.send_json({"type": "PONG"})
            except Exception:
                if data_text.strip().upper() == "PING":
                    await websocket.send_text("PONG")
    except (WebSocketDisconnect, Exception):
        manager.disconnect(batch_id, websocket)
