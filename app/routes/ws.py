from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.websocket_manager import manager

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/{batch_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    batch_id: int,
):

    await manager.connect(batch_id, websocket)

    try:

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(batch_id, websocket)
