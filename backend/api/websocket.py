# backend/api/websocket.py
# ============================================================
# WebSocket connection manager.
# Keeps track of all teacher dashboard connections per class.
# When an alert fires, we push it to every connected teacher.
# ============================================================

from fastapi import WebSocket
from typing import Dict, List
import json


class ConnectionManager:
    def __init__(self):
        # class_id → list of open WebSocket connections
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, class_id: str):
        await websocket.accept()
        self.active.setdefault(class_id, []).append(websocket)

    def disconnect(self, websocket: WebSocket, class_id: str):
        if class_id in self.active:
            try:
                self.active[class_id].remove(websocket)
            except ValueError:
                pass

    async def broadcast(self, class_id: str, payload: dict):
        """Send payload to every teacher watching this class."""
        dead = []
        for ws in self.active.get(class_id, []):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws, class_id)


# Singleton — imported by endpoints.py
manager = ConnectionManager()
