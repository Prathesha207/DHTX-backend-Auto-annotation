import json
import logging
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("WSClient")

class WSClient:
    def __init__(self, uri="ws://127.0.0.1:8000/ws"):
        self.uri = uri
        self.ws = None
        self.events = []
        
    def connect(self):
        self.ws = connect(self.uri)
        
    def close(self):
        if self.ws:
            self.ws.close()
            
    def subscribe(self, batch_id):
        self.send({"type": "subscribe", "batch_id": batch_id})
        
    def send(self, data):
        if self.ws:
            self.ws.send(json.dumps(data))
            
    def recv(self, timeout=None):
        if self.ws:
            try:
                raw = self.ws.recv(timeout)
                data = json.loads(raw)
                self.events.append(data)
                return data
            except TimeoutError:
                return None
            except ConnectionClosed:
                return None
        return None
        
    def clear_events(self):
        self.events = []
        
    def get_events(self, type_filter=None):
        if type_filter:
            return [e for e in self.events if e.get("type") == type_filter]
        return self.events
