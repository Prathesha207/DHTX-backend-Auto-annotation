import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tests.utils.api_client import APIClient
from tests.utils.websocket_client import WSClient
from tests.utils.api_verifier import APIVerifier
from tests.utils.assertions import assert_backend_running

def run():
    # 1. Assert backend running
    assert_backend_running()
    
    # 2. Check health endpoint schema
    client = APIClient()
    health_data = client.get_health()
    assert APIVerifier.verify_health_response(health_data), "Health response schema invalid"
    
    # 3. Check DB opens (implicit if health endpoint passes and returns worker status)
    # The backend handles DB initialization on startup.
    
    # 4. Check WS accepts connection
    ws = WSClient("ws://127.0.0.1:8000/ws/0")
    try:
        ws.connect()
        ws.close()
    except Exception as e:
        raise AssertionError(f"WebSocket connection failed: {e}")
        
if __name__ == "__main__":
    run()
