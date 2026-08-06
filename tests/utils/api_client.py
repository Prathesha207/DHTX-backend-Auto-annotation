import requests
import json
import os

class APIClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def get_health(self):
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def create_batch(self, batch_name="Test Batch", total_videos=1):
        resp = requests.post(f"{self.base_url}/upload/batch/create", json={
            "batch_name": batch_name,
            "total_videos": total_videos
        })
        resp.raise_for_status()
        return resp.json()

    def upload_file(self, batch_id, file_path, queue_position=0):
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "video/mp4")}
            data = {"queue_position": queue_position}
            resp = requests.post(f"{self.base_url}/upload/batch/{batch_id}/file", data=data, files=files)
            if resp.status_code == 201:
                return True, resp.json()
            elif resp.status_code == 422:
                # Expected for corrupted uploads
                return False, resp.json()
            else:
                resp.raise_for_status()

    def get_manifest(self, batch_id):
        resp = requests.get(f"{self.base_url}/upload/batch/{batch_id}/manifest")
        resp.raise_for_status()
        return resp.json()

    def start_batch(self, batch_id):
        resp = requests.post(f"{self.base_url}/batches/{batch_id}/start")
        resp.raise_for_status()
        return resp.json()

    def get_batch_status(self, batch_id):
        resp = requests.get(f"{self.base_url}/batches/{batch_id}")
        resp.raise_for_status()
        return resp.json()

    def cancel_batch(self, batch_id):
        resp = requests.post(f"{self.base_url}/batches/{batch_id}/cancel")
        resp.raise_for_status()
        return resp.json()

    def resume_batch(self, batch_id):
        resp = requests.post(f"{self.base_url}/batches/{batch_id}/start")
        resp.raise_for_status()
        return resp.json()
