import logging

logger = logging.getLogger("APIVerifier")

class APIVerifier:
    @staticmethod
    def verify_health_response(data: dict):
        required_keys = ["status", "hardware", "worker"]
        for k in required_keys:
            if k not in data:
                logger.error(f"Health response missing key: {k}")
                return False
        return True

    @staticmethod
    def verify_manifest_response(data: dict):
        if "accepted" not in data or "uploaded_files" not in data:
            logger.error("Manifest response missing expected keys")
            return False
        if not isinstance(data["uploaded_files"], list):
            logger.error("Manifest uploaded_files is not a list")
            return False
        return True

    @staticmethod
    def verify_batch_create_response(data: dict):
        if "batch_id" not in data:
            logger.error("Batch create response missing batch_id")
            return False
        return True

    @staticmethod
    def verify_upload_response(data: dict):
        if "video_run_id" not in data:
            logger.error("Upload response missing video_run_id")
            return False
        return True
