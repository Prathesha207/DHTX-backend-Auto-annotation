import os
import logging
from datetime import datetime
from pathlib import Path

class HourlyFolderHandler(logging.Handler):
    def __init__(self, base_dir="logs", max_bytes=5 * 1024 * 1024, backup_count=3):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_hour = None
        self.file_handler = None
        self._update_handler(init=True)

    def _get_log_path(self, dt: datetime):
        date_str = dt.strftime("%Y-%m-%d")
        hour_str = dt.strftime("%H")
        folder = self.base_dir / date_str / hour_str
        folder.mkdir(parents=True, exist_ok=True)
        session_str = dt.strftime("session_%Y-%m-%d_%H-%M-%S")
        return folder / f"{session_str}.log"

    def _update_handler(self, init=False):
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%d-%H")
        
        if self.current_hour != current_hour:
            if self.file_handler:
                self.file_handler.close()
            
            self.current_hour = current_hour
            log_path = self._get_log_path(now)
            
            from logging.handlers import RotatingFileHandler
            self.file_handler = RotatingFileHandler(
                log_path, maxBytes=self.max_bytes, backupCount=self.backup_count
            )
            if self.formatter:
                self.file_handler.setFormatter(self.formatter)
                
            if not init and self.file_handler and self.formatter:
                # Optional: Log that a new file was created upon rollover
                pass

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        if self.file_handler:
            self.file_handler.setFormatter(fmt)

    def emit(self, record):
        self._update_handler()
        if self.file_handler:
            self.file_handler.emit(record)

def setup_api_logger():
    logger = logging.getLogger("api-service")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = HourlyFolderHandler(base_dir="logs", max_bytes=5*1024*1024, backup_count=3)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        logger.info(f"[LOGGER INIT] Log file created: {handler.file_handler.baseFilename} (max 5MB, 3 backups)")
        
    return logger

api_logger = setup_api_logger()
