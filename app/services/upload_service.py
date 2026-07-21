from pathlib import Path
import shutil
import re
from uuid import uuid4
from datetime import datetime
from fastapi import UploadFile

class UploadService:

    ALLOWED_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".mpeg",
        ".mpg",
    }

    @staticmethod
    def sanitize_filename(name: str) -> str:
        # Convert non-alphanumeric to underscore
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
        # Collapse multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Strip trailing underscores
        return sanitized.strip('_')

    @staticmethod
    def save_video(
        file: UploadFile,
        upload_root: str,
        sanitized_name: str = None
    ) -> dict:
        upload_dir = Path(upload_root)
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_path = Path(file.filename)
        extension = original_path.suffix.lower()

        if extension not in UploadService.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        if not sanitized_name:
            sanitized_name = UploadService.sanitize_filename(original_path.stem)

        file_uuid = uuid4().hex
        filename = f"{file_uuid}{extension}"
        destination = upload_dir / filename

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "path": str(destination),
            "original_name": sanitized_name,
            "uuid": file_uuid
        }

    @staticmethod
    def save_videos(
        files: list[UploadFile],
        upload_root: str,
    ) -> list[dict]:
        saved_files = []
        name_counts = {}

        for file in files:
            base_name = UploadService.sanitize_filename(Path(file.filename).stem)
            
            if base_name in name_counts:
                name_counts[base_name] += 1
                sanitized_name = f"{base_name}_{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                sanitized_name = base_name

            saved_files.append(
                UploadService.save_video(
                    file=file,
                    upload_root=upload_root,
                    sanitized_name=sanitized_name
                )
            )

        return saved_files

    @staticmethod
    def create_batch_output_directory(
        output_root: str,
    ) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_dir = Path(output_root) / current_date
        date_dir.mkdir(parents=True, exist_ok=True)

        # Find the next Batch_X
        existing_batches = []
        for d in date_dir.iterdir():
            if d.is_dir() and d.name.startswith("Batch_"):
                try:
                    num = int(d.name.split("_")[1])
                    existing_batches.append(num)
                except ValueError:
                    pass
        
        next_num = max(existing_batches) + 1 if existing_batches else 1
        batch_dir = date_dir / f"Batch_{next_num}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        return str(batch_dir)