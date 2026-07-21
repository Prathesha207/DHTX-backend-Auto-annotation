from pathlib import Path
import shutil
from uuid import uuid4

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
    def save_video(
        file: UploadFile,
        upload_root: str,
    ) -> str:

        upload_dir = Path(upload_root)
        upload_dir.mkdir(parents=True, exist_ok=True)

        extension = Path(file.filename).suffix.lower()

        if extension not in UploadService.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        filename = f"{uuid4().hex}{extension}"

        destination = upload_dir / filename

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return str(destination)

    @staticmethod
    def save_videos(
        files: list[UploadFile],
        upload_root: str,
    ) -> list[str]:

        saved_files = []

        for file in files:
            saved_files.append(
                UploadService.save_video(
                    file=file,
                    upload_root=upload_root,
                )
            )

        return saved_files

    @staticmethod
    def create_output_directory(
        output_root: str,
    ) -> str:

        output_dir = (
            Path(output_root)
            / uuid4().hex
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return str(output_dir)