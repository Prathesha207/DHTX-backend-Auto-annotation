import re

from sqlalchemy.orm import Session

from app.services.video_run_service import VideoRunService


_PROGRESS_PATTERN = re.compile(r"\[PROGRESS\]\s*frame=(\d+)")


class ProgressParser:

    @staticmethod
    def parse(
        db: Session,
        *,
        line: str,
        video,
    ) -> bool:
        """
        Parses ML stdout.

        Example:

        [PROGRESS] frame=152
        """

        match = _PROGRESS_PATTERN.search(line)

        if not match:
            return False

        current_frame = int(match.group(1))

        total_frames = video.total_frames or 0

        VideoRunService.update_progress(
            db=db,
            video=video,
            current_frame=current_frame,
            total_frames=total_frames,
        )

        return True