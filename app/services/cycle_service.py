from datetime import datetime

from sqlalchemy.orm import Session

from app.crud.cycle import create_cycle
from app.crud.video_run import get_video_run
from app.services.websocket_manager import manager


class CycleService:

    @staticmethod
    def save_cycle(
        db: Session,
        *,
        video_run_id: int,
        cycle_number: int,
        start_frame: int | None = None,
        end_frame: int | None = None,
        duration_seconds: float | None = None,
        final_verdict: str,
        output_video_path: str,
        tube_blue: str | None = None,
        transition_middle: str | None = None,
        transition_end: str | None = None,
        detected_sequence: str | None = None,
        tube_order_result: str | None = None,
        anomaly_ratio: float | None = None,
        ok_votes: int | None = None,
        anomaly_votes: int | None = None,
        total_frames: int | None = None,
        warmup_frames: int | None = None,
        inference_frames: int | None = None,
        average_fps: float | None = None,
    ):

        cycle = create_cycle(
            db=db,
            video_run_id=video_run_id,
            cycle_number=cycle_number,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_seconds=duration_seconds,
            final_verdict=final_verdict,
            output_video_path=output_video_path,
            tube_blue=tube_blue,
            transition_middle=transition_middle,
            transition_end=transition_end,
            detected_sequence=detected_sequence,
            tube_order_result=tube_order_result,
            anomaly_ratio=anomaly_ratio,
            ok_votes=ok_votes,
            anomaly_votes=anomaly_votes,
            total_frames=total_frames,
            warmup_frames=warmup_frames,
            inference_frames=inference_frames,
            average_fps=average_fps,
            created_at=datetime.now().isoformat(),
        )

        video = get_video_run(db, video_run_id)
        if video:
            manager.send_threadsafe(
                video.batch_id,
                {
                    "type": "cycle",
                    "cycle": cycle_number,
                    "verdict": final_verdict
                }
            )

        return cycle