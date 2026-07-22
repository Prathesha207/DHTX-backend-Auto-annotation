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

        from app.models.cycle import Cycle
        existing = db.query(Cycle).filter(
            Cycle.video_run_id == video_run_id,
            Cycle.cycle_number == cycle_number
        ).first()

        if existing:
            existing.start_frame = start_frame
            existing.end_frame = end_frame
            existing.duration_seconds = duration_seconds
            existing.final_verdict = final_verdict
            existing.output_video_path = output_video_path
            existing.tube_blue = tube_blue
            existing.transition_middle = transition_middle
            existing.transition_end = transition_end
            existing.detected_sequence = detected_sequence
            existing.tube_order_result = tube_order_result
            existing.anomaly_ratio = anomaly_ratio
            existing.ok_votes = ok_votes
            existing.anomaly_votes = anomaly_votes
            existing.total_frames = total_frames
            existing.warmup_frames = warmup_frames
            existing.inference_frames = inference_frames
            existing.average_fps = average_fps
            db.commit()
            db.refresh(existing)
            cycle = existing
        else:
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

    @staticmethod
    def create_placeholder(
        db: Session,
        *,
        video_run_id: int,
        cycle_number: int,
        start_frame: int,
    ):
        """
        Create a placeholder cycle row when MODEL1_VALIDATION passes.
        This row will be updated with final values at CYCLE_FINISHED.
        Uses "UNKNOWN" as verdict until finalized.
        """
        return CycleService.save_cycle(
            db=db,
            video_run_id=video_run_id,
            cycle_number=cycle_number,
            start_frame=start_frame,
            final_verdict="UNKNOWN",
            output_video_path="",
        )

    @staticmethod
    def finalize_cycle(
        db: Session,
        *,
        video_run_id: int,
        cycle_number: int,
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
        start_frame: int | None = None,
    ):
        """
        Update the placeholder cycle row with final values.
        Uses save_cycle's upsert behavior to update the existing row.
        """
        return CycleService.save_cycle(
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
        )

    @staticmethod
    def discard_placeholder(
        db: Session,
        *,
        video_run_id: int,
        cycle_number: int,
    ):
        """
        Delete a placeholder cycle row (e.g. when socket is lost mid-cycle
        during MODEL2_SKIP or MODEL2_VALIDATION).
        """
        from app.models.cycle import Cycle
        row = db.query(Cycle).filter(
            Cycle.video_run_id == video_run_id,
            Cycle.cycle_number == cycle_number,
        ).first()
        if row:
            db.delete(row)
            db.commit()