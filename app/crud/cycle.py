from sqlalchemy.orm import Session

from app.models.cycle import Cycle


def create_cycle(
    db: Session,
    video_run_id: int,
    cycle_number: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
    duration_seconds: float | None = None,
    final_verdict: str = "PENDING",
    output_video_path: str = "",
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
    created_at: str = "",
) -> Cycle:

    cycle = Cycle(
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
        created_at=created_at,
    )

    db.add(cycle)
    db.commit()
    db.refresh(cycle)

    return cycle


def get_cycle(
    db: Session,
    cycle_id: int,
) -> Cycle | None:

    return (
        db.query(Cycle)
        .filter(Cycle.id == cycle_id)
        .first()
    )

def get_cycles(
    db: Session,
) -> list[Cycle]:

    return (
        db.query(Cycle)
        .order_by(Cycle.id)
        .all()
    )
    
def get_video_cycles(
    db: Session,
    video_run_id: int,
) -> list[Cycle]:

    return (
        db.query(Cycle)
        .filter(Cycle.video_run_id == video_run_id)
        .order_by(Cycle.cycle_number)
        .all()
    )

def get_video_summary(
    db: Session,
    video_run_id: int,
):
    return (
        db.query(Cycle)
        .filter(Cycle.video_run_id == video_run_id)
        .all()
    )
    
def delete_cycle(
    db: Session,
    cycle: Cycle,
):

    db.delete(cycle)
    db.commit()


def update_cycle(
    db: Session,
    cycle_id: int,
    final_verdict: str,
    anomaly_ratio: float = 0.0,
    normal_votes: int = 0,
    anomaly_votes: int = 0,
) -> Cycle | None:
    cycle = db.query(Cycle).filter(Cycle.id == cycle_id).first()
    if not cycle:
        return None
    
    cycle.final_verdict = final_verdict
    cycle.anomaly_ratio = anomaly_ratio
    cycle.ok_votes = normal_votes
    cycle.anomaly_votes = anomaly_votes
    
    db.commit()
    db.refresh(cycle)
    return cycle