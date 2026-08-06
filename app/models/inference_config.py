from sqlalchemy import Column, Integer, String, Boolean

from app.database.database import Base


class InferenceConfig(Base):

    __tablename__ = "inference_config"

    id = Column(Integer, primary_key=True)  # singleton row, id=1

    # ── Model1 (Socket Detection) ─────────────────────────────────
    model1_frame_count = Column(Integer, nullable=False, default=30)
    model1_pass_frames = Column(Integer, nullable=False, default=28)

    # ── Model2 (Tube Detection) ───────────────────────────────────
    model2_start_skip_frame = Column(Integer, nullable=False, default=10)
    model2_frame_count = Column(Integer, nullable=False, default=20)
    model2_pass_frames = Column(Integer, nullable=False, default=18)

    # ── Socket removal ────────────────────────────────────────────
    socket_absent_frames = Column(Integer, nullable=False, default=10)
    socket_loss_abort_frames = Column(Integer, nullable=False, default=15)

    # ── Logging toggles ──────────────────────────────────────────
    enable_debug_logging = Column(Boolean, nullable=False, default=False)
    enable_perf_logging = Column(Boolean, nullable=False, default=True)
    
    # ── Operational Settings ──────────────────────────────────────
    upload_timeout_seconds = Column(Integer, nullable=False, default=300)
    # ── Watchdog ──────────────────────────────────────────────────
    watchdog_enabled = Column(Boolean, nullable=False, default=True)
    watchdog_warning_seconds = Column(Integer, nullable=False, default=30)
    watchdog_fail_seconds = Column(Integer, nullable=False, default=60)
    
    # ── Streaming ─────────────────────────────────────────────────
    preview_fps = Column(Integer, nullable=False, default=20)
    
    default_output_dir = Column(String, nullable=False, default="outputs")

    # ── Timestamps ───────────────────────────────────────────────
    created_at = Column(String)
    updated_at = Column(String)
