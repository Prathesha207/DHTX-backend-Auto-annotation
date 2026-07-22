PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name          TEXT,
    input_type          TEXT NOT NULL CHECK(input_type IN ('video','folder')),
    input_path          TEXT NOT NULL,
    output_path         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued'
                            CHECK(status IN ('queued','running','completed','cancelled','failed')),
    total_videos        INTEGER NOT NULL,
    completed_videos    INTEGER DEFAULT 0,
    failed_videos       INTEGER DEFAULT 0,
    progress            REAL DEFAULT 0,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    completed_at        TEXT
);

CREATE TABLE IF NOT EXISTS video_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id            INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    input_filename      TEXT NOT NULL,
    input_path          TEXT NOT NULL,
    output_video_path   TEXT,
    excel_report_path   TEXT,
    queue_position      INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued'
                            CHECK(status IN ('queued','running','completed','cancelled','failed')),
    progress            REAL DEFAULT 0,
    current_frame       INTEGER DEFAULT 0,
    total_frames        INTEGER,
    width               INTEGER,
    height              INTEGER,
    fps                 REAL,
    duration_seconds    REAL,
    error_message       TEXT,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    completed_at        TEXT
);

CREATE TABLE IF NOT EXISTS cycles (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_run_id            INTEGER NOT NULL REFERENCES video_runs(id) ON DELETE CASCADE,
    cycle_number             INTEGER NOT NULL,
    start_frame             INTEGER,
    end_frame               INTEGER,
    duration_seconds        REAL,
    final_verdict           TEXT NOT NULL CHECK(final_verdict IN ('NORMAL','ANOMALY','UNKNOWN')),
    output_video_path       TEXT NOT NULL,
    tube_blue               TEXT,
    transition_middle       TEXT,
    transition_end          TEXT,
    detected_sequence       TEXT,
    tube_order_result       TEXT,
    anomaly_ratio           REAL,
    ok_votes                INTEGER,
    anomaly_votes           INTEGER,
    total_frames            INTEGER,
    warmup_frames           INTEGER,
    inference_frames        INTEGER,
    average_fps              REAL,
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id            INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    video_run_id        INTEGER REFERENCES video_runs(id) ON DELETE CASCADE,
    cycle_id            INTEGER,
    frame_number        INTEGER,
    state               TEXT,
    timestamp           TEXT NOT NULL,
    level               TEXT NOT NULL DEFAULT 'info' CHECK(level IN ('info','warning','error','debug','perf')),
    message             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_config (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    model1_frame_count        INTEGER NOT NULL DEFAULT 30,
    model1_pass_frames        INTEGER NOT NULL DEFAULT 28,
    model2_start_skip_frame   INTEGER NOT NULL DEFAULT 10,
    model2_frame_count        INTEGER NOT NULL DEFAULT 20,
    model2_pass_frames        INTEGER NOT NULL DEFAULT 18,
    socket_absent_frames      INTEGER NOT NULL DEFAULT 10,
    socket_loss_abort_frames  INTEGER NOT NULL DEFAULT 15,
    enable_debug_logging      BOOLEAN NOT NULL DEFAULT 0,
    enable_perf_logging       BOOLEAN NOT NULL DEFAULT 1,
    created_at                TEXT,
    updated_at                TEXT
);
