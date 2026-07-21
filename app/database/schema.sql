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
    timestamp           TEXT NOT NULL,
    level               TEXT NOT NULL DEFAULT 'info' CHECK(level IN ('info','warning','error')),
    message             TEXT NOT NULL
);
