# DHTX Backend Architecture Audit – Model Execution & Statistics Analysis

## 1. Complete Inference Pipeline

The DHTX backend processes video sequentially through a strictly orchestrated state machine designed to minimize computational waste.

**Trace Path:**
1. **Inference Starts**: Triggered by a websocket command, `ml_runner.py` initializes the video capture and the `InferenceStateMachine`.
2. **Who Calls the Models**: The `InferenceStateMachine` explicitly controls execution. `ml_runner` reads `cap.read()` and pushes frames to `state_machine.run()`, which dispatches to state-specific handlers.
3. **Execution Order**: `WAIT_FOR_SOCKET` → `MODEL1_VALIDATION` → `MODEL2_SKIP` → `MODEL2_VALIDATION` → `WAIT_SOCKET_REMOVAL` → `CYCLE_FINISHED`.
4. **Frame Lifecycle**: Decoded → Handed to State Machine → State Logic Evaluated → HUD Drawn → Frame Written to Video (`cycle_mgr.write()`) → Emitted to Frontend via Websocket → Discarded.
5. **When Each Model Starts/Stops**: 
   - **Model 1 (Socket)**: Starts immediately and runs continuously on *every* active frame.
   - **YOLO-Pose (Hand)**: Starts only when a socket is locked (`MODEL1_VALIDATION` and `MODEL2_VALIDATION`). Paused otherwise.
   - **Model 2 (Tube Segmentation)**: Starts *only* in `MODEL2_VALIDATION` and stops the moment a final verdict is reached.
6. **Cycle Boundaries**:
   - **Starts**: Upon successful completion of `MODEL1_VALIDATION` (the database placeholder is created and recording begins).
   - **Ends**: When the socket is physically removed (`socket_absent_frames` triggered in `WAIT_SOCKET_REMOVAL`), triggering `CYCLE_FINISHED`.

---

## 2. Socket Detection Analysis

**How it Works:** 
Socket Detection uses a lightweight YOLO model to locate the main assembly.

- **Executed every frame?** Yes, it is the anchor of the entire system. It runs in almost every state.
- **Does it stop after a socket is found?** No, it continues tracking the socket to ensure it hasn't been moved or removed.
- **Does it cache results?** It implicitly uses temporal streaks (`socket_absent_count`) to prevent flickering.
- **Does it skip frames?** No, it processes every incoming frame without skipping.
- **PASS/FAIL Conditions:** A pass requires the socket to be present for `model1_pass_frames` out of a rolling `model1_frame_count` buffer. Failure/Abort occurs if `socket_loss_abort_frames` consecutive misses happen.
- **Key Variables:** `last_sock_hit`, `last_socket_centre`, `invisible_roi`, `m1_valid_pass`, `m1_valid_total`, `socket_absent_count`.

---

## 3. Hand Detection Analysis

**How it Works:**
Hand Detection uses YOLO-Pose to ensure the operator's hands are not occluding the tubes during critical inspections.

- **When it Begins:** Only during `MODEL1_VALIDATION` (warmup) and `MODEL2_VALIDATION` (tube inspection).
- **When it Stops:** Paused during `WAIT_FOR_SOCKET`, `MODEL2_SKIP`, and `WAIT_SOCKET_REMOVAL` to save GPU cycles.
- **How Presence is Decided:** It checks if any detected hand keypoints fall within the `invisible_roi` (a padded bounding box around the socket).
- **Success/Failure:** If a hand is present, `hand_in_roi` becomes `True`. This *pauses* the validation counters (both M1 and M2) so the system waits rather than failing the inspection due to an occluded tube.
- **Existing Variables:** `hand_in_roi`.

---

## 4. Tube Detection Analysis

**How it Works:**
Tube Detection uses DeepLabV3+ to perform semantic segmentation of the colored tubes.

- **When it Starts:** Strictly at the beginning of `MODEL2_VALIDATION`.
- **When it Stops:** The moment `vote_counter` reaches a conclusive verdict (Normal or Anomaly).
- **Does it run every frame?** Yes, but *only* while in `MODEL2_VALIDATION`.
- **Is the mask cached?** Yes. An Optical Flow EMA (Exponential Moving Average) "Identity Lock" tracks the tubes temporally. Additionally, the final mask (`self.pred_map`) is permanently cached when transitioning to `WAIT_SOCKET_REMOVAL` so it persists on screen.
- **When the mask disappears:** Only when the cycle resets (`_reset_all()`) and transitions back to `WAIT_FOR_SOCKET`.
- **Sequence Validation:** Uses `evaluate_tube_order` to map segmented blobs to real-world coordinates and calculate angles relative to the socket center.
- **NORMAL vs ANOMALY:** Decided by a gating system (`SequenceStabilityGate` and `AnomalyConfirmGate`) which feeds into a `VoteCounter`. If `normal_votes` hits threshold, it's NORMAL. If `anomaly_votes` hits threshold, it's ANOMALY.
- **Variables:** `seg_engine`, `pred_map`, `status_dict`, `order_status`, `detected_seq`, `vote_counter`.

---

## 5. State Machine Audit

| State | Entry Condition | Exit Condition | Models Running | Variables Updated | Events Emitted |
|-------|----------------|----------------|----------------|-------------------|----------------|
| **WAIT_FOR_SOCKET** | System start or cycle reset | Socket detected | Socket (M1) | `m1_valid_total`, `m1_valid_pass` | `socket_detected` |
| **MODEL1_VALIDATION** | Socket found | `m1_valid_pass >= config` | Socket, Hand | `m1_valid_total`, `m1_valid_pass`, `hand_in_roi` | `validation_passed` / `validation_failed` |
| **MODEL2_SKIP** | M1 Validation Passed | `m2_skip_count >= config` | Socket | `m2_skip_count` | None |
| **MODEL2_VALIDATION** | Skip count reached | `votes >= config` | Socket, Hand, Tube | `vote_counter`, `pred_map`, `status_dict` | `inference_result` |
| **WAIT_SOCKET_REMOVAL** | Verdict reached | `socket_absent_count >= config` | Socket | `socket_absent_count` | `remove_socket_prompt` |
| **CYCLE_FINISHED** | Socket physically removed | Auto-transitions instantly | None | Clears cache (`_reset_all`) | `cycle_complete` |

---

## 6. Frame Processing Audit

Exact Execution Order per Frame:
1. **Frame Read**: `ml_runner` calls `cap.read()`.
2. **Perf Start**: `perf.start_frame()` triggers.
3. **Dispatch**: Routed to current State Logic.
4. **Socket (M1)**: `detect_socket` is executed.
5. **Hand (YOLO-Pose)**: Executed *if* required by the active state.
6. **Tube (M2)**: Executed *if* in `MODEL2_VALIDATION`.
7. **Decision Logic**: Gates and Vote Counters updated. State transitions executed if thresholds met.
8. **HUD Drawing**: OpenCV labels and masks are drawn onto `vis`.
9. **Frontend Copy**: A snapshot is converted to Base64 and emitted via websocket.
10. **Recording**: `cycle_mgr.write(vis)` saves the frame to the `.mp4` file.
11. **Telemetry**: `perf.end_frame()` calculates millisecond latency.
12. **Next Frame**: Loop repeats.

---

## 7. Existing Runtime Statistics

These statistics are currently tracked natively in the backend:
- `fps_ema`: Exponential Moving Average of frames per second.
- `frame_ms`: Total latency of the previous frame in milliseconds.
- `frame_idx`: Absolute video frame counter.
- `cycle_no`: The current inspection cycle ID.
- `status_dict`: Dictionary of presence (e.g., Yellow: "Present").
- `detected_seq`: The geometric sequence of the tubes.
- `vote_counter.normal_votes`: Count of positive sequence frames.
- `vote_counter.anomaly_votes`: Count of negative sequence frames.
- `seq_gate._stable_ct`: Number of consecutive identical geometry reads.
- `socket_absent_count`: Number of consecutive frames the socket is missing.
- `m1_valid_pass`: Number of successful warmup frames.
- `perf_summary`: Internal dictionary of execution ms broken down by `model1`, `model2`, `hand`, and `encode`.

---

## 8. Missing Statistics

Based on the existing pipeline, the following statistics can be exposed to a frontend dashboard with **zero ML overhead** (as the math already exists):
- **Total Parts Inspected**: Computed from `cycle_mgr.cycle_no`.
- **Total Passes / Fails**: Already tracked by `cycle_mgr.passed` and `cycle_mgr.failed`.
- **Hand Interference Events**: Count how many times `hand_in_roi` triggered during a run.
- **Average Inspection Time**: Milliseconds between `MODEL1_VALIDATION` start and `WAIT_SOCKET_REMOVAL` start.
- **Model Bottlenecking**: Exposing `perf.frame_summary()` to the UI to show exactly which model (YOLO vs DeepLab) is consuming the most ms per frame.
- **Camera Drops**: Exposing `socket_absent_count` as a "Signal Quality" or "Flicker" metric.

---

## 9. Frontend Statistics Design Recommendations

If you build a real-time React/Next.js dashboard, you should request these payloads from the backend websocket:

**1. Global Session Metrics (Resets per video)**
- **Total Processed / Passed / Failed**: Directly mapped to `cycle_mgr`.
- **Current FPS**: Mapped to `fps_ema`. Updates every frame.

**2. Live Cycle Metrics (Resets every cycle)**
- **Cycle Phase**: Text mapping of `self.state` (e.g., "Warming Up", "Inspecting", "Awaiting Removal").
- **Sequence Confidence**: A progress bar derived from `vote_counter.normal_votes` / `config.model2_pass_frames`.
- **Anomaly Confidence**: A progress bar derived from `vote_counter.anomaly_votes`.

**3. Hardware Diagnostics (Continuous)**
- **Latency Breakdown**: A stacked bar chart derived from `perf.frame_summary()`.
  - M1 (ms)
  - M2 (ms)
  - Hand (ms)
  - Encode (ms)

By utilizing these existing variables, your frontend can look like a highly advanced SCADA/HMI interface without requiring any expensive recalculations in the backend ML code.
