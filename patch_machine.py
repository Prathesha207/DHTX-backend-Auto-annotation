import re

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add react HUD import
if 'render_react_style_hud' not in text:
    text = text.replace('import app.services.ml_adapter as inf_mod', 'from utils.react_hud_renderer import render_react_style_hud\nimport os\nimport app.services.ml_adapter as inf_mod')

# 2. Add config to __init__
if 'self.hud_style' not in text:
    text = text.replace('        self._hud_frame_ms_avg = 0.0', '        self._hud_frame_ms_avg = 0.0\n        self.hud_style = os.environ.get("RECORDING_HUD_STYLE", "ML").upper()')

# 3. Replace the HUD drawing in run()
target = '''            # -- Rendering (HUD + production bar) --
            hud_state = self._get_hud_state()

            # Production status bar
            vis = draw_production_status_bar(
                vis, hud_state,
                self.cycle_mgr.cycle_no,
                self.cycle_mgr.passed,
                self.cycle_mgr.failed,
                self.cycle_mgr.unknown,
            )

            # HUD
            vis = draw_hud(
                vis, self._fps_ema, self.frame_idx, hud_state,
                self.last_sock_hit,
                self.status_dict, self.order_status, self.detected_seq,
                anomaly_counter=self.anomaly_gate._count if self.anomaly_gate else 0,
                hand_in_roi=self.hand_in_roi,
                warmup_frame=self.m1_valid_total,
                warmup_retry=0,
                vote_counter=self.vote_counter,
                infer_frames=self.infer_frames,
                seq_stable_ctr=self.seq_gate._stable_ct if self.seq_gate else 0,
                cycle_no=self.cycle_mgr.cycle_no,
            )

            if self.enable_debug and self.current_dbg:
                vis = draw_debug_overlay(vis, self.current_dbg)

            # -- FPS --
            frame_ms = self.perf.end_frame()
            if frame_ms > 0:
                cur_fps = 1000.0 / frame_ms
                self._fps_ema = 0.88 * self._fps_ema + 0.12 * cur_fps

            # -- Emit [STATUS] for frontend --
            if self.frame_idx % 5 == 0 or self.state != self._last_emitted_state:
                self._emit_status()
                self._last_emitted_state = self.state

            # -- Emit [FRAME] for live preview --
            if manager.has_clients(self.batch_id):
                self._emit_frame(vis, src_w)

            # -- Write to cycle video --
            self.cycle_mgr.write(vis)
            self.last_vis = vis'''

replacement = '''            # -- FPS --
            frame_ms = self.perf.end_frame()
            if frame_ms > 0:
                cur_fps = 1000.0 / frame_ms
                self._fps_ema = 0.88 * self._fps_ema + 0.12 * cur_fps

            # -- Emit [STATUS] for frontend --
            if self.frame_idx % 5 == 0 or self.state != self._last_emitted_state:
                self._emit_status()
                self._last_emitted_state = self.state

            # -- Create Temporary Copy for Recording --
            stream_frame = vis
            record_frame = vis.copy()

            # -- Emit [FRAME] for live preview (Clean Frame) --
            if manager.has_clients(self.batch_id):
                self._emit_frame(stream_frame, src_w)

            # -- Rendering (HUD + production bar) for Recording --
            hud_state = self._get_hud_state()
            
            if self.hud_style == "ML":
                # Production status bar
                record_frame = draw_production_status_bar(
                    record_frame, hud_state,
                    self.cycle_mgr.cycle_no,
                    self.cycle_mgr.passed,
                    self.cycle_mgr.failed,
                    self.cycle_mgr.unknown,
                )
    
                # OpenCV HUD
                record_frame = draw_hud(
                    record_frame, self._fps_ema, self.frame_idx, hud_state,
                    self.last_sock_hit,
                    self.status_dict, self.order_status, self.detected_seq,
                    anomaly_counter=self.anomaly_gate._count if self.anomaly_gate else 0,
                    hand_in_roi=self.hand_in_roi,
                    warmup_frame=self.m1_valid_total,
                    warmup_retry=0,
                    vote_counter=self.vote_counter,
                    infer_frames=self.infer_frames,
                    seq_stable_ctr=self.seq_gate._stable_ct if self.seq_gate else 0,
                    cycle_no=self.cycle_mgr.cycle_no,
                )
            elif self.hud_style == "REACT":
                # React Style HUD
                record_frame = render_react_style_hud(
                    frame=record_frame,
                    cycle_no=self.cycle_mgr.cycle_no,
                    fps=float(self._fps_ema),
                    frame_idx=self.frame_idx,
                    state_str=hud_state,
                    sock_hit=self.last_sock_hit,
                    status_dict=self.status_dict,
                    order_status=self.order_status,
                    detected_seq=self.detected_seq,
                    anomaly_count=self.anomaly_gate._count if self.anomaly_gate else 0,
                    hand_in_roi=self.hand_in_roi,
                    warmup_frame=self.m1_valid_total,
                    warmup_retry=0,
                    vote_counter=self.vote_counter,
                    seq_stable_ctr=self.seq_gate._stable_ct if self.seq_gate else 0,
                    passed=self.cycle_mgr.passed,
                    failed=self.cycle_mgr.failed,
                    unknown=self.cycle_mgr.unknown,
                    infer_ms=frame_ms,
                    mask_is_locked=self.mask_is_locked if hasattr(self, 'mask_is_locked') else False,
                )

            if self.enable_debug and self.current_dbg:
                record_frame = draw_debug_overlay(record_frame, self.current_dbg)

            # -- Write to cycle video (Recorded Frame) --
            self.cycle_mgr.write(record_frame)
            self.last_vis = record_frame'''

text = text.replace(target, replacement)

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patch applied successfully.")
