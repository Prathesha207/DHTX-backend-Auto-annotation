import re

file_path = 'd:\\Dhtx-auto-annotation\\backend\\app\\services\\inference_state_machine.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pat = re.compile(r'm2_pass_val = max\(self\.vote_counter\.normal_votes, self\.vote_counter\.anomaly_votes\) if self\.vote_counter else 0\s*\n\s*m1_pass_val = self\.m1_valid_pass\n\s*if self\.state in \(MODEL2_SKIP, MODEL2_VALIDATION, WAIT_SOCKET_REMOVAL, CYCLE_FINISHED\):\n\s*m1_pass_val = self\.config\.model1_pass_frames')

new = '''
                # Synchronize HUD strings with Frontend progress state
                if self.state == MODEL1_VALIDATION:
                    m1_pass_val = self.m1_valid_total
                    m1_tgt_val = self.config.model1_frame_count
                    m1_lbl = "valid frames"
                    m2_pass_val = 0
                    m2_tgt_val = self.config.model2_pass_frames
                    m2_lbl = "inspected"
                elif self.state == MODEL2_SKIP:
                    m1_pass_val = self.config.model1_pass_frames
                    m1_tgt_val = self.config.model1_pass_frames
                    m1_lbl = "valid frames"
                    m2_pass_val = self.m2_skip_count
                    m2_tgt_val = self.config.model2_start_skip_frame
                    m2_lbl = "skipped"
                elif self.state in (MODEL2_VALIDATION, WAIT_SOCKET_REMOVAL, CYCLE_FINISHED):
                    m1_pass_val = self.config.model1_pass_frames
                    m1_tgt_val = self.config.model1_pass_frames
                    m1_lbl = "valid frames"
                    m2_pass_val = max(self.vote_counter.normal_votes, self.vote_counter.anomaly_votes) if self.vote_counter else 0
                    m2_tgt_val = self.config.model2_pass_frames
                    m2_lbl = "inspected"
                else:
                    m1_pass_val = self.m1_valid_pass
                    m1_tgt_val = self.config.model1_pass_frames
                    m1_lbl = "valid frames"
                    m2_pass_val = 0
                    m2_tgt_val = self.config.model2_pass_frames
                    m2_lbl = "inspected"
'''

content = pat.sub(new.strip(), content, count=1)

# Now update the draw_hud call
pat2 = re.compile(r'm1_pass=m1_pass_val, m1_target=self\.config\.model1_pass_frames,\n\s*m2_pass=m2_pass_val, m2_target=self\.config\.model2_pass_frames')
new2 = 'm1_pass=m1_pass_val, m1_target=m1_tgt_val, m1_lbl=m1_lbl, m2_pass=m2_pass_val, m2_target=m2_tgt_val, m2_lbl=m2_lbl'
content = pat2.sub(new2, content, count=1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated inference_state_machine.py HUD parameters")
