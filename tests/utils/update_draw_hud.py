import re

file_path = 'd:\\Dhtx-auto-annotation\\backend\\models\\inference_video_full_detection_new.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pat = re.compile(r'def draw_hud\(frame, fps, frame_idx, state, socket_hit, hand_in_roi, cycle_no, frame_ms, m1_pass=0, m1_target=0, m2_pass=0, m2_target=0\):')
new = 'def draw_hud(frame, fps, frame_idx, state, socket_hit, hand_in_roi, cycle_no, frame_ms, m1_pass=0, m1_target=0, m1_lbl="Frames", m2_pass=0, m2_target=0, m2_lbl="Frames"):'

content = pat.sub(new, content, count=1)

pat2 = re.compile(r'_put\(out, f"\{m1_pass\} / \{m1_target\} Frames", bx, by, FS_SM, \(100, 220, 100\) if m1_pass >= m1_target else \(0, 165, 255\), TK1\)')
new2 = '_put(out, f"{m1_pass} / {m1_target} {m1_lbl}", bx, by, FS_SM, (100, 220, 100) if m1_pass >= m1_target else (0, 165, 255), TK1)'

content = pat2.sub(new2, content, count=1)

pat3 = re.compile(r'_put\(out, f"\{m2_pass\} / \{m2_target\} Frames", bx, by, FS_SM, \(100, 220, 100\) if m2_pass >= m2_target else \(0, 165, 255\), TK1\)')
new3 = '_put(out, f"{m2_pass} / {m2_target} {m2_lbl}", bx, by, FS_SM, (100, 220, 100) if m2_pass >= m2_target else (0, 165, 255), TK1)'

content = pat3.sub(new3, content, count=1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated draw_hud in inference script")
