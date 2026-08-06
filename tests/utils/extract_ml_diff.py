import difflib

with open('backend/models/inference_video_full_detection.py', 'r', encoding='utf-8') as f1:
    old_lines = f1.readlines()

with open('backend/models/inference_video_full_detection_new.py', 'r', encoding='utf-8') as f2:
    new_lines = f2.readlines()

diff = list(difflib.unified_diff(old_lines, new_lines, fromfile='OLD', tofile='NEW', n=3))

import re

# We want to identify the exact ML logic changes.
# Let's ignore the UI drawing functions.
ignored_funcs = ['draw_seg_overlay', 'draw_raw_argmax_fallback', 'draw_socket_box', 'draw_final_verdict_overlay', 'draw_debug_overlay', 'draw_production_status_bar', 'draw_hud', 'get_verdict_dir', '_finalize_cycle', '_run_fresh_inference']

# Group diff chunks by context
chunks = []
current_chunk = []
for line in diff:
    if line.startswith('@@'):
        if current_chunk:
            chunks.append(current_chunk)
        current_chunk = [line]
    elif current_chunk is not None:
        current_chunk.append(line)
if current_chunk:
    chunks.append(current_chunk)

def should_keep(chunk):
    text = ''.join(chunk)
    # Ignore purely import changes at the top
    if 'import ' in text and 'def ' not in text and 'class ' not in text:
        # Check if it's purely imports
        if all(line.startswith(('+', '-', ' ', '@@')) and ('import ' in line or line.strip() in ['+', '-']) for line in chunk):
            return False
            
    # Ignore chunks that are purely adding/removing the ignored functions
    for func in ignored_funcs:
        if f'def {func}' in text:
            return False
            
    # Ignore the huge block of argument changes in process_video_cycles signature if we already know it
    if 'def process_video_cycles' in text and 'yield_mode' in text:
        # We know this changed, but let's keep it if there's ML logic in the body
        pass
        
    return True

filtered_chunks = [c for c in chunks if should_keep(c)]

with open('ml_diff_filtered.txt', 'w', encoding='utf-8') as f:
    for c in filtered_chunks:
        f.writelines(c)
        f.write('\n')
print(f"Extracted {len(filtered_chunks)} relevant diff chunks out of {len(chunks)} total.")
