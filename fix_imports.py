import sys

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the import lines
content = content.replace('import models.inference_video_full_detection as inf_mod', 'from app.services import ml_adapter as inf_mod')
content = content.replace('from models.inference_video_full_detection import (', 'from app.services.ml_adapter import (')

# Remove 'run_frame_inference' and 'draw_final_verdict_overlay' from the imports
content = content.replace('    run_frame_inference,\n', '')
content = content.replace('    draw_seg_overlay,\n', '')
content = content.replace('    draw_final_verdict_overlay,\n', '')

# In _handle_model2_validation, update run_frame_inference to use seg_engine.infer
content = content.replace(
'''        raw_pred, vis = run_frame_inference(
            self.seg_engine, frame, vis,
            hud_state["warmup_frame"], self.config.model1_pass_frames
        )''',
'''        
        vis, raw_pred = self.seg_engine.infer(frame, vis)'''
)

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Imports and infer call fixed.")
