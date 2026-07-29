import re

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove run_frame_inference and draw_final_verdict_overlay from imports
text = re.sub(r'\s*run_frame_inference,', '', text)
text = re.sub(r'\s*draw_final_verdict_overlay,', '', text)

# Replace run_frame_inference usage
target = '''        result = run_frame_inference(
            self.seg_engine, frame,
            socket_centre=self.last_socket_centre,
            socket_bbox=sock_hit["bbox"] if sock_hit else None,
        )'''
replacement = '''        result = self.seg_engine.infer(
            frame,
            socket_centre=self.last_socket_centre,
            socket_bbox=sock_hit["bbox"] if sock_hit else None,
        )'''
text = text.replace(target, replacement)

# Replace draw_final_verdict_overlay usage
target2 = '''            card = draw_final_verdict_overlay(
                self.last_vis, final_verdict,
                cycle_no=self.cycle_mgr.cycle_no, stats=stats_card)
            self.cycle_mgr.hold_final_frame(card)'''
replacement2 = '''            # Removed draw_final_verdict_overlay
            self.cycle_mgr.hold_final_frame(self.last_vis)'''
text = text.replace(target2, replacement2)

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patch applied")
