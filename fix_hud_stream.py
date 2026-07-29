import os

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''                # -- Dispatch to current state handler ----------------
                vis = self._dispatch_frame(frame, vis, ZERO_PRED)

                # -- Rendering (HUD + production bar) -----------------'''

new_block = '''                # -- Dispatch to current state handler ----------------
                vis = self._dispatch_frame(frame, vis, ZERO_PRED)
                
                # Copy for frontend stream before HUD text is applied
                frontend_vis = vis.copy()

                # -- Rendering (HUD + production bar) -----------------'''

content = content.replace(old_block, new_block)

old_emit = '''                # -- Emit [FRAME] for live preview --------------------
                if manager.has_clients(self.batch_id):
                    self._emit_frame(vis, src_w)'''

new_emit = '''                # -- Emit [FRAME] for live preview --------------------
                if manager.has_clients(self.batch_id):
                    self._emit_frame(frontend_vis, src_w)'''

content = content.replace(old_emit, new_emit)

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated HUD streaming.")
