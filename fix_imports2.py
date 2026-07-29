import os

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '    draw_raw_argmax_fallback,\n    draw_socket_box,\n    draw_production_status_bar,\n    draw_hud,\n    draw_debug_overlay,',
    '    draw_raw_argmax_fallback,\n    draw_socket_box,\n    draw_production_status_bar,\n    draw_hud,\n    draw_debug_overlay,\n    draw_seg_overlay,\n    draw_final_verdict_overlay,'
)

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated imports.")
