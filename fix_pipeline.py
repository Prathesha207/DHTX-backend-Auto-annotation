import os

with open('app/services/inference_state_machine.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'def _finalize_current_cycle(self):',
    'def _finalize_current_cycle(self, abort=False):'
)

# Wait, let's also fix the abort handling inside _finalize_current_cycle if needed.
# Since it just does DB/Excel writes, it doesn't crash if we pass abort=True now, it'll just do normal finalization.

old_block = '''        result = run_frame_inference(
            self.seg_engine, frame,
            socket_centre=self.last_socket_centre,
            socket_bbox=sock_hit["bbox"] if sock_hit else None,
            warmup_done=True,  # M2 always runs post-warmup
            enable_debug=self.enable_debug,
        )
        self.perf.end_section("model2")

        self.pred_map = result["pred_map"]
        self.status_dict = result["status_dict"]
        raw_order = result["raw_order"]
        self.detected_seq = result["detected_seq"]
        self.current_dbg = result["dbg"]'''

new_block = '''        self.pred_map = self.seg_engine.infer(
            frame,
            socket_centre=self.last_socket_centre,
            apply_identity_lock=True
        )

        from app.services.ml_adapter import (
            restrict_mask_to_socket_roi, evaluate_tube_order,
            MASK_ROI_CLASSES, MASK_ROI_SHAPE, MASK_ROI_AUTO_SCALE,
            MASK_ROI_RADIUS, MASK_ROI_RADIUS_X, MASK_ROI_RADIUS_Y,
            MASK_ROI_RADIUS_UP, MASK_ROI_RADIUS_DOWN,
            MASK_ROI_RADIUS_LEFT, MASK_ROI_RADIUS_RIGHT,
            MASK_ROI_OFFSET_X, MASK_ROI_OFFSET_Y, MASK_ROI_POLYGON
        )

        sock_bbox = sock_hit["bbox"] if sock_hit else None
        sock_w = (sock_bbox[2] - sock_bbox[0]) if sock_bbox else None
        sock_h = (sock_bbox[3] - sock_bbox[1]) if sock_bbox else None
        bbox_size = (sock_w, sock_h) if sock_w and sock_h else None

        self.pred_map = restrict_mask_to_socket_roi(
            self.pred_map, center=self.last_socket_centre,
            bbox_size=bbox_size,
            classes=MASK_ROI_CLASSES,
            shape=MASK_ROI_SHAPE,
            auto_scale=MASK_ROI_AUTO_SCALE,
            radius=MASK_ROI_RADIUS,
            radius_x=MASK_ROI_RADIUS_X,
            radius_y=MASK_ROI_RADIUS_Y,
            radius_up=MASK_ROI_RADIUS_UP,
            radius_down=MASK_ROI_RADIUS_DOWN,
            radius_left=MASK_ROI_RADIUS_LEFT,
            radius_right=MASK_ROI_RADIUS_RIGHT,
            offset_x=MASK_ROI_OFFSET_X,
            offset_y=MASK_ROI_OFFSET_Y,
            polygon=MASK_ROI_POLYGON
        )

        self.status_dict, raw_order, self.detected_seq, self.current_dbg = evaluate_tube_order(
            self.pred_map, sock_bbox,
            debug=self.enable_debug
        )
        self.perf.end_section("model2")'''

if old_block in content:
    content = content.replace(old_block, new_block)
else:
    print("WARNING: run_frame_inference block not found.")

with open('app/services/inference_state_machine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated _handle_model2_validation and _finalize_current_cycle")
