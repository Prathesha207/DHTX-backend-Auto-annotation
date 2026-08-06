import cv2
import numpy as np
from typing import Optional, Dict, Any
from models.renderers.base_renderer import BaseVisionRenderer, VisionMetadata

def draw_seg_overlay(frame, pred_map, alpha=None):
    from models.inference_video_full_detection import CLASS_INFO, OVERLAY_ALPHA
    out = frame.copy()
    a   = alpha if alpha is not None else OVERLAY_ALPHA
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show:
            continue
        mask = pred_map == ci
        if not mask.any():
            continue
        layer        = np.zeros_like(frame)
        layer[mask]  = bgr
        out = cv2.addWeighted(out, 1.0, layer, a, 0)
    return out

def draw_raw_argmax_fallback(frame, raw_pred):
    from models.inference_video_full_detection import CLASS_INFO
    out = frame.copy()
    for ci, (_, bgr, show) in CLASS_INFO.items():
        if not show:
            continue
        mask = raw_pred == ci
        if not mask.any():
            continue
        layer       = np.zeros_like(frame)
        layer[mask] = tuple(int(v * 0.40) for v in bgr)
        out = cv2.addWeighted(out, 1.0, layer, 0.50, 0)
    H, W = out.shape[:2]
    cv2.putText(out, "RAW ARGMAX (pre-filter)", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 200), 1, cv2.LINE_AA)
    return out

def draw_socket_box(frame, hit, fill_alpha=None):
    from models.inference_video_full_detection import SOCKET_BOX_FILL_ALPHA, CLS_SOCKET
    if hit is None:
        return frame

    fill_alpha = SOCKET_BOX_FILL_ALPHA if fill_alpha is None else fill_alpha

    x1, y1, x2, y2 = hit["bbox"]
    is_p  = hit["class"] == CLS_SOCKET

    col      = (0, 220, 100) if is_p else (50, 50, 230)
    fill_col = (150, 255, 190) if is_p else (140, 140, 255)

    label = f"{'Socket' if is_p else 'No Socket'}  {hit['conf']*100:.0f}%"

    obb_pts = hit.get("obb_points")

    overlay = frame.copy()
    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], fill_col)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_col, -1)
    frame = cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0)

    if obb_pts:
        pts = np.array(obb_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=col,
                      thickness=2, lineType=cv2.LINE_AA)
        label_x = min(p[0] for p in obb_pts)
        label_y = min(p[1] for p in obb_pts)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label_x, label_y = x1, y1

    fs = 0.60
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    by = max(label_y - 6, th + 6)
    cv2.rectangle(frame, (label_x, by - th - 6), (label_x + tw + 10, by + bl), col, -1)
    cv2.putText(frame, label, (label_x + 5, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


class VisionOverlayRenderer(BaseVisionRenderer):
    """
    Renderer responsible for drawing raw computer vision outputs:
    segmentations, bounding boxes, labels, and contours.
    """
    
    def render(self, frame: np.ndarray, vision_metadata: VisionMetadata) -> np.ndarray:
        vis = frame.copy()
        
        # Original state-dependent rendering logic
        from models.inference_video_full_detection import (
            STATE_MODEL2, STATE_SKIP, STATE_WAIT_REMOVAL, STATE_CYCLE_COMPLETE
        )
        
        sm_state = getattr(vision_metadata, 'sm_state', '')
        has_tubes = getattr(vision_metadata, 'has_tubes', False)
        socket_hit = getattr(vision_metadata, 'socket_hit', None)
        
        if getattr(vision_metadata, 'hand_in_roi', False):
            # When hand is in ROI, inference is paused and segmentation masks are NEVER shown.
            pass
        elif sm_state == STATE_MODEL2:
            if has_tubes and getattr(vision_metadata, 'pred_map', None) is not None:
                vis = draw_seg_overlay(vis, vision_metadata.pred_map)
            elif getattr(vision_metadata, 'raw_pred', None) is not None:
                vis = draw_raw_argmax_fallback(vis, vision_metadata.raw_pred)
        elif sm_state in (STATE_SKIP, STATE_WAIT_REMOVAL, STATE_CYCLE_COMPLETE):
            # FIX: Only draw frozen masks if the socket is STILL physically present in the frame.
            # This prevents masks from floating in mid-air after the socket is removed.
            if socket_hit is not None and getattr(vision_metadata, 'mask_frozen_pred', None) is not None:
                if has_tubes:
                    vis = draw_seg_overlay(vis, vision_metadata.mask_frozen_pred)
                elif getattr(vision_metadata, 'raw_pred', None) is not None:
                    vis = draw_raw_argmax_fallback(vis, vision_metadata.raw_pred)
                
        if socket_hit is not None:
            vis = draw_socket_box(vis, socket_hit)
            
        return vis
