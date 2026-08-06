import cv2
import numpy as np
import os
from pathlib import Path
from models.renderers.base_renderer import BaseRenderer, FrameMetadata
from models.inference_video_full_detection import (
    IN_CHANNELS, USE_RADIAL_CHANNEL,
    STATE_IDLE, STATE_WARMUP,
    STATE_HAND, STATE_INSPECT, STATE_NORMAL, STATE_ANOMALY,
    STATE_PARTIAL, _PROD_STATUS, _CHIP_LABEL, N_ANOMALY_CONFIRM,
    WARMUP_FRAMES, MAX_WARMUP_RETRIES, _ROI_COL_AMBER, MIN_SEQ_STABLE,
    TUBE_LABELS, TUBE_SHORT, FRAME_MS_WARN_THRESHOLD, CLASS_INFO, CLS_SOCKET
)

_STATE_STYLE = {
    STATE_IDLE:    {"chip_bg": (45, 45, 45),   "chip_fg": (150, 150, 150), "border": (80, 80, 80)},
    STATE_WARMUP:  {"chip_bg": (100, 80, 10),  "chip_fg": (255, 220, 50),  "border": (180, 140, 20)},
    STATE_HAND:    {"chip_bg": (0, 120, 210),  "chip_fg": (255, 255, 255), "border": (0, 165, 255)},
    STATE_INSPECT: {"chip_bg": (20, 90, 170),  "chip_fg": (255, 255, 255), "border": (40, 130, 210)},
    STATE_NORMAL:  {"chip_bg": (10, 140, 40),  "chip_fg": (255, 255, 255), "border": (30, 200, 60)},
    STATE_ANOMALY: {"chip_bg": (15, 15, 210),  "chip_fg": (255, 255, 255), "border": (30, 30, 240)},
    STATE_PARTIAL: {"chip_bg": (20, 130, 160), "chip_fg": (255, 255, 255), "border": (40, 175, 195)},
}

def _put(img, text, x, y, fs, col, thick=1):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick, cv2.LINE_AA)

def draw_production_status_bar(frame, state, cycle_no, passed, failed, unknown):
    out  = frame.copy()
    H, W = out.shape[:2]
    S    = W / 1280.0

    sty        = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
    status_txt = _PROD_STATUS.get(state, state)
    cycle_txt  = f"CYCLE #{cycle_no:03d}" if cycle_no else "CYCLE  --"

    fs_main = max(0.75, 0.85 * S)
    fs_sub  = max(0.55, 0.62 * S)
    pad_x   = max(18, int(22 * S))
    pad_y   = max(10, int(12 * S))
    gap     = max(28, int(34 * S))

    seg1 = cycle_txt
    seg2 = f"STATUS: {status_txt}"
    seg3 = f"PASSED: {passed}"
    seg4 = f"FAILED: {failed}"

    (w1, h1), _ = cv2.getTextSize(seg1, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w2, h2), _ = cv2.getTextSize(seg2, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w3, h3), _ = cv2.getTextSize(seg3, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)
    (w4, h4), _ = cv2.getTextSize(seg4, cv2.FONT_HERSHEY_DUPLEX, fs_main, 2)

    total_w = w1 + w2 + w3 + w4 + gap * 3 + pad_x * 2
    bar_h   = max(h1, h2, h3, h4) + pad_y * 2
    bx1 = (W - total_w) // 2
    by1 = max(8, int(10 * S))
    bx2 = bx1 + total_w
    by2 = by1 + bar_h

    ovl = out.copy()
    cv2.rectangle(ovl, (int(bx1), int(by1)), (int(bx2), int(by2)), (12, 12, 12), -1)
    cv2.addWeighted(ovl, 0.80, out, 0.20, 0, out)
    cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), sty["border"], 2)

    ty = by2 - pad_y
    cx = bx1 + pad_x
    _put(out, seg1, cx, ty, fs_main, (255, 255, 255), 2);  cx += w1 + gap
    cv2.line(out, (int(cx - gap // 2), int(by1 + 6)), (int(cx - gap // 2), int(by2 - 6)), (90, 90, 90), 1)
    _put(out, seg2, cx, ty, fs_main, sty["chip_fg"] if state not in (STATE_NORMAL, STATE_ANOMALY)
         else (60, 230, 60) if state == STATE_NORMAL else (60, 60, 255), 2)
    cx += w2 + gap
    cv2.line(out, (int(cx - gap // 2), int(by1 + 6)), (int(cx - gap // 2), int(by2 - 6)), (90, 90, 90), 1)
    _put(out, seg3, cx, ty, fs_main, (60, 230, 60), 2);  cx += w3 + gap
    cv2.line(out, (int(cx - gap // 2), int(by1 + 6)), (int(cx - gap // 2), int(by2 - 6)), (90, 90, 90), 1)
    _put(out, seg4, cx, ty, fs_main, (60, 60, 255), 2)

    if unknown:
        sub = f"unknown: {unknown}"
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 1)
        _put(out, sub, (W - sw) // 2, by2 + int(18 * S), fs_sub, (150, 150, 150), 1)

    return out


def draw_hud(frame, fps, frame_idx, state, socket_hit,
             status_dict, order_status, detected_seq,
             anomaly_counter=0, hand_in_roi=False,
             warmup_frame=0, warmup_retry=0,
             vote_counter=None, infer_frames=0,
             seq_stable_ctr=0, cycle_no=0, frame_ms=0.0, frame_ms_avg=0.0,
             mask_is_locked=False):
    out  = frame.copy()
    H, W = out.shape[:2]
    S    = W / 1280.0
    PAD  = max(10, int(12 * S))
    FS_XS = max(0.40, 0.42 * S);  FS_SM = max(0.48, 0.52 * S)
    FS_MD = max(0.58, 0.62 * S);  FS_LG = max(0.70, 0.76 * S)
    TK1   = max(1, int(S));        TK2   = max(1, int(2 * S))
    ROW   = max(26, int(28 * S)); DOT   = max(5, int(6 * S))

    TOP_OFFSET = max(70, int(78 * S))

    LP_W  = max(280, int(300 * S));  LP_H = PAD * 2 + ROW * 10 + 8
    LP_X, LP_Y = 10, 10 + TOP_OFFSET
    RP_W  = max(300, int(325 * S));  RP_H = PAD * 2 + ROW * 9 + 20
    RP_X  = W - RP_W - 10;  RP_Y = 10 + TOP_OFFSET

    ovl = out.copy()
    for (px, py, pw, ph) in [(LP_X, LP_Y, LP_W, LP_H),
                             (RP_X, RP_Y, RP_W, RP_H)]:
        cv2.rectangle(ovl, (int(px), int(py)), (int(px + pw), int(py + ph)), (14, 14, 14), -1)
    cv2.addWeighted(ovl, 0.72, out, 0.28, 0, out)

    sty = _STATE_STYLE.get(state, _STATE_STYLE[STATE_IDLE])
    for (px, py, pw, ph), bc in [
        ((LP_X, LP_Y, LP_W, LP_H), sty["border"]),
        ((RP_X, RP_Y, RP_W, RP_H), (70, 70, 70))
    ]:
        cv2.rectangle(out, (int(px), int(py)), (int(px + pw), int(py + ph)), bc, 1)

    lx = LP_X + PAD;  ly = LP_Y + PAD + ROW - 4;  vx = lx + max(60, int(64 * S))
    _put(out, "FPS",   lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{fps:5.1f}", vx, ly, FS_LG, (220, 220, 220), TK2);  ly += ROW + 4
    _put(out, "FRAME", lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_idx:06d}", vx, ly, FS_MD, (200, 200, 200), TK1);  ly += ROW + 4

    ms_col = (60, 60, 255) if frame_ms_avg >= FRAME_MS_WARN_THRESHOLD else (200, 200, 200)
    _put(out, "MS",    lx, ly, FS_XS, (120, 120, 120), TK1)
    _put(out, f"{frame_ms:5.1f} (avg {frame_ms_avg:5.1f})",
         vx, ly, FS_SM, ms_col, TK1);  ly += ROW + 6

    chip = _CHIP_LABEL.get(state, state)
    (cw, ch), bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, FS_SM, TK1)
    cpx, cpy = max(10, int(11 * S)), max(6, int(7 * S))
    cx1, cy1 = lx, ly;  cx2, cy2 = cx1 + cw + cpx * 2, cy1 + ch + bl + cpy * 2
    cv2.rectangle(out, (int(cx1), int(cy1)), (int(cx2), int(cy2)), sty["chip_bg"], -1)
    cv2.rectangle(out, (int(cx1), int(cy1)), (int(cx2), int(cy2)), sty["border"], 1)
    _put(out, chip, cx1 + cpx, cy1 + cpy + ch, FS_SM, sty["chip_fg"], TK1)
    if 0 < anomaly_counter < N_ANOMALY_CONFIRM:
        _put(out, f"({anomaly_counter}/{N_ANOMALY_CONFIRM})",
             cx2 + 6, cy1 + cpy + ch, FS_XS, (160, 80, 80), TK1)
    ly = cy2 + 6

    if state == STATE_WARMUP and WARMUP_FRAMES > 0:
        bw   = cx2 - cx1;  bh = max(6, int(7 * S))
        prog = min(warmup_frame / (WARMUP_FRAMES * (warmup_retry + 1)), 1.0)
        cv2.rectangle(out, (int(cx1), int(ly)), (int(cx1 + bw), int(ly + bh)), (60, 60, 60), -1)
        cv2.rectangle(out, (int(cx1), int(ly)), (int(cx1 + int(bw * prog)), int(ly + bh)),
                      (180, 140, 20), -1);  ly += bh + 4
        if warmup_retry > 0:
            _put(out, f"retry {warmup_retry}/{MAX_WARMUP_RETRIES}",
                 cx1, ly + ROW - 6, FS_XS, (160, 120, 40), TK1);  ly += ROW

    if state == STATE_HAND:
        _put(out, "DETECTIONS PAUSED (hand in ROI)",
             lx, ly + ROW - 6, FS_XS, _ROI_COL_AMBER, TK1);  ly += ROW
    else:
        _put(out, f"LIVE INFER   [{infer_frames}f]",
             lx, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        if mask_is_locked:
            _put(out, "MASK: LOCKED", lx + 190, ly + ROW - 6, FS_XS, (80, 255, 160), TK1)
        else:
            _put(out, "MASK: REFRESHED", lx + 190, ly + ROW - 6, FS_XS, (0, 165, 255), TK1)
        ly += ROW

        if seq_stable_ctr > 0:
            sc_col = (60, 220, 60) if seq_stable_ctr >= MIN_SEQ_STABLE else (160, 160, 40)
            _put(out, f"SEQ STABLE  {seq_stable_ctr}/{MIN_SEQ_STABLE}",
                 lx, ly + ROW - 6, FS_XS, sc_col, TK1);  ly += ROW

    if vote_counter is not None and getattr(vote_counter, "total", 0) > 0:
        ly += 2
        _put(out, f"OK : {getattr(vote_counter, 'normal_votes', 0)}",
             lx, ly + ROW - 6, FS_XS, (60, 220, 60), TK1);  ly += ROW
        _put(out, f"AN : {getattr(vote_counter, 'anomaly_votes', 0)}",
             lx, ly + ROW - 6, FS_XS, (80, 80, 230), TK1);  ly += ROW

    rx = RP_X + PAD;  ry = RP_Y + PAD
    _put(out, "INSPECTION STATUS", rx, ry + ROW - 6, FS_XS, (100, 100, 100), TK1)
    ry += ROW + 4
    cv2.line(out, (int(rx), int(ry)), (int(RP_X + RP_W - PAD), int(ry)), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        hc = _ROI_COL_AMBER;  ht = "HAND     IN ROI"
        cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT + 2), hc, -1)
        cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT + 2), (255, 255, 255), 1)
    else:
        hc = (70, 70, 70);  ht = "HAND     CLEAR"
        cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT), hc, -1)
    _put(out, ht, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, hc, TK1);  ry += ROW + 4

    s_col, s_txt = (
        ((0, 210, 100), "SOCKET   PRESENT")
        if socket_hit and socket_hit.get("class") == CLS_SOCKET
        else ((60, 60, 220), "SOCKET   ABSENT")
    )
    cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT), s_col, -1)
    _put(out, s_txt, rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, s_col, TK1);  ry += ROW + 4
    cv2.line(out, (int(rx), int(ry)), (int(RP_X + RP_W - PAD), int(ry)), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        for ci in (2, 3, 4):
            dc  = (50, 50, 50)
            tc3 = (140, 140, 140)
            cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT), dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   PAUSED",
                 rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4
    else:
        for ci in (2, 3, 4):
            pres = status_dict.get(ci, "Absent") == "Present"
            dc   = CLASS_INFO[ci][1] if pres else (50, 50, 50)
            tc3  = (180, 255, 180) if pres else (90, 90, 90)
            cv2.circle(out, (int(rx + DOT), int(ry + ROW // 2 - 3)), int(DOT), dc, -1)
            _put(out, f"{TUBE_LABELS[ci]}   {'PRESENT' if pres else 'ABSENT'}",
                 rx + DOT * 2 + 6, ry + ROW - 6, FS_SM, tc3, TK1);  ry += ROW + 4

    cv2.line(out, (int(rx), int(ry)), (int(RP_X + RP_W - PAD), int(ry)), (45, 45, 45), 1);  ry += 8

    if hand_in_roi:
        seq_str = "-"
    else:
        seq_str = " > ".join(TUBE_SHORT[c] for c in detected_seq) if detected_seq else "-"
    _put(out, f"SEQ  {seq_str}", rx, ry + ROW - 6, FS_SM, (160, 160, 160), TK1);  ry += ROW + 6

    if hand_in_roi:
        _put(out, "INFERENCE   PAUSED", rx, ry + ROW - 4, FS_MD, _ROI_COL_AMBER, TK2)
    else:
        vcol, vtxt = {
            "OK":      ((60, 220, 60),  "ORDER  [OK]  NORMAL"),
            "ANOMALY": ((40, 40, 230),  "ORDER  [!!]  ANOMALY"),
            "PARTIAL": ((30, 160, 200), "ORDER  [??]  PARTIAL"),
        }.get(order_status, ((100, 100, 100), "ORDER  ---  N/A"))
        _put(out, vtxt, rx, ry + ROW - 4, FS_MD, vcol, TK2)
    return out

def draw_final_verdict_overlay(frame, verdict, cycle_no=None, stats=None):
    out     = frame.copy()
    H, W    = out.shape[:2]
    is_anom = verdict == "ANOMALY"
    dim     = np.zeros_like(out)
    dim[:]  = (0, 0, 100) if is_anom else (0, 70, 10)
    out     = cv2.addWeighted(out, 0.40, dim, 0.60, 0)
    tc      = (60, 60, 255) if is_anom else (60, 230, 60)
    label   = f"CYCLE #{cycle_no:03d}  FINAL: {verdict}" if cycle_no else f"FINAL: {verdict}"
    sub     = "Check cable tube order!" if is_anom else "Cable correctly inserted."
    fs_big  = max(1.6, min(W, H) / 240.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs_big, 3)
    tx, ty  = (W - tw) // 2, H // 3
    cv2.putText(out, label, (int(tx + 3), int(ty + 3)),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (int(tx), int(ty)),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, tc, 3, cv2.LINE_AA)
    fs_sub = max(0.8, fs_big * 0.42)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_sub, 2)
    sy = ty + th + max(20, int(H * 0.04))
    cv2.putText(out, sub, (int((W - sw) // 2), int(sy)),
                cv2.FONT_HERSHEY_SIMPLEX, fs_sub, tc, 2, cv2.LINE_AA)
    if stats:
        lines = [
            f"Cycle frames  : {stats.get('total_frames', '-')}",
            f"Warmup frames : {stats.get('warmup_frames', '-')}",
            f"Infer frames  : {stats.get('infer_frames', '-')}",
            f"Normal votes  : {stats.get('normal_votes', '-')}",
            f"Anomaly votes : {stats.get('anomaly_votes', '-')}",
            f"Anomaly ratio : {stats.get('anomaly_ratio', 0):.1%}",
            f"Channels      : {IN_CHANNELS}  radial={USE_RADIAL_CHANNEL}",
        ]
        fs_s = max(0.55, fs_big * 0.32)
        rh   = max(24, int(H * 0.038))
        bw   = max(300, int(W * 0.30))
        bh   = rh * len(lines) + 24
        bx   = (W - bw) // 2
        by   = sy + max(30, int(H * 0.05))
        cv2.rectangle(out, (int(bx - 8), int(by - 8)), (int(bx + bw + 8), int(by + bh + 8)),
                      (30, 30, 30), -1)
        cv2.rectangle(out, (int(bx - 8), int(by - 8)), (int(bx + bw + 8), int(by + bh + 8)), tc, 1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (int(bx), int(by + (i + 1) * rh)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_s, (210, 210, 210),
                        1, cv2.LINE_AA)
    return out

def get_verdict_dir(out_dir: str, verdict: str):
    """
    Returns (full_path, folder_name) for saving output.
    """
    folder = verdict if verdict in ("NORMAL", "ANOMALY", "UNKNOWN") else "UNKNOWN"
    d = os.path.join(out_dir, folder)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d, folder


class CurrentRenderer(BaseRenderer):
    def render(self, frame: np.ndarray, metadata: FrameMetadata) -> np.ndarray:
        vis = frame.copy()
        
        # 1. Render Masks and Boxes (Fixes Issue #1 and #2)
        debug_info = metadata.debug_info
        
        # Mask and Box rendering is handled strictly by VisionOverlayRenderer
        # (Fixes Issue #1 and #2)
        
        vis = draw_production_status_bar(
            vis, 
            metadata.state, 
            metadata.cycle_number, 
            metadata.counters.get('passed', 0), 
            metadata.counters.get('failed', 0), 
            metadata.counters.get('unknown', 0)
        )
        
        vis = draw_hud(
            vis,
            metadata.fps,
            metadata.frame_number,
            metadata.state,
            metadata.socket_hit,
            metadata.status_dict,
            metadata.order_status,
            metadata.detected_seq,
            metadata.counters.get('anomaly_counter', 0),
            metadata.hand_in_roi,
            metadata.counters.get('warmup_frame', 0),
            metadata.counters.get('warmup_retry', 0),
            metadata.counters.get('vote_counter', None),
            metadata.counters.get('infer_frames', 0),
            metadata.counters.get('seq_stable_ctr', 0),
            metadata.cycle_number,
            metadata.elapsed_ms,
            metadata.avg_elapsed_ms,
            metadata.mask_is_locked
        )
        
        if getattr(metadata, "is_final", False):
            vis = draw_final_verdict_overlay(
                vis, metadata.verdict, cycle_no=metadata.cycle_number, stats=metadata.counters.get("stats_card"))

        return vis
