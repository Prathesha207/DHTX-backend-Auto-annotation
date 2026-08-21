from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np

if TYPE_CHECKING:
    from services.inference_service import OverlayData

_DUPLEX = cv2.FONT_HERSHEY_DUPLEX

_VERDICT_COLORS = {
    'NORMAL':  (0,   220,  0),
    'PASS':    (0,   220,  0),
    'ANOMALY': (0,   0,   220),
    'PARTIAL': (0,   165, 255),
    'UNKNOWN': (110, 110, 110),
}
_VERDICT_LABELS = {
    'NORMAL':  'PASS - NORMAL',
    'ANOMALY': 'FAIL - ANOMALY',
    'PARTIAL': 'PARTIAL',
    'UNKNOWN': 'EVALUATING...',
}
_STATUS_COLORS = {
    'WAITING':  (11,  158, 245),
    'RUNNING':  (235, 144,  52),
    'COMPLETE': (34,  197,  94),
    'FAILED':   (60,   60, 230),
    'IDLE':     (150, 150, 150),
}

def _put_text(img, text, x, y, fs, color, th=1, center=False):
    (tw, _), _ = cv2.getTextSize(text, _DUPLEX, fs, th)
    if center:
        x = x - tw / 2
    cv2.putText(img, text, (int(x), int(y)), _DUPLEX, fs, color, th, cv2.LINE_AA)
    return tw

def _panel(img, x, y, w, h, bg=None, alpha=0.90, border=True):
    col = bg if bg is not None else (25, 31, 41)
    ov  = img.copy()
    cv2.rectangle(ov, (int(x), int(y)), (int(x+w), int(y+h)), col, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    if border:
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (55,65,81), 1)

def _sock_state(o):
    st = o.state.upper().replace('STATE_', '')
    if st == 'IDLE':
        return 'WAITING'
    if o.m1_total > 0 and o.m1_frames >= o.m1_total:
        return 'COMPLETE'
    return 'RUNNING'

def _tube_state(o):
    if o.m1_total > 0 and o.m1_frames < o.m1_total:
        return 'WAITING'
    if o.m2_total > 0 and o.m2_frames >= o.m2_total:
        return 'COMPLETE'
    st = o.state.upper().replace('STATE_', '')
    return 'IDLE' if st == 'IDLE' else 'RUNNING'

def render_overlay(frame: np.ndarray, overlay) -> np.ndarray:
    vis  = frame
    H, W = vis.shape[:2]
    S    = W / 1920.0

    # 1. Top-Left: Cycle + FPS
    TL_W = int(340 * S)
    TL_H = int(48 * S)
    TL_X = int(20 * S)
    TL_Y = int(20 * S)
    _panel(vis, TL_X, TL_Y, TL_W, TL_H)
    dx = TL_X + int(185 * S)
    cv2.line(vis, (dx, TL_Y+int(8*S)), (dx, TL_Y+TL_H-int(8*S)), (55,65,81), 1)
    _put_text(vis, 'Cycle #{:03d}'.format(overlay.cycle_no),
              TL_X+16*S, TL_Y+30*S, 0.56*S, (255,255,255), 1)
    _put_text(vis, '{:.1f} FPS'.format(overlay.fps),
              dx+16*S, TL_Y+30*S, 0.56*S, (52,144,235), 2)

    # 2. Top-Center: Verdict badge
    TC_W = int(280 * S)
    TC_H = int(60 * S)
    TC_X = int((W - TC_W) / 2)
    TC_Y = int(20 * S)
    vd  = (overlay.verdict or 'UNKNOWN').upper()
    vc  = _VERDICT_COLORS.get(vd, (110,110,110))
    vlb = _VERDICT_LABELS.get(vd, vd)
    _panel(vis, TC_X, TC_Y, TC_W, TC_H, bg=(20,24,28), alpha=0.92, border=False)
    cv2.rectangle(vis, (TC_X, TC_Y), (TC_X+TC_W, TC_Y+TC_H), vc, max(2, int(2*S)))
    _put_text(vis, vlb, TC_X+TC_W/2, TC_Y+40*S, 0.72*S, vc,
              max(1, int(2*S)), center=True)

    # 3. Top-Right: Frame + Latency
    TR_W = int(200 * S)
    TR_H = int(70 * S)
    TR_X = W - TR_W - int(20 * S)
    TR_Y = int(20 * S)
    _panel(vis, TR_X, TR_Y, TR_W, TR_H)
    _put_text(vis, 'Frame',  TR_X+16*S, TR_Y+25*S, 0.45*S, (170,170,170), 1)
    _put_text(vis, '{:06d}'.format(overlay.frame_idx),
              TR_X+100*S, TR_Y+25*S, 0.48*S, (255,255,255), 2)
    _put_text(vis, 'Latency', TR_X+16*S, TR_Y+52*S, 0.45*S, (170,170,170), 1)
    _put_text(vis, '{:.1f} ms'.format(overlay.frame_ms),
              TR_X+100*S, TR_Y+52*S, 0.48*S, (255,255,255), 2)

    # 4. Bottom-Left: Inspection Progress
    BL_W = int(360 * S)
    BL_H = int(380 * S)
    BL_X = int(20 * S)
    BL_Y = max(TL_Y + TL_H + int(10 * S), H - BL_H - int(20 * S))
    _panel(vis, BL_X, BL_Y, BL_W, BL_H)
    _put_text(vis, 'Inspection Progress', BL_X+15*S, BL_Y+35*S,
              0.70*S, (255,255,255), max(1, int(S)))
    cv2.line(vis,
             (BL_X+int(15*S), int(BL_Y+50*S)),
             (BL_X+BL_W-int(15*S), int(BL_Y+50*S)),
             (80,80,80), 1)

    def subcard(sx, sy, title, state, cur, req):
        sc = _STATUS_COLORS.get(state.upper(), (255,255,255))
        _put_text(vis, title,           sx,        sy,       0.56*S, (235,235,235), 1)
        _put_text(vis, 'Status',        sx,        sy+34*S,  0.45*S, (160,160,160), 1)
        _put_text(vis, state.upper(),   sx+185*S,  sy+34*S,  0.45*S, sc,           2)
        _put_text(vis, 'Progress',      sx,        sy+64*S,  0.45*S, (160,160,160), 1)
        _put_text(vis, '{} / {}'.format(cur, req), sx+185*S, sy+64*S, 0.45*S, (255,255,255), 2)

    subcard(BL_X+18*S, BL_Y+82*S,  'Socket Detection',
            _sock_state(overlay), overlay.m1_frames, overlay.m1_total)
    cv2.line(vis, (BL_X+int(16*S), int(BL_Y+172*S)),
             (BL_X+BL_W-int(16*S), int(BL_Y+172*S)), (55,65,81), 1)
    subcard(BL_X+18*S, BL_Y+192*S, 'Tube Detection',
            _tube_state(overlay), overlay.m2_frames, overlay.m2_total)
    cv2.line(vis, (BL_X+int(16*S), int(BL_Y+282*S)),
             (BL_X+BL_W-int(16*S), int(BL_Y+282*S)), (55,65,81), 1)
    _put_text(vis, 'Current Step', BL_X+18*S, BL_Y+315*S, 0.45*S, (160,160,160), 1)
    step = overlay.state.replace('STATE_', '').replace('_', ' ')
    _put_text(vis, step, BL_X+18*S, BL_Y+345*S, 0.55*S, (52,144,235), 2)

    # 5. Bottom-Right: Socket / Tube / Hand badges
    BD_W = int(270 * S)
    BD_H = int(42 * S)
    BD_G = int(10 * S)
    BD_X = W - BD_W - int(20 * S)
    BD_Y = H - (BD_H * 3) - (BD_G * 2) - int(24 * S)

    def badge(bx, by, label, status, accent, active):
        if active:
            bg    = tuple(int(c * 0.28) for c in accent)
            bdr   = accent
            dot   = accent
            l_col = (235, 235, 235)
            s_col = accent
        else:
            bg    = (52, 45, 42)
            bdr   = (96, 88, 82)
            dot   = (120, 120, 120)
            l_col = (205, 205, 205)
            s_col = (160, 160, 160)
        ov2 = vis.copy()
        cv2.rectangle(ov2, (int(bx),int(by)), (int(bx+BD_W),int(by+BD_H)), bg, -1)
        cv2.addWeighted(ov2, 0.90, vis, 0.10, 0, vis)
        cv2.rectangle(vis, (int(bx),int(by)), (int(bx+BD_W),int(by+BD_H)), bdr, 1)
        cv2.circle(vis, (int(bx+18*S), int(by+BD_H/2)), max(4,int(4*S)), dot, -1)
        _put_text(vis, label,  bx+34*S, by+BD_H/2+5*S, 0.50*S, l_col, 1)
        tw = cv2.getTextSize(status, _DUPLEX, 0.50*S, 1)[0][0]
        _put_text(vis, status, bx+BD_W-tw-18*S, by+BD_H/2+5*S, 0.50*S, s_col, 1)

    badge(BD_X, BD_Y,
          'Socket', 'PRESENT' if overlay.socket_present else 'ABSENT',
          (94, 197, 34), overlay.socket_present)
    badge(BD_X, BD_Y + BD_H + BD_G,
          'Tube', 'DETECTED' if overlay.tube_detected else 'MISSING',
          (246, 130, 59), overlay.tube_detected)
    badge(BD_X, BD_Y + (BD_H + BD_G) * 2,
          'Hand', 'PRESENT' if overlay.hand_present else 'CLEAR',
          (11, 158, 245), overlay.hand_present)

    return vis
