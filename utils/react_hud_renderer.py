import cv2
import numpy as np

# ─── DESIGN TOKENS  ────────────────────────────────────────────────────────────
# All colors are BGR to match OpenCV convention.

COLORS = {
    "bg_dark":        (20, 14, 10),       # Near-black panel background
    "bg_card":        (30, 20, 14),       # Slightly lighter card
    "border":         (60, 45, 35),       # Subtle border line
    "text_primary":   (248, 250, 252),    # #f8fafc – main text
    "text_secondary": (146, 157, 172),    # #929dac – muted text
    "text_dim":       (98, 105, 118),     # #626976 – very muted
    # Accent / status (BGR)
    "blue":      (240, 155,  45),         # #2D9BF0
    "blue_lt":   ( 40,  28,   8),         # very dark blue tint
    "green":     ( 52, 179,  47),         # #2FB344
    "green_lt":  (  8,  28,   5),
    "red":       ( 57,  57, 214),         # #D63939
    "red_lt":    ( 10,  10,  40),
    "yellow":    ( 40, 159, 245),         # #F59F00
    "yellow_lt": (  5,  25,  40),
    "cyan":      (200, 180,  20),         # cyan-ish
    "cyan_lt":   ( 30,  25,   3),
    "accent":    (196, 107,  32),         # #206BC4
}

FONT   = cv2.FONT_HERSHEY_SIMPLEX
BOLD   = 2


# ─── Scale helper ──────────────────────────────────────────────────────────────

def _scale(frame_h: int) -> float:
    """Return a font-scale factor based on frame height so overlays look good
    at any resolution (480p → 4K). Minimum bumped up so labels are always readable."""
    return max(0.65, min(2.0, frame_h / 720.0))


# ─── Primitive helpers ─────────────────────────────────────────────────────────

def _draw_rect_alpha(img, tl, br, color, alpha: float = 0.82):
    """Filled rectangle with alpha blending — ROI-based, no full-frame copy."""
    ih, iw = img.shape[:2]
    x1, y1 = max(0, tl[0]), max(0, tl[1])
    x2, y2 = min(iw, br[0]), min(ih, br[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _round_rect(img, tl, br, color, radius: int = 6, alpha: float = 1.0):
    """Filled rounded rectangle — ROI-based blending, no full-frame copy."""
    ih, iw = img.shape[:2]
    x1, y1 = max(0, tl[0]), max(0, tl[1])
    x2, y2 = min(iw, br[0]), min(ih, br[1])
    if x2 <= x1 or y2 <= y1:
        return img
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if alpha >= 1.0:
        # Draw directly on image — zero copies needed
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        # ROI-based alpha blend — only copy the small panel region, not the whole frame
        roi = img[y1:y2, x1:x2]
        overlay = roi.copy()  # small: e.g., 300x100px instead of 1920x1080
        rw, rh = x2 - x1, y2 - y1
        cv2.rectangle(overlay, (r, 0), (rw - r, rh), color, -1)
        cv2.rectangle(overlay, (0, r), (rw, rh - r), color, -1)
        for cx, cy in [(r, r), (rw - r, r), (r, rh - r), (rw - r, rh - r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    return img


def _text_with_outline(img, text, org, font, scale, color, thick: int = 1):
    """Draw text with a dark outline for maximum legibility on any background."""
    outline_col = (0, 0, 0)
    cv2.putText(img, text, org, font, scale, outline_col, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def _round_rect_border(img, tl, br, color, radius: int = 6, thick: int = 1):
    """Draw a rounded rectangle BORDER (no fill) via 4 arcs + 4 lines."""
    x1, y1 = tl; x2, y2 = br
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # Straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, cv2.LINE_AA)
    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thick, cv2.LINE_AA)


def _pill(img, x: int, y: int, text: str, fg, bg, fs: float,
          dot_color=None) -> int:
    """Draw a Tabler-style pill badge; returns the x position after the pill."""
    (tw, th), bl = cv2.getTextSize(text, FONT, fs, 1)
    px, py = int(10 * (fs / 0.55)), int(5 * (fs / 0.55))
    w = tw + px * 2 + (int(12 * fs) + 6 if dot_color else 0)
    h = th + py * 2

    _round_rect(img, (x, y), (x + w, y + h), bg, radius=h // 2, alpha=1.0)
    # Rounded border (not square rectangle)
    _round_rect_border(img, (x, y), (x + w, y + h), fg, radius=h // 2, thick=1)

    tx = x + px
    if dot_color:
        dot_r = int(4 * (fs / 0.55))
        cx = x + px + dot_r
        cy = y + h // 2
        cv2.circle(img, (cx, cy), dot_r, dot_color, -1)
        tx = cx + dot_r + 4

    _text_with_outline(img, text, (tx, y + h - py - bl + 1), FONT, fs, fg, 1)
    return x + w + int(8 * (fs / 0.55))


# ─── Main renderer ─────────────────────────────────────────────────────────────

def render_react_style_hud(frame, cycle_no, fps, frame_idx, state_str,
                           sock_hit, status_dict, order_status,
                           detected_seq, elapsed_s=0.0, model_ms: dict = None,
                           hand_in_roi: bool = False, model_progress: dict = None,
                           model_status: dict = None):
    """
    Renders an overlay that closely mirrors the InferenceCanvas React UI onto the
    OpenCV frame – used for the saved MP4 output.

    Layout mirrors InferenceCanvas.tsx:
      Top-left   : Cycle # • FPS pill  (blue badge)
      Top-right  : LIVE / STARTING + dark stats card (FRAME / MS)
      Top-center : ANOMALY DETECTED / NORMAL banner (when anomaly/normal)
      Bottom-left: Model-status panel  (like LiveModelsOverlay)
      Bottom-right: Status chips  (Socket / Hand / Tubes)
    """
    H, W = frame.shape[:2]
    S     = _scale(H)
    pad   = int(14 * S)
    thick = max(1, int(S))

    # Font sizes: medium so labels are readable but don't dominate
    fs_sm  = max(0.45, 0.55 * S)   # small labels / chips
    fs_md  = max(0.55, 0.65 * S)   # card values / model names
    fs_lg  = max(0.75, 0.90 * S)   # NORMAL / ANOMALY banner

    # ── State normalisation ──────────────────────────────────────────────────
    STATE_NAMES = {
        "WAIT_FOR_SOCKET":   "WAIT",
        "MODEL1_VALIDATION": "VALID",
        "MODEL2_SKIP":       "SKIP",
        "MODEL2_VALIDATION": "INSPECT",
        "WAIT_SOCKET_REMOVAL": "DONE",
        "CYCLE_FINISHED":    "DONE",
        "HAND":              "HAND",
    }
    state_name = STATE_NAMES.get(state_str, str(state_str))

    # ── Top-left: Cycle # • FPS badge ───────────────────────────────────────
    fps_txt = f"Cycle #{cycle_no}  \u2022  {fps:.1f} fps"
    (tw, th), bl = cv2.getTextSize(fps_txt, FONT, fs_sm, 1)
    pill_pad = int(12 * S)
    pill_h   = th + pill_pad
    pill_w   = tw + pill_pad * 2
    _round_rect(frame, (pad, pad), (pad + pill_w, pad + pill_h),
                COLORS["blue_lt"], radius=pill_h // 2, alpha=1.0)
    _round_rect_border(frame, (pad, pad), (pad + pill_w, pad + pill_h),
                       COLORS["blue"], radius=pill_h // 2, thick=1)
    _text_with_outline(frame, fps_txt,
                       (pad + pill_pad, pad + pill_h - pill_pad // 2 - bl + 1),
                       FONT, fs_sm, COLORS["blue"], 1)

    # ── Top-right: Live badge + stats card ──────────────────────────────────
    # Compute avg ms from EMA fps
    avg_ms = round(1000.0 / max(fps, 0.01), 1)
    frame_txt = f"{int(frame_idx):06d}"
    ms_txt    = f"{avg_ms:.1f}  (avg {avg_ms:.1f})"

    # Determine card width from the widest text
    fw_frame = cv2.getTextSize(frame_txt, FONT, fs_md, BOLD)[0][0]
    fw_ms    = cv2.getTextSize(ms_txt,    FONT, fs_md, BOLD)[0][0]
    lw_frame = cv2.getTextSize("FRAME", FONT, fs_sm, 1)[0][0]
    lw_ms    = cv2.getTextSize("MS",    FONT, fs_sm, 1)[0][0]
    card_inner = max(fw_frame + lw_frame, fw_ms + lw_ms) + int(24 * S)
    card_w  = card_inner + int(20 * S)
    card_h  = int(90 * S)
    cx0     = W - pad - card_w
    cy0     = pad

    _round_rect(frame, (cx0, cy0), (cx0 + card_w, cy0 + card_h),
                COLORS["bg_dark"], radius=int(6 * S), alpha=0.88)
    _round_rect_border(frame, (cx0, cy0), (cx0 + card_w, cy0 + card_h),
                       COLORS["border"], radius=int(6 * S), thick=1)

    # FRAME row
    row1_y = cy0 + int(30 * S)
    _text_with_outline(frame, "FRAME", (cx0 + int(10 * S), row1_y),
                       FONT, fs_sm, COLORS["text_secondary"], 1)
    _text_with_outline(frame, frame_txt,
                       (cx0 + card_w - int(10 * S) - fw_frame, row1_y),
                       FONT, fs_md, COLORS["text_primary"], BOLD)

    # MS row
    row2_y = cy0 + int(62 * S)
    _text_with_outline(frame, "MS", (cx0 + int(10 * S), row2_y),
                       FONT, fs_sm, COLORS["text_secondary"], 1)
    _text_with_outline(frame, ms_txt,
                       (cx0 + card_w - int(10 * S) - fw_ms, row2_y),
                       FONT, fs_md, COLORS["text_primary"], BOLD)

    # LIVE badge (above card, right-aligned)
    live_txt = "LIVE"
    (lw, lh), _ = cv2.getTextSize(live_txt, FONT, fs_sm, 1)
    live_pad = int(8 * S)
    live_w   = lw + live_pad * 2 + int(16 * S)
    live_h   = lh + live_pad
    lx0      = cx0 + card_w - live_w

    # ── Top-center: Anomaly / Normal banner ──────────────────────────────────
    ban_txt = ban_fg = ban_bg = None
    if order_status == "ANOMALY":
        ban_txt = "ANOMALY DETECTED"
        ban_fg  = COLORS["red"]
        ban_bg  = COLORS["red_lt"]
    elif order_status == "NORMAL":
        ban_txt = "NORMAL"
        ban_fg  = COLORS["green"]
        ban_bg  = COLORS["green_lt"]

    if ban_txt:
        (btw, bth), _ = cv2.getTextSize(ban_txt, FONT, fs_lg, BOLD)
        bpad = int(20 * S)
        bw = btw + bpad * 2
        bh = bth + int(12 * S)
        bx = (W - bw) // 2
        by = int(H * 0.06)
        _round_rect(frame, (bx, by), (bx + bw, by + bh),
                    ban_bg, radius=int(4 * S), alpha=0.92)
        _round_rect_border(frame, (bx, by), (bx + bw, by + bh),
                           ban_fg, radius=int(4 * S), thick=2)
        _text_with_outline(frame, ban_txt,
                           (bx + bpad, by + bh - int(6 * S)),
                           FONT, fs_lg, ban_fg, BOLD)

    # ── Hand-paused center banner ─────────────────────────────────────────────
    if state_name == "HAND":
        warn_txt = "INFERENCE PAUSED (HAND)"
        (wtw, wth), _ = cv2.getTextSize(warn_txt, FONT, fs_lg, BOLD)
        wpad = int(20 * S)
        ww = wtw + wpad * 2;  wh = wth + int(16 * S)
        wx = (W - ww) // 2;   wy = H // 2 - wh // 2
        _round_rect(frame, (wx, wy), (wx + ww, wy + wh),
                    COLORS["yellow_lt"], radius=int(8 * S), alpha=0.90)
        _round_rect_border(frame, (wx, wy), (wx + ww, wy + wh),
                           COLORS["yellow"], radius=int(8 * S), thick=2)
        _text_with_outline(frame, warn_txt,
                           (wx + wpad, wy + wh - int(8 * S)),
                           FONT, fs_lg, COLORS["yellow"], BOLD)

    # ── Bottom-left: Model panel (mirrors LiveModelsOverlay) ─────────────────
    model_row_h = int(56 * S)
    model_names = ["Socket Detector", "Hand Detector", "Tube Detector"]
    panel_w = int(300 * S)
    panel_h = len(model_names) * model_row_h + int(24 * S)
    px0 = pad
    py0 = H - pad - panel_h
    _round_rect(frame, (px0, py0), (px0 + panel_w, py0 + panel_h),
                COLORS["bg_dark"], radius=int(8 * S), alpha=0.82)
    _round_rect_border(frame, (px0, py0), (px0 + panel_w, py0 + panel_h),
                       COLORS["border"], radius=int(8 * S), thick=1)

    # Model status derived strictly from injected state machine metadata
    model_status = model_status or {}
    tubes_present = any(v == "Present" for v in (status_dict or {}).values())
    hand_detected = hand_in_roi if hand_in_roi else (state_name == "HAND")
    is_socket_present = sock_hit is not None and isinstance(sock_hit, dict) and sock_hit.get("class") == 0
    model_states = {
        "Socket Detector": model_status.get("Socket Detector", {}).get("running", is_socket_present),
        "Hand Detector":   model_status.get("Hand Detector", {}).get("running", hand_detected),
        "Tube Detector":   model_status.get("Tube Detector", {}).get("running", tubes_present),
    }

    # Per-model execution time in ms
    _ms = model_ms or {}
    # Progress counts from state machine (same values as frontend X/30, X/20)
    _prog = model_progress or {}

    # Left col (timing in ms), right col (frame count like "4/30")
    model_left = {
        "Socket Detector": f"{int(_ms.get('model1', 0))}ms" if _ms.get('model1') else "-",
        "Hand Detector":   f"{int(_ms.get('hand',   0))}ms" if _ms.get('hand')   else "-",
        "Tube Detector":   f"{int(_ms.get('model2', 0))}ms" if _ms.get('model2') else "-",
    }
    model_right = {
        "Socket Detector": (f"{_prog.get('m1_current', 0)}/{_prog.get('m1_target', 30)}"
                            if is_socket_present else "-"),
        "Hand Detector":   "Present" if hand_detected else "Absent",
        "Tube Detector":   (f"{_prog.get('m2_current', 0)}/{_prog.get('m2_target', 20)}"
                            if tubes_present else "-"),
    }
    # Progress bar fill ratio (0.0–1.0) for each model
    model_progress_pct = {
        "Socket Detector": (_prog.get('m1_current', 0) / max(_prog.get('m1_target', 30), 1)
                            if is_socket_present else 0.0),
        "Hand Detector":   1.0 if hand_detected else 0.0,
        "Tube Detector":   (_prog.get('m2_current', 0) / max(_prog.get('m2_target', 20), 1)
                            if tubes_present else 0.0),
    }

    for i, mname in enumerate(model_names):
        ry0 = py0 + int(12 * S) + i * model_row_h
        is_on       = bool(model_states.get(mname))
        dot_c       = COLORS["green"] if is_on else COLORS["text_dim"]
        label       = mname.replace(" Detector", "")
        state_label = "RUNNING" if is_on else "IDLE"
        state_col   = COLORS["green"] if is_on else COLORS["text_dim"]

        # Status dot + glow ring
        dot_cx = px0 + int(14 * S)
        dot_cy = ry0 + int(model_row_h * 0.38)
        cv2.circle(frame, (dot_cx, dot_cy), int(6 * S), dot_c, -1)
        if is_on:
            cv2.circle(frame, (dot_cx, dot_cy), int(10 * S), dot_c, 1, cv2.LINE_AA)

        # Model name (left)
        _text_with_outline(frame, label,
                           (dot_cx + int(16 * S), ry0 + int(model_row_h * 0.48)),
                           FONT, fs_md, COLORS["text_primary"], thick)
        # RUNNING / IDLE status (right-aligned)
        slw = cv2.getTextSize(state_label, FONT, fs_sm, 1)[0][0]
        _text_with_outline(frame, state_label,
                           (px0 + panel_w - int(12 * S) - slw,
                            ry0 + int(model_row_h * 0.48)),
                           FONT, fs_sm, state_col, 1)

        # Bottom row: timing left, frame count right (exactly like frontend)
        left_txt  = model_left.get(mname, "-")
        right_txt = model_right.get(mname, "-")
        _text_with_outline(frame, left_txt,
                           (dot_cx + int(16 * S), ry0 + int(model_row_h * 0.82)),
                           FONT, fs_sm, COLORS["text_secondary"], 1)
        rtw = cv2.getTextSize(right_txt, FONT, fs_sm, 1)[0][0]
        _text_with_outline(frame, right_txt,
                           (px0 + panel_w - int(12 * S) - rtw,
                            ry0 + int(model_row_h * 0.82)),
                           FONT, fs_sm, COLORS["text_secondary"], 1)

        # Progress bar — fills proportionally to actual frame count progress
        bar_x0 = px0 + int(12 * S)
        bar_y0 = ry0 + model_row_h - int(7 * S)
        bar_w  = panel_w - int(24 * S)
        bar_h  = max(3, int(4 * S))
        # Background track
        cv2.rectangle(frame, (bar_x0, bar_y0),
                      (bar_x0 + bar_w, bar_y0 + bar_h),
                      COLORS["border"], -1)
        # Fill
        fill_pct = model_progress_pct.get(mname, 0.0)
        if is_on:
            fill_w = max(bar_h, int(bar_w * min(max(fill_pct, 0.0), 1.0))) if fill_pct > 0 else bar_w
            cv2.rectangle(frame, (bar_x0, bar_y0),
                          (bar_x0 + fill_w, bar_y0 + bar_h),
                          COLORS["accent"], -1)

    # ── Bottom-right: Decision chips ─────────────────────────────────────────
    chips = []
    if is_socket_present:
        chips.append(("Socket present", COLORS["green"],    COLORS["green_lt"],  COLORS["green"]))
    else:
        chips.append(("No socket",      COLORS["text_secondary"], COLORS["bg_dark"], None))
    if state_name == "HAND" or hand_detected:
        chips.append(("Hand present",   COLORS["yellow"],  COLORS["yellow_lt"], COLORS["yellow"]))
    else:
        chips.append(("Hand clear",     COLORS["green"],   COLORS["green_lt"],  COLORS["green"]))
    if tubes_present:
        chips.append(("Tubes detected", COLORS["cyan"],    COLORS["cyan_lt"],   COLORS["cyan"]))
    else:
        chips.append(("Tubes absent",   COLORS["text_secondary"], COLORS["bg_dark"], None))

    # Draw chips bottom-up
    cy_pos = H - pad
    for (ctxt, cfg, cbg, cdot) in reversed(chips):
        (tw, th), bl = cv2.getTextSize(ctxt, FONT, fs_md, 1)
        cpx = int(14 * (fs_md / 0.70))
        dot_r = int(6 * S)
        cw = tw + cpx * 2 + (dot_r * 2 + 8 if cdot else 0)
        ch = th + int(8 * (fs_md / 0.70)) * 2
        cx0 = W - pad - cw
        cy_pos -= ch
        _round_rect(frame, (cx0, cy_pos), (cx0 + cw, cy_pos + ch),
                    cbg, radius=ch // 2, alpha=1.0)
        _round_rect_border(frame, (cx0, cy_pos), (cx0 + cw, cy_pos + ch),
                           cfg, radius=ch // 2, thick=2)
        tx = cx0 + cpx
        if cdot:
            dcx = cx0 + cpx + dot_r
            dcy = cy_pos + ch // 2
            cv2.circle(frame, (dcx, dcy), dot_r, cdot, -1)
            tx = dcx + dot_r + 6
        _text_with_outline(frame, ctxt,
                           (tx, cy_pos + ch - int(8 * (fs_md / 0.70)) - bl + 1),
                           FONT, fs_md, cfg, thick)
        cy_pos -= int(8 * S)

    return frame
