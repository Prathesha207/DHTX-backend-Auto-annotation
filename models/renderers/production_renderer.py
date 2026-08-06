import cv2
import numpy as np
import os
from pathlib import Path
from models.renderers.base_renderer import BaseRenderer, FrameMetadata


class ProductionRenderer(BaseRenderer):
    def render(self, frame: np.ndarray, metadata: FrameMetadata) -> np.ndarray:
        vis = frame.copy()
        
        # (Masks and boxes are now drawn by VisionOverlayRenderer)
        # 2. Setup Responsive Constants
        H, W = vis.shape[:2]
        S = W / 1920.0
        
        def _put_text(img, text, x, y, font_scale, color, thickness=1, center=False):
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            if center:
                x = x - tw // 2
            cv2.putText(img, text, (int(x), int(y)), font, font_scale, color, thickness, cv2.LINE_AA)
            return tw, th
            
        def _draw_panel(img, x, y, w, h, bg_color=None, alpha=0.90, border=True):
            bg = bg_color if bg_color is not None else (41, 31, 25)          # #1F2937
            border_col = (81, 65, 55)      # #374151

            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                bg,
                -1,
            )

            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

            if border:
                cv2.rectangle(
                    img,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    border_col,
                    1,
                )

        # ----------------------------------------------------
        # TOP LEFT : Cycle + FPS
        # ----------------------------------------------------

        TL_W = int(340 * S)
        TL_H = int(48 * S)

        TL_X = int(20 * S)
        TL_Y = int(20 * S)

        _draw_panel(
            vis,
            TL_X,
            TL_Y,
            TL_W,
            TL_H,
        )

        divider_x = TL_X + int(185 * S)

        cv2.line(
            vis,
            (divider_x, TL_Y + 8),
            (divider_x, TL_Y + TL_H - 8),
            (81, 65, 55),      # Divider
            1,
        )

        _put_text(
            vis,
            f"Cycle #{metadata.cycle_number:03d}",
            TL_X + 16*S,
            TL_Y + 30*S,
            0.56*S,
            (255,255,255),
            1,
        )

        _put_text(
            vis,
            f"{metadata.fps:.1f} FPS",
            divider_x + 16*S,
            TL_Y + 30*S,
            0.56*S,
            (235, 144, 52),      # Azure Blue (BGR)
            2,
        )

        # 4. Top Center: Verdict Badge
        TC_W, TC_H = int(260 * S), int(60 * S)
        TC_X, TC_Y = int((W - TC_W) / 2), int(20 * S)
        
        verdict = metadata.verdict
        
        if verdict:
            v_colors = {
                "NORMAL": (0, 220, 0),
                "PASS": (0, 220, 0),
                "ANOMALY": (0, 0, 220),
                "PARTIAL": (0, 165, 255),
                "UNKNOWN": (100, 100, 100)
            }
            v_col = v_colors.get(verdict, (100, 100, 100))
            
            _draw_panel(vis, TC_X, TC_Y, TC_W, TC_H, bg_color=(20, 24, 28), alpha=0.9, border=False)
            cv2.rectangle(vis, (TC_X, TC_Y), (TC_X+TC_W, TC_Y+TC_H), v_col, max(2, int(2*S)))
            
            _put_text(vis, verdict, TC_X + TC_W/2, TC_Y + 40*S, 0.8 * S, v_col, max(1, int(2*S)), center=True)

        # ----------------------------------------------------
        # TOP RIGHT : Performance
        # ----------------------------------------------------

        TR_W = int(200 * S)
        TR_H = int(70 * S)

        TR_X = W - TR_W - int(20*S)
        TR_Y = int(20*S)

        _draw_panel(
            vis,
            TR_X,
            TR_Y,
            TR_W,
            TR_H,
        )

        label_color = (170,170,170)
        value_color = (255,255,255)

        # Frame

        _put_text(
            vis,
            "Frame",
            TR_X + 16*S,
            TR_Y + 25*S,
            0.45*S,
            label_color,
            1,
        )

        _put_text(
            vis,
            f"{metadata.frame_number:06d}",
            TR_X + 92*S,
            TR_Y + 25*S,
            0.48*S,
            value_color,
            2,
        )

        # Latency

        _put_text(
            vis,
            "Latency",
            TR_X + 16*S,
            TR_Y + 52*S,
            0.45*S,
            label_color,
            1,
        )

        _put_text(
            vis,
            f"{metadata.elapsed_ms:.1f} ms",
            TR_X + 92*S,
            TR_Y + 52*S,
            0.48*S,
            value_color,
            2,
        )

        # 6. Bottom Left: Inspection Progress
        BL_W, BL_H = int(360 * S), int(380 * S)
        BL_X, BL_Y = int(20 * S), H - BL_H - int(20 * S)
        _draw_panel(vis, BL_X, BL_Y, BL_W, BL_H)
        
        _put_text(vis, "Inspection Progress", BL_X + 15*S, BL_Y + 35*S, 0.7 * S, (255, 255, 255), max(1, int(1*S)))
        cv2.line(vis, (BL_X + 15, int(BL_Y + 50*S)), (BL_X + BL_W - 15, int(BL_Y + 50*S)), (100, 100, 100), max(1, int(1*S)))
        
        sock_model = metadata.ui_models.get("socket", {})
        tube_model = metadata.ui_models.get("tube", {})
        
        def _draw_subcard(img, sx, sy, title, state, current, required):
            """
            Enterprise inspection section
            """

            # ---------- Colors ----------
            section_color = (235, 235, 235)     # Section title
            label_color   = (160, 160, 160)     # Labels
            value_color   = (255, 255, 255)     # Progress value

            state = state.upper()
            progress = f"{current} / {required}"

            status_colors = {
                "WAITING":  (11, 158, 245),     # Orange
                "RUNNING":  (235, 144, 52),     # Azure Blue
                "COMPLETE": (34, 197, 94),      # Green
                "FAILED":   (60, 60, 230),      # Red
                "IDLE":     (150, 150, 150),    # Gray
            }

            status_color = status_colors.get(state, value_color)

            # ---------------------------------------------------
            # Section Title
            # ---------------------------------------------------

            _put_text(
                img,
                title,
                sx,
                sy,
                0.56 * S,
                section_color,
                1,
            )

            # ---------------------------------------------------
            # Status
            # ---------------------------------------------------

            y1 = sy + 34 * S

            _put_text(
                img,
                "Status",
                sx,
                y1,
                0.45 * S,
                label_color,
                1,
            )

            _put_text(
                img,
                state,
                sx + 185 * S,
                y1,
                0.45 * S,
                status_color,
                2,
            )

            # ---------------------------------------------------
            # Progress
            # ---------------------------------------------------

            y2 = sy + 64 * S

            _put_text(
                img,
                "Progress",
                sx,
                y2,
                0.45 * S,
                label_color,
                1,
            )

            _put_text(
                img,
                progress,
                sx + 185 * S,
                y2,
                0.45 * S,
                value_color,
                2,
            )

        prog = metadata.inspection_progress
        s_cur = prog.socket_current if prog else 0
        s_req = prog.socket_required if prog else 0
        t_cur = prog.tube_current if prog else 0
        t_req = prog.tube_required if prog else 0

        _draw_subcard(
            vis,
            BL_X + 18*S,
            BL_Y + 82*S,
            "Socket Detection",
            sock_model.get("state", "IDLE"),
            s_cur,
            s_req,
        )

        cv2.line(
            vis,
            (BL_X + 16, int(BL_Y + 172*S)),
            (BL_X + BL_W - 16, int(BL_Y + 172*S)),
            (81, 65, 55),
            1,
        )

        _draw_subcard(
            vis,
            BL_X + 18*S,
            BL_Y + 192*S,
            "Tube Detection",
            tube_model.get("state", "IDLE"),
            t_cur,
            t_req,
        )

        cv2.line(
            vis,
            (BL_X + 16, int(BL_Y + 282*S)),
            (BL_X + BL_W - 16, int(BL_Y + 282*S)),
            (81, 65, 55),
            1,
        )

        _put_text(
            vis,
            "Current Step",
            BL_X + 18*S,
            BL_Y + 315*S,
            0.45 * S,
            (160, 160, 160),
            1,
        )

        current_state = metadata.state.replace("STATE_", "").replace("_", " ")
        _put_text(
            vis,
            current_state,
            BL_X + 18*S,
            BL_Y + 345*S,
            0.55 * S,
            (235, 144, 52),
            2,
        )

        # 7. Bottom Right: Status Pills
        BR_W, BR_H = int(240 * S), int(160 * S)
        BR_X, BR_Y = W - BR_W - int(20 * S), H - BR_H - int(20 * S)
        
        pill_h = int(45 * S)
        pill_spacing = int(12 * S)
        
        def _draw_badge(
            img,
            x,
            y,
            w,
            h,
            label,
            status,
            accent,
            active=True,
        ):
            """
            Enterprise rectangular status badge
            """

            radius = max(4, int(4 * S))

            if active:
                bg = tuple(int(c * 0.28) for c in accent)
                border = accent
                dot = accent
                label_col = (235, 235, 235)
                status_col = accent
            else:
                bg = (42, 45, 52)
                border = (82, 88, 96)
                dot = (120, 120, 120)
                label_col = (205, 205, 205)
                status_col = (160, 160, 160)

            overlay = img.copy()

            # Background
            cv2.rectangle(
                overlay,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                bg,
                -1,
            )

            cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)

            # Border
            cv2.rectangle(
                img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                border,
                1,
            )

            # Status dot
            cv2.circle(
                img,
                (int(x + 18 * S), int(y + h / 2)),
                max(4, int(4 * S)),
                dot,
                -1,
            )

            # Left label
            _put_text(
                img,
                label,
                x + 34 * S,
                y + h / 2 + 3 * S,
                0.50 * S,
                label_col,
                1,
            )

            # Right status
            (tw, _), _ = cv2.getTextSize(
                status,
                cv2.FONT_HERSHEY_DUPLEX,
                0.50 * S,
                1,
            )

            _put_text(
                img,
                status,
                x + w - tw - 18 * S,
                y + h / 2 + 3 * S,
                0.50 * S,
                status_col,
                1,
            )

        BR_W = int(270 * S)
        BR_H = int(42 * S)
        gap = int(10 * S)

        BR_X = W - BR_W - int(20 * S)
        # 3 badges + 2 gaps. Target bottom margin: 24 * S
        BR_Y = H - (BR_H * 3) - (gap * 2) - int(24 * S)

        # Socket
        socket_present = metadata.socket_hit is not None

        _draw_badge(
            vis,
            BR_X,
            BR_Y,
            BR_W,
            BR_H,
            "Socket",
            "PRESENT" if socket_present else "ABSENT",
            (34, 197, 94),      # Emerald
            socket_present,
        )

        # Tube
        tube_present = any(metadata.status_dict.values()) if metadata.status_dict else False

        _draw_badge(
            vis,
            BR_X,
            BR_Y + BR_H + gap,
            BR_W,
            BR_H,
            "Tube",
            "DETECTED" if tube_present else "MISSING",
            (246, 130, 59),     # Azure Blue (BGR format)
            tube_present,
        )

        # Hand
        hand_present = metadata.hand_in_roi

        _draw_badge(
            vis,
            BR_X,
            BR_Y + (BR_H + gap) * 2,
            BR_W,
            BR_H,
            "Hand",
            "PRESENT" if hand_present else "CLEAR",
            (11, 158, 245),     # Amber (BGR format)
            hand_present,
        )

        if getattr(metadata, "is_final", False):
            is_anom = metadata.verdict == "ANOMALY"
            dim = np.zeros_like(vis)
            dim[:] = (0, 0, 100) if is_anom else (0, 70, 10)
            vis = cv2.addWeighted(vis, 0.40, dim, 0.60, 0)
            
            tc = (60, 60, 255) if is_anom else (60, 230, 60)
            label = f"CYCLE #{metadata.cycle_number:03d}  FINAL: {metadata.verdict}"
            fs_big = max(1.6, min(W, H) / 240.0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs_big, 3)
            tx, ty = (W - tw) // 2, H // 3
            
            cv2.putText(vis, label, (int(tx) + 3, int(ty) + 3), cv2.FONT_HERSHEY_DUPLEX, fs_big, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, label, (int(tx), int(ty)), cv2.FONT_HERSHEY_DUPLEX, fs_big, tc, 3, cv2.LINE_AA)

        return vis
