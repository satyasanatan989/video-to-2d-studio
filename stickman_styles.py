import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path
_detector = None

def get_pose_detector(model_path=None):
    global _detector
    if _detector is None:
        if model_path is None:
            model_path = str(Path(__file__).parent / 'pose_landmarker_lite.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        _detector = vision.PoseLandmarker.create_from_options(options)
    return _detector

def render_stickman_styled_frame(frame: np.ndarray, style: str, audio_loudness: float,
                                out_w: int, out_h: int, prev_landmarks: np.ndarray | None,
                                frame_idx: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    detector = get_pose_detector()
    
    # Run pose detection on downscaled frame for performance
    track_w, track_h = 540, 960
    resized = cv2.resize(frame, (track_w, track_h))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)

    current_lms = None
    alpha = 0.35
    if res.pose_landmarks and len(res.pose_landmarks) > 0:
        raw_lms = np.array([[lm.x * out_w, lm.y * out_h] for lm in res.pose_landmarks[0]])
        if prev_landmarks is None:
            current_lms = raw_lms
        else:
            current_lms = alpha * raw_lms + (1.0 - alpha) * prev_landmarks
    elif prev_landmarks is not None:
        current_lms = prev_landmarks

    # Render according to chosen Stickman Video Director style
    if "Style 1A" in style:
        # Style 1A: Classic Minimalist Light (Pure White Canvas, Black Stickman)
        canvas = np.full((out_h, out_w, 3), (255, 255, 255), dtype=np.uint8)
        table_y = int(out_h * 0.70)
        cv2.line(canvas, (int(out_w * 0.08), table_y), (int(out_w * 0.92), table_y), (180, 185, 195), 3, cv2.LINE_AA)

        if current_lms is not None:
            def p(idx): return (int(current_lms[idx, 0]), int(current_lms[idx, 1]))
            sh_l, sh_r = p(11), p(12)
            hip_l, hip_r = p(23), p(24)
            BLACK = (18, 18, 22)
            ACCENT_BLUE = (220, 80, 20)

            neck_pt = ((sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2)
            hip_c = ((hip_l[0] + hip_r[0]) // 2, min(table_y, (hip_l[1] + hip_r[1]) // 2))
            cv2.line(canvas, neck_pt, hip_c, BLACK, 8, cv2.LINE_AA)
            cv2.line(canvas, sh_l, sh_r, BLACK, 8, cv2.LINE_AA)

            def draw_limb(p1, p2):
                cv2.line(canvas, p1, p2, BLACK, 7, cv2.LINE_AA)
                cv2.circle(canvas, p1, 5, ACCENT_BLUE, -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, 5, ACCENT_BLUE, -1, cv2.LINE_AA)

            draw_limb(sh_l, p(13))
            draw_limb(p(13), p(15))
            draw_limb(sh_r, p(14))
            draw_limb(p(14), p(16))

            sh_dist = np.linalg.norm(current_lms[11] - current_lms[12])
            head_r = max(55, int(sh_dist * 0.40))
            head_c = (p(0)[0], p(0)[1] - head_r // 5)
            cv2.line(canvas, (head_c[0], head_c[1] + head_r), neck_pt, BLACK, 6, cv2.LINE_AA)
            cv2.circle(canvas, head_c, head_r, BLACK, 6, cv2.LINE_AA)

            cv2.circle(canvas, (head_c[0] - 16, head_c[1] - 4), 5, BLACK, -1, cv2.LINE_AA)
            cv2.circle(canvas, (head_c[0] + 16, head_c[1] - 4), 5, BLACK, -1, cv2.LINE_AA)
            mouth_open = int(6 + audio_loudness * 14)
            cv2.ellipse(canvas, (head_c[0], head_c[1] + 20), (16, mouth_open), 0, 0, 180, BLACK, 4, cv2.LINE_AA)

        cv2.putText(canvas, "STYLE 1A: LIGHT MINIMALIST", (out_w // 2 - 190, 140), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 25), 2, cv2.LINE_AA)

    elif "Style 1B" in style:
        # Style 1B: Classic Minimalist Dark (Pure Black Canvas, White Stickman)
        canvas = np.full((out_h, out_w, 3), (12, 12, 14), dtype=np.uint8)
        table_y = int(out_h * 0.70)
        cv2.line(canvas, (int(out_w * 0.08), table_y), (int(out_w * 0.92), table_y), (60, 65, 75), 3, cv2.LINE_AA)

        if current_lms is not None:
            def p(idx): return (int(current_lms[idx, 0]), int(current_lms[idx, 1]))
            sh_l, sh_r = p(11), p(12)
            hip_l, hip_r = p(23), p(24)
            WHITE = (250, 250, 252)
            ACCENT_CYAN = (255, 200, 60)

            neck_pt = ((sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2)
            hip_c = ((hip_l[0] + hip_r[0]) // 2, min(table_y, (hip_l[1] + hip_r[1]) // 2))
            cv2.line(canvas, neck_pt, hip_c, WHITE, 8, cv2.LINE_AA)
            cv2.line(canvas, sh_l, sh_r, WHITE, 8, cv2.LINE_AA)

            def draw_limb(p1, p2):
                cv2.line(canvas, p1, p2, WHITE, 7, cv2.LINE_AA)
                cv2.circle(canvas, p1, 5, ACCENT_CYAN, -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, 5, ACCENT_CYAN, -1, cv2.LINE_AA)

            draw_limb(sh_l, p(13))
            draw_limb(p(13), p(15))
            draw_limb(sh_r, p(14))
            draw_limb(p(14), p(16))

            sh_dist = np.linalg.norm(current_lms[11] - current_lms[12])
            head_r = max(55, int(sh_dist * 0.40))
            head_c = (p(0)[0], p(0)[1] - head_r // 5)
            cv2.line(canvas, (head_c[0], head_c[1] + head_r), neck_pt, WHITE, 6, cv2.LINE_AA)
            cv2.circle(canvas, head_c, head_r, WHITE, 6, cv2.LINE_AA)

            cv2.circle(canvas, (head_c[0] - 16, head_c[1] - 4), 5, WHITE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (head_c[0] + 16, head_c[1] - 4), 5, WHITE, -1, cv2.LINE_AA)
            mouth_open = int(6 + audio_loudness * 14)
            cv2.ellipse(canvas, (head_c[0], head_c[1] + 20), (16, mouth_open), 0, 0, 180, WHITE, 4, cv2.LINE_AA)

    elif "Style 2A" in style:
        # Style 2A: Modern Studio Tech (Apple-style White Studio, Perspective Grid, Cyan UI, Red Beanie Zeke)
        canvas = np.full((out_h, out_w, 3), (250, 250, 252), dtype=np.uint8)
        floor_y = int(out_h * 0.62)
        cv2.line(canvas, (0, floor_y), (out_w, floor_y), (220, 225, 230), 2, cv2.LINE_AA)

        vp_x, vp_y = out_w // 2, int(out_h * 0.40)
        for gx in range(-out_w, 2 * out_w, 80):
            cv2.line(canvas, (gx, out_h), (vp_x + (gx - vp_x) // 6, floor_y), (225, 230, 235), 1, cv2.LINE_AA)
        for gy in range(floor_y, out_h, 60):
            cv2.line(canvas, (0, gy), (out_w, gy), (228, 232, 238), 1, cv2.LINE_AA)

        # Holographic Cyan Box Left
        cv2.rectangle(canvas, (60, 200), (320, 480), (255, 248, 235), -1)
        cv2.rectangle(canvas, (60, 200), (320, 480), (255, 210, 80), 2, cv2.LINE_AA)
        cv2.putText(canvas, "TECH RADAR", (80, 240), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 160, 40), 2, cv2.LINE_AA)
        for bi in range(12):
            bh = int(10 + 35 * np.sin(frame_idx * 0.2 + bi * 0.5) * (audio_loudness + 0.3))
            cv2.line(canvas, (90 + bi * 18, 440), (90 + bi * 18, 440 - bh), (255, 180, 40), 3, cv2.LINE_AA)

        # Holographic Circle Right
        cv2.rectangle(canvas, (out_w - 320, 220), (out_w - 60, 460), (255, 248, 235), -1)
        cv2.rectangle(canvas, (out_w - 320, 220), (out_w - 60, 460), (255, 210, 80), 2, cv2.LINE_AA)
        cv2.circle(canvas, (out_w - 190, 340), int(35 + audio_loudness * 18), (255, 190, 60), 3, cv2.LINE_AA)

        if current_lms is not None:
            def p(idx): return (int(current_lms[idx, 0]), int(current_lms[idx, 1]))
            sh_l, sh_r = p(11), p(12)
            hip_l, hip_r = p(23), p(24)
            YELLOW = (30, 215, 255)
            OUTLINE = (15, 15, 18)

            torso_pts = np.array([sh_l, sh_r, (hip_r[0], max(hip_r[1], floor_y + 100)), (hip_l[0], max(hip_l[1], floor_y + 100))], np.int32)
            cv2.fillPoly(canvas, [torso_pts], YELLOW)
            cv2.polylines(canvas, [torso_pts], True, OUTLINE, 6, cv2.LINE_AA)

            def draw_stick(p1, p2, thick=16):
                cv2.line(canvas, p1, p2, OUTLINE, thick, cv2.LINE_AA)
                cv2.circle(canvas, p1, thick//2, OUTLINE, -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, thick//2, OUTLINE, -1, cv2.LINE_AA)

            draw_stick(sh_l, p(13), 20)
            draw_stick(p(13), p(15), 16)
            draw_stick(sh_r, p(14), 20)
            draw_stick(p(14), p(16), 16)

            sh_dist = np.linalg.norm(current_lms[11] - current_lms[12])
            head_r = max(55, int(sh_dist * 0.42))
            head_c = (p(0)[0], p(0)[1] - head_r // 6)

            cv2.circle(canvas, head_c, head_r + 4, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, head_c, head_r, (255, 255, 255), -1, cv2.LINE_AA)

            # Red Beanie
            RED = (40, 45, 235)
            beanie_poly = np.array([
                (head_c[0] - head_r - 4, head_c[1] - 8),
                (head_c[0] - head_r + 5, head_c[1] - head_r - 20),
                (head_c[0], head_c[1] - head_r - 35),
                (head_c[0] + head_r - 5, head_c[1] - head_r - 20),
                (head_c[0] + head_r + 4, head_c[1] - 8),
                (head_c[0] + head_r, head_c[1] - 18),
                (head_c[0], head_c[1] - 22),
                (head_c[0] - head_r, head_c[1] - 18)
            ], np.int32)
            cv2.fillPoly(canvas, [beanie_poly], RED)
            cv2.polylines(canvas, [beanie_poly], True, OUTLINE, 4, cv2.LINE_AA)
            cv2.ellipse(canvas, (head_c[0], head_c[1] - 14), (head_r + 6, 12), 0, 0, 360, RED, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (head_c[0], head_c[1] - 14), (head_r + 6, 12), 0, 0, 360, OUTLINE, 4, cv2.LINE_AA)

            cv2.circle(canvas, (head_c[0] - 18, head_c[1] + 8), 6, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (head_c[0] + 18, head_c[1] + 8), 6, OUTLINE, -1, cv2.LINE_AA)
            mouth_h = int(4 + audio_loudness * 18)
            cv2.ellipse(canvas, (head_c[0], head_c[1] + 32), (18, mouth_h), 0, 0, 360, OUTLINE, -1, cv2.LINE_AA)

    else:
        # Style 2B: 2D Vector Broadcast Studio (Podcast Character Rig)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for y in range(out_h):
            r = y / out_h
            canvas[y, :] = (int(22 + 14 * r), int(18 + 8 * r), int(16 + 6 * r))
        cv2.circle(canvas, (out_w // 2, int(out_h * 0.32)), 450, (48, 38, 32), -1, cv2.LINE_AA)
        cv2.GaussianBlur(canvas, (101, 101), 0, dst=canvas)

        table_y = int(out_h * 0.70)
        cv2.rectangle(canvas, (0, table_y), (out_w, out_h), (18, 22, 28), -1)
        cv2.line(canvas, (0, table_y), (out_w, table_y), (180, 130, 70), 8, cv2.LINE_AA)

        if current_lms is not None:
            def p(idx): return (int(current_lms[idx, 0]), int(current_lms[idx, 1]))
            sh_l, sh_r = p(11), p(12)
            hip_l, hip_r = p(23), p(24)
            SKIN = (182, 214, 248)
            POLO = (26, 26, 30)
            OUTLINE = (10, 10, 12)

            torso = np.array([sh_l, sh_r, (hip_r[0], max(hip_r[1], table_y + 80)), (hip_l[0], max(hip_l[1], table_y + 80))], np.int32)
            cv2.fillPoly(canvas, [torso], POLO)
            cv2.polylines(canvas, [torso], True, OUTLINE, 6, cv2.LINE_AA)

            def draw_vlimb(p1, p2, col, thick):
                cv2.line(canvas, p1, p2, OUTLINE, thick + 10, cv2.LINE_AA)
                cv2.line(canvas, p1, p2, col, thick, cv2.LINE_AA)
                cv2.circle(canvas, p1, (thick+10)//2, OUTLINE, -1, cv2.LINE_AA)
                cv2.circle(canvas, p2, (thick+10)//2, OUTLINE, -1, cv2.LINE_AA)

            draw_vlimb(sh_l, p(13), POLO, 38)
            draw_vlimb(p(13), p(15), SKIN, 30)
            draw_vlimb(sh_r, p(14), POLO, 38)
            draw_vlimb(p(14), p(16), SKIN, 30)

            for wi in [15, 16]:
                cv2.circle(canvas, p(wi), 24, OUTLINE, -1, cv2.LINE_AA)
                cv2.circle(canvas, p(wi), 19, SKIN, -1, cv2.LINE_AA)

            sh_dist = np.linalg.norm(current_lms[11] - current_lms[12])
            head_r = max(60, int(sh_dist * 0.42))
            head_c = (p(0)[0], p(0)[1] - head_r // 6)

            cv2.circle(canvas, (head_c[0], head_c[1] - 12), head_r + 14, (20, 20, 24), -1, cv2.LINE_AA)
            cv2.circle(canvas, head_c, head_r + 5, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, head_c, head_r, SKIN, -1, cv2.LINE_AA)

            for ex in [head_c[0] - head_r // 3, head_c[0] + head_r // 3]:
                cv2.circle(canvas, (ex, head_c[1] - 4), 14, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (ex, head_c[1] - 4), 14, OUTLINE, 3, cv2.LINE_AA)
                cv2.circle(canvas, (ex + 1, head_c[1] - 3), 7, (35, 24, 18), -1, cv2.LINE_AA)

            mouth_y = head_c[1] + head_r // 3
            mouth_h = int(4 + audio_loudness * 22)
            cv2.ellipse(canvas, (head_c[0], mouth_y), (int(head_r * 0.28), mouth_h), 0, 0, 360, (38, 28, 75), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (head_c[0], mouth_y), (int(head_r * 0.28), mouth_h), 0, 0, 360, OUTLINE, 3, cv2.LINE_AA)

        # Broadcast Mic
        mic_head = (int(out_w * 0.54), int(out_h * 0.60))
        cv2.line(canvas, (int(out_w * 0.68), out_h), (int(out_w * 0.64), int(out_h * 0.72)), (45, 48, 55), 18, cv2.LINE_AA)
        cv2.line(canvas, (int(out_w * 0.64), int(out_h * 0.72)), mic_head, (45, 48, 55), 16, cv2.LINE_AA)
        cv2.ellipse(canvas, mic_head, (36, 75), 32, 0, 360, (22, 22, 26), -1, cv2.LINE_AA)

    return canvas, current_lms
