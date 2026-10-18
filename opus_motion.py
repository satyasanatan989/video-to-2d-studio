"""
Opus 5.5 Motion Graphics & Kinetic Visualizer Engine
Inspired by https://github.com/yihui-dev/awesome-opus5-5-videos

Provides:
1. Curated prompt library loader & search across 703 viral Opus 5.5 prompts.
2. Motion graphic overlays (Audio Visualizer Rings, Cyber Grid, Kinetic Typography, Floating Particles, SaaS Glass Card)
   rendered in real-time onto video frames.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import cv2

DATA_PATH = Path(__file__).resolve().parent.parent / "awesome-opus5-5-videos" / "data" / "videos.json"

_CACHED_PROMPTS = None

def load_opus_prompts() -> list[dict]:
    global _CACHED_PROMPTS
    if _CACHED_PROMPTS is not None:
        return _CACHED_PROMPTS
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                _CACHED_PROMPTS = json.load(f)
                return _CACHED_PROMPTS
        except Exception:
            pass
    _CACHED_PROMPTS = []
    return _CACHED_PROMPTS


def search_opus_prompts(query: str = "", category: str = "all", limit: int = 50) -> list[dict]:
    prompts = load_opus_prompts()
    results = []
    q = query.lower().strip()
    for item in prompts:
        if category != "all" and item.get("category") != category:
            continue
        if q:
            match = (
                q in item.get("slug", "").lower()
                or q in item.get("author", "").lower()
                or q in item.get("prompt", "").lower()
                or any(q in t.lower() for t in item.get("tech_tags", []))
            )
            if not match:
                continue
        results.append(item)
        if len(results) >= limit:
            break
    return results


class OpusMotionOverlay:
    """
    Renders dynamic code-driven motion graphic elements onto frames.
    Can be applied on top of 2D characters, AnimeGAN, or stickman rigs.
    """

    @staticmethod
    def draw_audio_reactive_visualizer(canvas: np.ndarray, center: tuple[int, int], radius: int, loudness: float, frame_idx: int, color=(0, 220, 255)):
        """Draws a pulsing circular audio equalizer ring (Canvas/Three.js style)"""
        cx, cy = center
        bars = 48
        angle_step = (2 * math.pi) / bars
        
        # Inner glow ring
        inner_r = max(10, int(radius * (0.85 + 0.15 * loudness)))
        cv2.circle(canvas, (cx, cy), inner_r, (color[0]//2, color[1]//2, color[2]//2), 2, cv2.LINE_AA)

        for i in range(bars):
            angle = i * angle_step + (frame_idx * 0.02)
            # pseudo-frequency band
            freq_amp = math.sin(i * 0.4 + frame_idx * 0.15) * 0.5 + 0.5
            bar_len = int((15 + 60 * loudness * freq_amp))
            
            x1 = int(cx + (inner_r + 4) * math.cos(angle))
            y1 = int(cy + (inner_r + 4) * math.sin(angle))
            x2 = int(cx + (inner_r + 4 + bar_len) * math.cos(angle))
            y2 = int(cy + (inner_r + 4 + bar_len) * math.sin(angle))
            
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    @staticmethod
    def draw_cyber_grid(canvas: np.ndarray, frame_idx: int, color=(40, 45, 60)):
        """Draws subtle synthwave / cyber perspective motion grid floor"""
        h, w = canvas.shape[:2]
        horizon = int(h * 0.65)
        vp_x = w // 2
        
        # Horizontal moving scanning lines
        speed = (frame_idx * 3) % 40
        y = horizon
        step = 10
        while y < h:
            eff_y = y + speed
            if eff_y < h:
                alpha = min(1.0, (eff_y - horizon) / (h - horizon + 1e-5))
                line_color = tuple(int(c * alpha * 0.6) for c in color)
                cv2.line(canvas, (0, eff_y), (w, eff_y), line_color, 1, cv2.LINE_AA)
            y += step
            step = int(step * 1.25) + 2

        # Radial perspective lines
        for gx in range(-w, 2 * w, 120):
            cv2.line(canvas, (gx, h), (vp_x + (gx - vp_x) // 8, horizon), color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_floating_particles(canvas: np.ndarray, frame_idx: int, count: int = 35, color=(255, 230, 150)):
        """Floating ambient bokeh particles that drift upwards"""
        h, w = canvas.shape[:2]
        np.random.seed(42)  # Deterministic seed for particle base coords
        base_coords = np.random.rand(count, 3) # x_norm, y_norm, speed_factor

        for i, (bx, by, sp) in enumerate(base_coords):
            # drift
            y = int(((by - frame_idx * 0.003 * (0.8 + sp * 0.8)) % 1.0) * h)
            x = int(((bx + math.sin(frame_idx * 0.02 + i) * 0.04) % 1.0) * w)
            radius = int(2 + sp * 4)
            particle_alpha = 0.4 + 0.6 * math.sin(frame_idx * 0.05 + i * 0.7)
            cur_color = tuple(int(c * particle_alpha) for c in color)
            cv2.circle(canvas, (x, y), radius, cur_color, -1, cv2.LINE_AA)

    @staticmethod
    def draw_kinetic_quote_banner(canvas: np.ndarray, text: str, frame_idx: int, loudness: float):
        """Draws sleek animated Glassmorphism kinetic title bar optimized for Shorts & Reels"""
        if not text:
            return
        h, w = canvas.shape[:2]
        bar_w = min(int(w * 0.92), 920)
        bar_h = 110
        x1 = (w - bar_w) // 2
        y1 = int(h * 0.08)
        x2 = x1 + bar_w
        y2 = y1 + bar_h

        # Semi-transparent dark pill background
        sub_img = canvas[y1:y2, x1:x2]
        glass_rect = np.full(sub_img.shape, (14, 16, 24), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.20, glass_rect, 0.80, 0)

        # Pulse border
        border_color = (0, 220, 255) if loudness > 0.35 else (80, 95, 120)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 3, cv2.LINE_AA)

        # Pulsing status dot
        dot_r = int(6 + loudness * 6)
        cv2.circle(canvas, (x1 + 35, y1 + bar_h // 2), dot_r, (0, 225, 255), -1, cv2.LINE_AA)

        # Typography: Auto-scaled and centered
        font = cv2.FONT_HERSHEY_DUPLEX
        clean_text = text.upper()
        scale = 0.95
        thick = 2
        text_size, _ = cv2.getTextSize(clean_text, font, scale, thick)
        if text_size[0] > (bar_w - 90):
            scale = max(0.55, scale * ((bar_w - 90) / text_size[0]))
            text_size, _ = cv2.getTextSize(clean_text, font, scale, thick)

        tx = x1 + 65 + ((bar_w - 75) - text_size[0]) // 2
        ty = y1 + (bar_h + text_size[1]) // 2

        # Text shadow for maximum legibility on mobile screens
        cv2.putText(canvas, clean_text, (tx + 2, ty + 2), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, clean_text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
