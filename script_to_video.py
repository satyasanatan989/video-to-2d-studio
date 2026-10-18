"""
MoneyPrinter-Style Automated Script-to-Video Engine for Shorts & Reels
Combines:
1. Free Edge-TTS narration with natural neural voices.
2. Dynamic multi-scene script breakdown with word-by-word timed subtitles.
3. 2D Vector animated speaker rendering with real-time audio lip-sync.
4. Opus 5.5 reactive visualizer, ambient particles, and kinetic captions.
"""
from __future__ import annotations
import asyncio
import os
import math
import subprocess
import wave
import numpy as np
import cv2
import imageio_ffmpeg
import edge_tts

VOICES = {
    "Christopher (Deep Male Narrator - US)": "en-US-ChristopherNeural",
    "Guy (Energetic Male - US)": "en-US-GuyNeural",
    "Jenny (Professional Female - US)": "en-US-JennyNeural",
    "Aria (Expressive Female - US)": "en-US-AriaNeural",
    "Ryan (Casual Male - UK)": "en-GB-RyanNeural",
    "Sonia (Storyteller Female - UK)": "en-GB-SoniaNeural",
    "Prabhat (Male - India)": "en-IN-PrabhatNeural",
    "Neerja (Female - India)": "en-IN-NeerjaNeural"
}

def ffmpeg_exe():
    return imageio_ffmpeg.get_ffmpeg_exe()


def generate_speech_audio(script_text: str, voice_id: str, output_wav: str) -> float:
    """Generates speech via Microsoft Edge neural TTS and converts to 16kHz mono WAV"""
    temp_mp3 = output_wav + ".temp.mp3"
    
    # Run edge-tts CLI directly
    cmd_tts = ["edge-tts", "--voice", voice_id, "--text", script_text, "--write-media", temp_mp3]
    subprocess.run(cmd_tts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Convert to standard 16kHz WAV for audio-reactive envelope extraction
    cmd_wav = [ffmpeg_exe(), "-y", "-i", temp_mp3, "-vn", "-ac", "1", "-ar", "16000", output_wav]
    subprocess.run(cmd_wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)

    with wave.open(output_wav, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration


def extract_audio_envelope(wav_path: str, fps: int = 25) -> list[float]:
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    samples_per_frame = int(16000 / fps)
    n_chunks = int(len(samples) / samples_per_frame)
    rms_list = []
    for f in range(n_chunks):
        chunk = samples[f * samples_per_frame : (f + 1) * samples_per_frame]
        rms_list.append(float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0)

    max_v = max(rms_list) if rms_list else 1.0
    return [min(1.0, v / (max_v + 1e-5)) for v in rms_list]


def render_animated_script_video(
    script_text: str,
    wav_path: str,
    output_mp4: str,
    style: str = "Style 2B: 2D Vector Broadcast Studio (Podcast Character Rig)",
    opus_overlay: str = "All Motion (Particles + Audio Visualizer) [RECOMMENDED]",
    topic_headline: str = "",
    fps: int = 25,
    progress_cb=None
) -> dict:
    from stickman_styles import render_stickman_styled_frame
    from opus_motion import OpusMotionOverlay

    audio_env = extract_audio_envelope(wav_path, fps=fps)
    total_frames = len(audio_env)
    out_w, out_h = 1080, 1920

    silent_mp4 = output_mp4 + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (out_w, out_h))

    # Split script into timed subtitle sentences
    words = script_text.split()
    words_per_frame = len(words) / max(1, total_frames)

    # Base synthetic pose landmarks (centered talking avatar)
    def make_talking_landmarks(frame_idx: int, loudness: float) -> np.ndarray:
        lms = np.zeros((33, 2), dtype=np.float32)
        # Gentle swaying breathing motion
        sway_x = math.sin(frame_idx * 0.05) * 8
        sway_y = math.cos(frame_idx * 0.08) * 5

        head_cx = (out_w // 2) + sway_x
        head_cy = int(out_h * 0.38) + sway_y

        lms[0] = [head_cx, head_cy]  # nose / head center
        # Shoulders
        sh_w = 210
        lms[11] = [head_cx - sh_w, head_cy + 170]
        lms[12] = [head_cx + sh_w, head_cy + 170]
        # Hips
        lms[23] = [head_cx - 150, int(out_h * 0.75)]
        lms[24] = [head_cx + 150, int(out_h * 0.75)]

        # Gesturing hands (sync with dialogue peaks)
        arm_gesture = math.sin(frame_idx * 0.12) * 25 * loudness
        lms[13] = [head_cx - sh_w - 60, head_cy + 340 + arm_gesture]
        lms[14] = [head_cx + sh_w + 60, head_cy + 340 - arm_gesture]
        lms[15] = [head_cx - 180 + arm_gesture, int(out_h * 0.65)]
        lms[16] = [head_cx + 180 - arm_gesture, int(out_h * 0.64) - arm_gesture]

        return lms

    prev_lms = None
    dummy_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for i in range(total_frames):
        loudness = audio_env[i]
        sim_lms = make_talking_landmarks(i, loudness)

        # 1. Render character avatar rig
        rendered, prev_lms = render_stickman_styled_frame(
            dummy_frame, style, loudness, out_w, out_h, prev_lms, i, sim_lms=sim_lms
        )

        # 2. Render Opus 5.5 Motion Overlays
        if opus_overlay != "None":
            if "Visualizer" in opus_overlay:
                OpusMotionOverlay.draw_audio_reactive_visualizer(rendered, (out_w // 2, out_h - 220), 100, loudness, i)
            elif "Cyber" in opus_overlay:
                OpusMotionOverlay.draw_cyber_grid(rendered, i)
            elif "Particles" in opus_overlay:
                OpusMotionOverlay.draw_floating_particles(rendered, i)
            elif "All Motion" in opus_overlay:
                OpusMotionOverlay.draw_floating_particles(rendered, i, count=25)
                OpusMotionOverlay.draw_audio_reactive_visualizer(rendered, (out_w // 2, out_h - 220), 90, loudness, i)

        # 3. Kinetic Headline Banner at top
        if topic_headline:
            OpusMotionOverlay.draw_kinetic_quote_banner(rendered, topic_headline, i, loudness)

        # 4. Dynamic Word-by-Word Subtitle Bar
        current_word_idx = min(len(words) - 1, int(i * words_per_frame))
        start_w = max(0, current_word_idx - 3)
        end_w = min(len(words), current_word_idx + 4)
        subtitle_chunk = " ".join(words[start_w:end_w])

        # Draw subtitle box
        sub_y = int(out_h * 0.82)
        font = cv2.FONT_HERSHEY_DUPLEX
        sub_scale = 1.05
        sub_thick = 2
        (sw, sh), _ = cv2.getTextSize(subtitle_chunk, font, sub_scale, sub_thick)
        sub_x = max(40, (out_w - sw) // 2)

        # Subtitle background pill
        pad = 20
        cv2.rectangle(rendered, (sub_x - pad, sub_y - sh - pad), (sub_x + sw + pad, sub_y + pad), (10, 12, 18), -1)
        cv2.rectangle(rendered, (sub_x - pad, sub_y - sh - pad), (sub_x + sw + pad, sub_y + pad), (0, 220, 255), 2, cv2.LINE_AA)

        # High-visibility text
        cv2.putText(rendered, subtitle_chunk, (sub_x + 2, sub_y + 2), font, sub_scale, (0, 0, 0), sub_thick + 2, cv2.LINE_AA)
        cv2.putText(rendered, subtitle_chunk, (sub_x, sub_y), font, sub_scale, (255, 255, 255), sub_thick, cv2.LINE_AA)

        writer.write(rendered)
        if progress_cb and i % 15 == 0:
            progress_cb(i / total_frames, f"Rendering Short: Frame {i+1} / {total_frames}")

    writer.release()

    # Mux clean TTS audio into output MP4
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", silent_mp4, "-i", wav_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", output_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if os.path.exists(silent_mp4):
        os.remove(silent_mp4)

    return {
        "frames": total_frames,
        "duration": total_frames / float(fps),
        "output_path": output_mp4
    }
