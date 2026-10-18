from __future__ import annotations
import os, sys, tempfile, subprocess, json, time
from pathlib import Path

# Force UTF-8 on Windows terminal so external libraries cannot crash on non-ASCII characters
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import streamlit as st
import imageio_ffmpeg

APP_TITLE = "Video-to-2D Animation Studio"
MAX_SECONDS = 120

st.set_page_config(page_title=APP_TITLE, page_icon="🎞️", layout="centered")


def ffmpeg_exe():
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_cmd(cmd, timeout=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def probe_video(path: str) -> dict:
    # FFmpeg's input banner gives stream information without requiring a separate ffprobe binary.
    result = run_cmd([ffmpeg_exe(), "-hide_banner", "-i", path])
    text = result.stderr.decode("utf-8", errors="replace")
    duration = None
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
    audio = bool(re.search(r"Stream #\d+:\d+.*Audio:", text, re.I))
    video_match = re.search(r"Stream #\d+:\d+.*Video:.*?(\d{2,5})x(\d{2,5})", text, re.I)
    resolution = (int(video_match.group(1)), int(video_match.group(2))) if video_match else None
    return {"duration": duration, "has_audio": audio, "resolution": resolution, "raw": text}


def fit_frame(frame: np.ndarray, out_w: int, out_h: int, mode: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if mode == "Preserve source aspect ratio":
        scale = min(out_w / w, out_h / h)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x, y = (out_w-nw)//2, (out_h-nh)//2
        canvas[y:y+nh, x:x+nw] = resized
        return canvas
    if mode == "Fit with blurred background":
        scale_bg = max(out_w / w, out_h / h)
        bg = cv2.resize(frame, (int(w*scale_bg), int(h*scale_bg)), interpolation=cv2.INTER_AREA if scale_bg < 1 else cv2.INTER_LINEAR)
        by, bx = (bg.shape[0]-out_h)//2, (bg.shape[1]-out_w)//2
        bg = bg[by:by+out_h, bx:bx+out_w]
        bg = cv2.GaussianBlur(bg, (0, 0), 18)
        scale = min(out_w / w, out_h / h)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        fg = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x, y = (out_w-nw)//2, (out_h-nh)//2
        bg[y:y+nh, x:x+nw] = fg
        return bg
    # Center crop to portrait. No stretching.
    target_ratio = out_w / out_h
    ratio = w / h
    if ratio > target_ratio:
        crop_w = int(h * target_ratio)
        x = max(0, (w-crop_w)//2)
        frame = frame[:, x:x+crop_w]
    elif ratio < target_ratio:
        crop_h = int(w / target_ratio)
        y = max(0, (h-crop_h)//2)
        frame = frame[y:y+crop_h, :]
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA if frame.shape[1] > out_w else cv2.INTER_LINEAR)


def cartoonize(frame: np.ndarray, strength: int, edge_strength: int, palette: int) -> np.ndarray:
    # CPU-friendly classic cartoon conversion. This is stylization, not full manual redraw.
    smooth = cv2.bilateralFilter(frame, d=7, sigmaColor=35 + strength*2, sigmaSpace=35 + strength*2)
    # Color quantization using reduced levels; keeps the effect reasonably light on RAM.
    levels = max(4, 12 - palette)
    quant = (smooth // (256 // levels)) * (256 // levels)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2 + edge_strength)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(quant, edges3)
    # Lightly mix the quantized color regions with outlines; adjustable edge intensity.
    if edge_strength < 3:
        result = cv2.addWeighted(quant, 0.25, result, 0.75, 0)
    return result


@st.cache_resource
def get_anime_model():
    import torch
    model = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='face_paint_512_v2', trust_repo=True)
    model.eval()
    return model


def cartoonize_anime(frame: np.ndarray) -> np.ndarray:
    import torch
    model = get_anime_model()
    h, w = frame.shape[:2]
    tw, th = (w // 16) * 16, (h // 16) * 16
    resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.inference_mode():
        out = model(tensor)
    out_img = ((out.squeeze(0).permute(1, 2, 0).clamp(-1, 1).numpy() + 1.0) * 127.5).astype(np.uint8)
    bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    if (tw, th) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return bgr


def render_video(input_path: str, silent_path: str, out_w: int, out_h: int, fps: float,
                 framing: str, strength: int, edge_strength: int, palette: int,
                 max_frames: int | None, progress_cb, style: str = "Classic hand-drawn cartoon",
                 opus_overlay: str = "None", kinetic_text: str = ""):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video. Try converting it to MP4 (H.264) first.")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1 or src_fps > 240:
        src_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration = frame_count / src_fps if frame_count else 0
    # Never upsample frame rate: this keeps the visual timeline close to the source duration.
    target_fps = min(30.0, max(12.0, float(fps)), src_fps)
    # Sample source frames to the chosen output FPS while processing incrementally.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, target_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize video writer.")
    prev_lms = None
    audio_env = None
    if "Style 1" in style or "Style 2" in style or opus_overlay != "None":
        try:
            import wave
            wav_path = silent_path + ".wav"
            run_cmd([ffmpeg_exe(), "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])
            with wave.open(wav_path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            samples_per_frame = int(16000 / target_fps)
            n_chunks = int(len(samples) / samples_per_frame)
            rms_list = [float(np.sqrt(np.mean(samples[f*samples_per_frame:(f+1)*samples_per_frame]**2))) if len(samples[f*samples_per_frame:(f+1)*samples_per_frame]) > 0 else 0.0 for f in range(n_chunks)]
            max_v = max(rms_list) if rms_list else 1.0
            audio_env = [v / (max_v + 1e-5) for v in rms_list]
        except Exception:
            audio_env = None

    i = 0
    source_index = 0
    next_output_time = 0.0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_time = source_index / src_fps
            # Select the source frame nearest each output timestamp; no full video is held in RAM.
            if source_time + (0.5 / src_fps) >= next_output_time:
                loudness = audio_env[i] if audio_env and i < len(audio_env) else 0.3
                if "Style 1" in style or "Style 2" in style:
                    from stickman_styles import render_stickman_styled_frame
                    converted, prev_lms = render_stickman_styled_frame(frame, style, loudness, out_w, out_h, prev_lms, i)
                elif "Anime" in style:
                    fitted = fit_frame(frame, out_w, out_h, framing)
                    converted = cartoonize_anime(fitted)
                else:
                    fitted = fit_frame(frame, out_w, out_h, framing)
                    converted = cartoonize(fitted, strength, edge_strength, palette)

                # Apply Opus 5.5 Motion Graphic Overlays
                if opus_overlay != "None":
                    from opus_motion import OpusMotionOverlay
                    if "Visualizer" in opus_overlay:
                        OpusMotionOverlay.draw_audio_reactive_visualizer(converted, (out_w // 2, out_h - 220), 100, loudness, i)
                    elif "Cyber" in opus_overlay:
                        OpusMotionOverlay.draw_cyber_grid(converted, i)
                    elif "Particles" in opus_overlay:
                        OpusMotionOverlay.draw_floating_particles(converted, i)
                    elif "All Motion" in opus_overlay:
                        OpusMotionOverlay.draw_floating_particles(converted, i, count=25)
                        OpusMotionOverlay.draw_audio_reactive_visualizer(converted, (out_w // 2, out_h - 220), 90, loudness, i)

                    if kinetic_text:
                        OpusMotionOverlay.draw_kinetic_quote_banner(converted, kinetic_text, i, loudness)

                writer.write(converted)
                i += 1
                next_output_time += 1.0 / target_fps
                if max_frames and i >= max_frames:
                    break
            source_index += 1
            if frame_count:
                progress_cb(min(0.99, source_index / frame_count), f"Rendering frames: {source_index:,} / {frame_count:,} source frames")
            elif source_index % 10 == 0:
                progress_cb(0.5, f"Processed {source_index:,} source frames…")
    finally:
        cap.release()
        writer.release()
    if i == 0:
        raise RuntimeError("No video frames were decoded.")
    return {"frames": i, "source_fps": src_fps, "output_fps": target_fps,
            "duration": i / target_fps, "elapsed": time.time()-t0}


def mux_original_audio(original_path: str, silent_video_path: str, output_path: str, has_audio: bool):
    ffmpeg = ffmpeg_exe()
    if has_audio:
        # Copy original audio packets directly; never decode/re-encode audio.
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", silent_video_path, "-i", original_path,
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "copy", "-map_metadata", "1",
               "-movflags", "+faststart", output_path]
    else:
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", silent_video_path,
               "-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]
    result = run_cmd(cmd, timeout=1800)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        if has_audio:
            raise RuntimeError("Audio stream-copy muxing failed. The app did not re-encode your audio. "
                               "Try an MP4/H.264 source with AAC audio, or choose a compatible output container.\n\n" + err[-2500:])
        raise RuntimeError("FFmpeg MP4 encoding failed:\n" + err[-2500:])


def verify_output(path: str, expected_audio: bool):
    info = probe_video(path)
    if not info["resolution"]:
        raise RuntimeError("Output verification failed: video stream could not be detected.")
    if expected_audio and not info["has_audio"]:
        raise RuntimeError("Output verification failed: original audio track is missing.")
    return info


st.title("🎞️ Video-to-2D Animation Studio")
st.caption("Turn Real-Life Videos OR Scripts into Viral 2D Animated Shorts & Reels with Lossless Audio")

tab_diffusion, tab_video, tab_script = st.tabs([
    "🌌 AI Video Diffusion Generator (LTX-Video & Wan2.1)",
    "📹 Video-to-2D Animation Studio",
    "⚡ Script-to-Video Generator (Unlimited 1-Min+ Shorts)"
])

with tab_diffusion:
    st.subheader("🌌 Pure Neural AI Video Diffusion Generator (Like Sora & LTX)")
    st.caption("Generate cinematic AI videos from text prompts or reference images using free open-source foundation models (**Lightricks LTX-Video** & **Wan2.1**) powered by high-performance cloud GPUs.")

    import diffusion_client
    diff_mode = st.radio("Diffusion Mode", ["Text-to-Video (LTX-Video Distilled)", "Image-to-Video (Wan2.1 Fast)"], horizontal=True)

    if diff_mode.startswith("Text-to-Video"):
        diff_prompt = st.text_area(
            "Cinematic Prompt",
            value="A confident entrepreneur standing on the neon-lit rooftop of a futuristic skyscraper at night, wind gently moving their jacket, cinematic anamorphic lighting, photorealistic 8k, highly detailed, dramatic camera push-in.",
            height=100
        )
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            diff_aspect = st.selectbox("Resolution / Framing", ["Vertical 9:16 (512 × 704 - Shorts/Reels)", "Landscape 16:9 (704 × 512 - YouTube)"])
        with col_d2:
            diff_dur = st.slider("Duration (seconds)", 2.0, 5.0, 3.0, 0.5)
        with col_d3:
            diff_seed = st.number_input("Seed", value=42, step=1)

        diff_w, diff_h = (512, 704) if "Vertical" in diff_aspect else (704, 512)

        if st.button("🚀 Generate AI Diffusion Video", type="primary", use_container_width=True, key="btn_gen_ltx"):
            with st.spinner("Connecting to Lightricks LTX-Video Cloud GPU endpoint... Generating diffusion frames..."):
                try:
                    res_video_path = diffusion_client.generate_ltx_diffusion_video(
                        prompt=diff_prompt,
                        width=diff_w,
                        height=diff_h,
                        duration=diff_dur,
                        seed=int(diff_seed)
                    )
                    st.success("Cinematic AI video generated successfully!")
                    with open(res_video_path, "rb") as f:
                        v_bytes = f.read()
                    st.video(v_bytes)
                    st.download_button("Download Diffusion MP4", data=v_bytes, file_name="ai_diffusion_video.mp4", mime="video/mp4", use_container_width=True)
                except Exception as ex:
                    st.error(f"Generation failed: {ex}")
                    st.info("Tip: Public Hugging Face GPU queues can occasionally be busy. Try again in 10-20 seconds or use a slightly shorter prompt.")

    else:
        wan_img = st.file_uploader("Upload starting image / portrait", type=["jpg", "png", "webp"], key="wan_img_upload")
        wan_prompt = st.text_area(
            "Motion / Animation Prompt",
            value="Make this person come alive, talking naturally with realistic facial expressions and head turns, cinematic film lighting, high quality motion",
            height=80
        )
        if wan_img:
            st.image(wan_img, width=300)
            if st.button("🚀 Animate Image with Wan2.1 Diffusion", type="primary", use_container_width=True, key="btn_gen_wan"):
                with tempfile.TemporaryDirectory() as t_dir:
                    t_img_path = os.path.join(t_dir, "input.jpg")
                    with open(t_img_path, "wb") as f:
                        f.write(wan_img.getbuffer())
                    with st.spinner("Connecting to Wan2.1 Fast Diffusion endpoint... Rendering frames..."):
                        try:
                            res_wan_path = diffusion_client.generate_wan_diffusion_video(
                                image_path=t_img_path,
                                prompt=wan_prompt,
                                width=512,
                                height=704,
                                duration=3.0
                            )
                            st.success("Wan2.1 video generated successfully!")
                            with open(res_wan_path, "rb") as f:
                                w_bytes = f.read()
                            st.video(w_bytes)
                            st.download_button("Download Wan2.1 MP4", data=w_bytes, file_name="wan2_diffusion_video.mp4", mime="video/mp4", use_container_width=True)
                        except Exception as ex:
                            st.error(f"Generation failed: {ex}")
                            st.info("Tip: Public Hugging Face GPU queues can occasionally be busy. Try again in a moment.")

with tab_script:
    st.subheader("⚡ Unlimited AI Script-to-Shorts Generator")
    st.caption("Generate complete 1080×1920 vertical Shorts directly from a script or topic with neural voiceover, animated character, dynamic lip-sync, and kinetic captions.")

    from script_to_video import VOICES, generate_speech_audio, render_animated_script_video

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        script_topic = st.text_input("Shorts Topic / Hook Headline", value="ACHIEVERS MINDSET: CONSISTENCY")
    with col_s2:
        selected_voice_name = st.selectbox("Neural Voice Narrator (Free Microsoft Edge TTS)", list(VOICES.keys()))
        selected_voice_id = VOICES[selected_voice_name]

    script_content = st.text_area(
        "Voiceover Script (Write or paste your script; 130–150 words = ~1 minute)",
        value="Stop waiting for motivation. Motivation is a trap. You don't need to feel like doing the work, you just need to start. True achievers show up on the days when it is pouring rain and their energy is zero. Ruthless consistency will always beat sudden bursts of intensity. Build the system, execute every single morning, and let the compound effect do the rest.",
        height=140
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        script_char_style = st.selectbox("Character Avatar Style", [
            "Style 2B: 2D Vector Broadcast Studio (Podcast Character Rig)",
            "Style 2A: Modern Studio Tech (Red Beanie Zeke & Cyan Grid)",
            "Style 1B: Classic Minimalist Dark (White Stickman Line-Art)",
            "Style 1A: Classic Minimalist Light (Black Stickman Line-Art)"
        ], key="script_char_style")
    with col_opt2:
        script_motion_overlay = st.selectbox("Motion Graphics Overlay", [
            "All Motion (Particles + Audio Visualizer) [RECOMMENDED]",
            "Audio Visualizer Ring (Reactive Equalizer Ring)",
            "Cyber Grid Horizon (Synthwave Floor Grid)",
            "Floating Ambient Bokeh Particles",
            "None (Character Only)"
        ], key="script_motion_overlay")

    if st.button("🚀 Generate 1-Minute+ Animated Short", type="primary", use_container_width=True):
        if not script_content.strip():
            st.error("Please provide a script text.")
        else:
            with tempfile.TemporaryDirectory(prefix="script2d_") as script_workdir:
                script_wav = os.path.join(script_workdir, "speech.wav")
                script_mp4 = os.path.join(script_workdir, "generated_short.mp4")
                p_bar = st.progress(0.0)
                p_status = st.empty()
                try:
                    p_status.text("Synthesizing neural voiceover with Edge-TTS…")
                    p_bar.progress(0.15)
                    audio_dur = generate_speech_audio(script_content, selected_voice_id, script_wav)
                    p_status.text(f"Voice synthesized ({audio_dur:.1f}s). Rendering 2D avatar and kinetic typography…")
                    p_bar.progress(0.35)

                    def script_cb(val, msg):
                        p_bar.progress(0.35 + float(val) * 0.60)
                        p_status.text(msg)

                    res_video = render_animated_script_video(
                        script_content, script_wav, script_mp4,
                        style=script_char_style,
                        opus_overlay=script_motion_overlay,
                        topic_headline=script_topic,
                        fps=25,
                        progress_cb=script_cb
                    )
                    p_bar.progress(1.0)
                    p_status.success(f"Generated {res_video['duration']:.1f}s Short successfully!")

                    with open(script_mp4, "rb") as f:
                        short_bytes = f.read()
                    st.video(short_bytes)
                    st.download_button(
                        "Download Generated 1-Minute Short MP4",
                        data=short_bytes,
                        file_name=f"{script_topic.lower().replace(' ', '_')[:30]}_short.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Generation failed: {ex}")
                    st.exception(ex)

with tab_video:
    st.info("Upload any talking-head video or podcast clip below to convert it into a 2D animated avatar with reactive motion graphics.")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"])
if uploaded:
    with tempfile.TemporaryDirectory(prefix="video2d_") as workdir:
        source_path = os.path.join(workdir, "source" + Path(uploaded.name).suffix.lower())
        with open(source_path, "wb") as f:
            f.write(uploaded.getbuffer())
        info = probe_video(source_path)
        if info["duration"] is None:
            st.error("Could not read video duration. Try a standard MP4/MOV file.")
            st.stop()
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"{info['duration']:.1f} s")
        c2.metric("Resolution", f"{info['resolution'][0]} × {info['resolution'][1]}" if info["resolution"] else "Unknown")
        c3.metric("Original audio", "Present" if info["has_audio"] else "None")
        if info["duration"] > MAX_SECONDS:
            st.error("This video is longer than 120 seconds. Trim it and upload again.")
            st.stop()
        st.video(uploaded)
        st.subheader("1. Character & Animation Style (from Stickman Video Director)")
        style = st.selectbox("Stickman Director Character Style", [
            "Style 2B: 2D Vector Broadcast Studio (Podcast Character Rig)",
            "Style 2A: Modern Studio Tech (Red Beanie Zeke & Cyan Grid)",
            "Style 1B: Classic Minimalist Dark (White Stickman Line-Art)",
            "Style 1A: Classic Minimalist Light (Black Stickman Line-Art)",
            "AI Anime Character (Neural Cel Shading)",
            "Classic hand-drawn cartoon",
            "Soft illustrated cartoon"
        ], help="Uses MediaPipe pose tracking, temporal EMA smoothing, and audio envelope lip-sync.")

        st.subheader("2. Motion Graphics & Visualizer (from Awesome Opus 5.5)")
        c_op1, c_op2 = st.columns([1, 1])
        with c_op1:
            opus_overlay = st.selectbox("Opus Code-Driven Motion Overlay", [
                "All Motion (Particles + Audio Visualizer) [RECOMMENDED]",
                "Audio Visualizer Ring (Reactive Equalizer Ring)",
                "Cyber Grid Horizon (Synthwave Floor Grid)",
                "Floating Ambient Bokeh Particles",
                "None (Character Only)"
            ])
        with c_op2:
            kinetic_text = st.text_input("Kinetic Quote Banner Text (Optional)", value="", placeholder="e.g. CONSISTENCY BEATS INTENSITY")

        framing = st.selectbox("Framing", ["Center crop to 9:16", "Fit with blurred background", "Preserve source aspect ratio"])
        if style.startswith("AI Anime"):
            quality = st.selectbox("Output size (Anime Neural default)", ["540 × 960 (recommended for Anime)", "720 × 1280 (slower)", "1080 × 1920 (HD)"])
            default_fps = 15
        elif "Style 1" in style or "Style 2" in style:
            quality = st.selectbox("Output size (Vector Motion-Capture default)", ["1080 × 1920 (Full HD Vector)", "720 × 1280 (Medium)"])
            default_fps = 25
        else:
            quality = st.selectbox("Output size (CPU-friendly default)", ["720 × 1280 (recommended)", "1080 × 1920 (slower)", "Preserve source aspect ratio at 720p"])
            default_fps = 24
        if quality.startswith("1080"):
            out_w, out_h = 1080, 1920
        elif quality.startswith("540"):
            out_w, out_h = 540, 960
        elif quality.startswith("Preserve"):
            src_w, src_h = info["resolution"] or (720, 1280)
            scale = min(720 / src_w, 1280 / src_h, 1.0)
            out_w, out_h = max(2, int(src_w*scale)//2*2), max(2, int(src_h*scale)//2*2)
        else:
            out_w, out_h = 720, 1280
        if style == "Soft illustrated cartoon":
            strength, edge_strength, palette = 7, 1, 7
        else:
            strength, edge_strength, palette = 5, 4, 4
        with st.expander("Advanced fine-tuning"):
            strength = st.slider("Smoothing strength", 1, 10, strength)
            edge_strength = st.slider("Outline strength", 1, 8, edge_strength)
            palette = st.slider("Color simplification", 1, 8, palette)
            output_fps = st.select_slider("Output frame rate", options=[12, 15, 20, 24, 25, 30], value=default_fps)
        st.caption("Original audio is mapped from the source and copied without re-encoding when the source audio codec is compatible with MP4. No music, narration, or effects are added.")
        if st.button("Generate animated video", type="primary", use_container_width=True):
            silent_path = os.path.join(workdir, "animated_silent.mp4")
            output_path = os.path.join(workdir, "animated_final.mp4")
            progress = st.progress(0.0)
            status = st.empty()
            def cb(value, message):
                progress.progress(min(1.0, max(0.0, float(value))))
                status.text(message)
            try:
                cb(0.01, "Starting frame-by-frame conversion…")
                result = render_video(source_path, silent_path, out_w, out_h, output_fps, framing,
                                      strength, edge_strength, palette, None, cb, style=style,
                                      opus_overlay=opus_overlay, kinetic_text=kinetic_text)
                cb(0.94, "Muxing original audio without re-encoding…" if info["has_audio"] else "Finalizing silent video…")
                mux_original_audio(source_path, silent_path, output_path, info["has_audio"])
                verify = verify_output(output_path, info["has_audio"])
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                progress.progress(1.0)
                status.success("Conversion complete and output stream checks passed.")
                st.video(video_bytes)
                st.download_button("Download animated MP4", data=video_bytes,
                                   file_name=f"{Path(uploaded.name).stem}_2d_cartoon.mp4",
                                   mime="video/mp4", use_container_width=True)
                st.json({"frames_processed": result["frames"], "output_fps": result["output_fps"],
                         "output_duration_seconds": round(verify["duration"], 2) if verify["duration"] else None,
                         "original_audio_present": info["has_audio"],
                         "audio_stream_copy_requested": info["has_audio"],
                         "note": "Stream presence verified; this app does not perform a bit-for-bit audio packet comparison."})
            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.exception(e)

st.divider()

with st.expander("✨ Awesome Opus 5.5 Motion Graphics & Explainer Library"):
    st.markdown("Curated library of **703 viral code-driven animations, 3D scenes & explainers** powered by [`yihui-dev/awesome-opus5-5-videos`](https://github.com/yihui-dev/awesome-opus5-5-videos).")
    from opus_motion import load_opus_prompts, search_opus_prompts
    
    col_op1, col_op2 = st.columns([1, 2])
    with col_op1:
        cat_filter = st.selectbox("Category Filter", ["all", "motion", "explainer", "3d", "interactive"])
    with col_op2:
        search_query = st.text_input("Search prompt text or tech (e.g. canvas, gsap, threejs, particles)", value="")
        
    filtered = search_opus_prompts(search_query, category=cat_filter, limit=12)
    st.caption(f"Showing {len(filtered)} prompts")
    
    for item in filtered:
        with st.container():
            st.markdown(f"**@{item.get('author', 'creator')}** — `{item.get('category')}` · *{', '.join(item.get('tech_tags', []))}*")
            if item.get("skillry_url"):
                st.markdown(f"[▶ Watch Remake on Skillry]({item['skillry_url']}) · [Original Post]({item.get('post_url', '#')})")
            st.code(item.get("prompt", ""), language="text")
            st.divider()

with st.expander("🎬 Stickman Video Director (AI Storyboard & Prompt Generator)"):
    st.markdown("Generate structured 5-stage director storyboards and production prompts powered by [`kaomei/stickman-video-director`](https://github.com/kaomei/stickman-video-director).")
    topic = st.text_input("Story topic or script idea", value="Achievers Mindset: Why consistency beats intensity every single time")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        dir_style = st.selectbox("Director Visual Style", [
            "Style 2A: Modern Studio Tech (Apple-keynote grid, cyan UI, red beanie Zeke)",
            "Style 1B: Classic Minimalist Dark (Pitch black canvas, glowing white lines)",
            "Style 1A: Classic Minimalist Light (Pure white canvas, black stick lines)",
            "Style 2B: Modern Cinematic Story (Colored narrative depth)",
            "Style 3: Prehistoric Dark Comedy (Caveman comedy)",
            "Style 4: Cozy Storytelling (Warm atmospheric 2D cartoon)"
        ])
    with col_d2:
        dir_duration = st.selectbox("Duration", ["30s (3 clips)", "60s (6 clips)", "90s (9 clips)"])

    if st.button("Generate Director Storyboard & Prompts"):
        st.subheader("📋 Phase A: 5-Stage Heartbeat Storyboard Proposal")
        st.markdown(f"""
**Topic**: {topic}  
**Visual Style**: {dir_style}  
**Duration**: {dir_duration} | **Aspect Ratio**: 9:16 (Shorts/Reels)  

---

#### 1. 🪝 Stage 1: Golden Hook [0–5s]
* **Narrative Beat**: *Disrupt assumption immediately.*
* **Visual Device**: Character in center frame facing an oversized obstacle or floating question mark.
* **English VO**: *"You think achievers work 16 hours a day without burnout? That's the biggest myth on the internet."*

#### 2. ⚡ Stage 2: Disrupting Assumptions [5–12s]
* **Narrative Beat**: *Shatter conventional thinking.*
* **Visual Device**: Character taps perspective floor grid, causing an expanding ripple wave that shatters a calendar into dust.
* **English VO**: *"Intensity gets you started, but consistency is the only engine that keeps the wheels turning."*

#### 3. 🔍 Stage 3: Unveiling Insider Secrets [12–20s]
* **Narrative Beat**: *Pull back the curtain on hidden mechanics.*
* **Visual Device**: Floating cyan glass UI reveals a micro-habit progress bar climbing smoothly while an intensity spikes line crashes.
* **English VO**: *"Top performers don't rely on willpower. They build friction-free systems that run automatically."*

#### 4. 💡 Stage 4: Ultimate Truth Revelation [20–25s]
* **Narrative Beat**: *Cognitive payoff.*
* **Visual Device**: Character strides forward confidently through an open portal into clean horizon light.
* **English VO**: *"Show up for 20 minutes when you least feel like it. That is how champions are made."*

#### 5. 🎯 Stage 5: Elevation & High-Engagement Discussion [25–30s]
* **Narrative Beat**: *Punchline and discussion catalyst.*
* **Visual Device**: Character turns to camera and points toward a glowing interactive prompt.
* **English VO**: *"What is the one habit you have kept for more than 90 days? Tell me in the comments below."*
        """)

st.caption("Runs locally. Uploaded video is processed on this computer; it is not sent to a cloud animation service. Use only videos you own or have permission to transform.")
