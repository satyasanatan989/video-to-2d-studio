"""
Diffusion AI Video Generation Engine (Option 1)
Connects directly to top-tier free open-source diffusion models:
- Lightricks LTX-Video Distilled (Direct Text-to-Video & Video-to-Video)
- Wan2.1 Fast (Photorealistic Image-to-Video & Diffusion Motion)
"""
from __future__ import annotations
import os
import shutil
import cv2
import numpy as np
from gradio_client import Client, handle_file

LTX_SPACE = "Lightricks/ltx-video-distilled"
WAN_SPACE = "multimodalart/wan2-1-fast"

_ltx_client = None
_wan_client = None

def get_ltx_client():
    global _ltx_client
    if _ltx_client is None:
        _ltx_client = Client(LTX_SPACE)
    return _ltx_client

def get_wan_client():
    global _wan_client
    if _wan_client is None:
        _wan_client = Client(WAN_SPACE)
    return _wan_client


def generate_ltx_diffusion_video(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 512,
    height: int = 704,
    duration: float = 3.0,
    seed: int = 42,
    randomize_seed: bool = True
) -> str:
    """Generates pure cinematic neural diffusion video using Lightricks LTX-Video"""
    client = get_ltx_client()
    result = client.predict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image_filepath=None,
        input_video_filepath=None,
        height_ui=float(height),
        width_ui=float(width),
        mode="text-to-video",
        duration_ui=float(duration),
        ui_frames_to_use=9.0,
        seed_ui=seed,
        randomize_seed=randomize_seed,
        ui_guidance_scale=1.0,
        improve_texture_flag=True,
        api_name="/text_to_video"
    )
    # result is (generated_video_dict, seed)
    video_info = result[0]
    if isinstance(video_info, dict) and "video" in video_info:
        return video_info["video"]
    elif isinstance(video_info, str):
        return video_info
    return str(video_info)


def generate_wan_diffusion_video(
    image_path: str,
    prompt: str,
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
    width: int = 512,
    height: int = 704,
    duration: float = 3.0,
    seed: int = 42,
    randomize_seed: bool = True
) -> str:
    """Generates photorealistic diffusion video from reference image using Wan2.1 Fast"""
    client = get_wan_client()
    img_param = handle_file(image_path)
    result = client.predict(
        input_image=img_param,
        prompt=prompt,
        height=float(height),
        width=float(width),
        negative_prompt=negative_prompt,
        duration_seconds=float(duration),
        guidance_scale=1.0,
        steps=4.0,
        seed=float(seed),
        randomize_seed=randomize_seed,
        api_name="/generate_video"
    )
    video_info = result[0]
    if isinstance(video_info, dict) and "video" in video_info:
        return video_info["video"]
    elif isinstance(video_info, str):
        return video_info
    return str(video_info)
