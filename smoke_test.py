"""Small local integration test: generate synthetic video+audio, cartoonize, mux original audio by stream copy."""
import os, tempfile, subprocess
from pathlib import Path
import cv2, numpy as np
import imageio_ffmpeg
from app import probe_video, render_video, mux_original_audio, verify_output

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
with tempfile.TemporaryDirectory(prefix="video2d_test_") as td:
    src = os.path.join(td, "source.mp4")
    silent = os.path.join(td, "silent.mp4")
    out = os.path.join(td, "out.mp4")
    # Create a 2-second synthetic source video.
    w, h, fps = 320, 240, 12
    writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    assert writer.isOpened(), "Could not create synthetic test video"
    for i in range(24):
        frame = np.full((h,w,3), (220,235,245), dtype=np.uint8)
        cv2.circle(frame, (50+i*8,120), 25, (30,80,220), -1)
        cv2.rectangle(frame, (150,60), (260,180), (40,180,80), -1)
        writer.write(frame)
    writer.release()
    # Add a 440 Hz tone as AAC audio to source; this is only test fixture audio.
    with_audio = os.path.join(td, "source_audio.mp4")
    cmd = [ffmpeg,"-y","-hide_banner","-loglevel","error","-i",src,"-f","lavfi","-i","sine=frequency=440:duration=2",
           "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest",with_audio]
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    info = probe_video(with_audio)
    assert info["has_audio"], "Test source audio not detected"
    render_video(with_audio, silent, 180, 320, 12, "Center crop to 9:16", 5, 3, 4, None, lambda *_: None)
    mux_original_audio(with_audio, silent, out, True)
    out_info = verify_output(out, True)
    assert out_info["has_audio"], "Output audio missing"
    assert os.path.getsize(out) > 10000, "Output unexpectedly small"
    print("PASS: synthetic input converted, MP4 output exists, original audio stream is present, stream-copy mux completed.")
    print("NOTE: this smoke test verifies stream presence and successful copy mux, not bit-for-bit packet equality.")
