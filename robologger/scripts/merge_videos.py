import glob
import os
import cv2
import imageio
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

import os
import glob


import click
import glob
import os
from real_env.utils.visualize_utils import merge_videos
import zarr


def merge_videos(
    video_paths: list[str],
    success_labels: list[bool],
    output_path: str,
    rows: int,
    cols: int,
    margin_size: int,
    extend_time: float = 1,
    blend_alpha: float = 0.2,
    cropping_top_down_left_right: tuple[float, float, float, float] = (0, 0, 0, 0),
    max_time: float = -1,
    speed: float = 1,
):
    os.environ["FFMPEG_BINARY"] = f"{os.environ['CONDA_PREFIX']}/bin/ffmpeg"

    from moviepy import VideoFileClip, clips_array, vfx

    # --- Configuration ---
    # 1. Path to your video folder
    # 2. Output file name
    expected_clips = rows * cols
    encoder = "h264"
    # encoder = "h264"
    ffmpeg_params = ["-c:v", encoder]

    assert cropping_top_down_left_right[0] + cropping_top_down_left_right[1] < 1
    assert cropping_top_down_left_right[2] + cropping_top_down_left_right[3] < 1

    assert len(video_paths) == expected_clips
    assert len(success_labels) == expected_clips

    # 2. Load clips and find the maximum duration
    video_clips: list[VideoFileClip] = []
    max_duration = 0
    print("Loading clips and determining maximum duration...")

    for i, path in enumerate(video_paths):
        try:
            clip = VideoFileClip(path)
            if max_time > 0:
                clip = clip.subclipped(0, max_time)
            clip = clip.resized(new_size=1 / max(rows, cols))
            clip = clip.with_speed_scaled(speed)
            assert isinstance(clip, VideoFileClip)
            clip_width, clip_height = clip.w, clip.h
            cropping = vfx.Crop(
                y1=cropping_top_down_left_right[0] * clip_height,
                y2=(1 - cropping_top_down_left_right[1]) * clip_height,
                x1=cropping_top_down_left_right[2] * clip_width,
                x2=(1 - cropping_top_down_left_right[3]) * clip_width,
            )
            clip = cropping.apply(clip)
            add_margin = vfx.Margin(
                color=(0, 0, 0),
                margin_size=margin_size,
            )
            clip = add_margin.apply(clip)
            assert isinstance(clip, VideoFileClip)
            video_clips.append(clip)
            if isinstance(clip.duration, float) and clip.duration > max_duration:
                max_duration = clip.duration
        except Exception as e:
            print(f"Could not load clip {path}: {e}")
            return

    print(
        f"Maximum video duration found: {max_duration:.2f} seconds. Will extend to {max_duration + extend_time: .2f} seconds"
    )
    final_duration = max_duration + extend_time

    # 3. Standardize duration (Keep last frame if shorter)
    extended_clips = []
    for i, clip in enumerate(video_clips):
        # Overlay a green or red transparent mask on the last frame of the clip
        if success_labels[i]:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Red
        assert isinstance(clip.duration, float)

        last_frame = clip.get_frame(clip.duration - 1 / float(clip.fps))
        assert isinstance(last_frame, np.ndarray)
        last_frame = (
            last_frame * (1 - blend_alpha)
            + np.ones_like(last_frame) * np.array(color)[None, None, :] * blend_alpha
        )
        last_frame = last_frame.astype(np.uint8)

        additional_duration = final_duration - clip.duration
        last_frame_clip = ImageClip(last_frame).with_duration(additional_duration)

        extended_clip = concatenate_videoclips([clip, last_frame_clip])
        extended_clips.append(extended_clip)
        print(
            f"  - Extended clip {i+1} from {clip.duration:.2f}s to {final_duration:.2f}s (Last frame freeze)."
        )

    # 4. Arrange clips into a 4x5 grid structure
    # The clips_array function takes a 2D list (rows of columns)
    try:

        # Create a list of lists with row-first order
        # clip_rows = []
        # for i in range(rows):
        #     start_index = i * cols
        #     end_index = start_index + cols
        #     clip_rows.append(extended_clips[start_index:end_index])

        # Arrange the clips with column-first order
        clip_cols = []
        for i in range(cols):
            start_index = i * rows
            end_index = start_index + rows
            clip_cols.append(extended_clips[start_index:end_index])
        clip_rows = list(zip(*clip_cols))

        final_clip = clips_array(clip_rows)

    except Exception as e:
        print(f"Error during clips_array arrangement: {e}")
        return

    print(f"\nExporting final video to '{output_path}' using {encoder}...")

    overlay_text = TextClip(
        font="SauceCodeProNerdFont-Regular",
        text=f"Speed:{speed}x",
        color="white",
        font_size=30,
        size=(1000, 100),
        margin=(10, 10),
        method="caption",
        text_align="left",
        vertical_align="left",
        horizontal_align="top",
        duration=final_clip.duration,
    )
    final_clip = CompositeVideoClip([final_clip, overlay_text])

    final_clip.write_videofile(
        output_path,
        codec=encoder,
        # audio_codec=audio_codec,
        # temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        # CRITICAL: This passes the GPU encoder flag to FFmpeg
        ffmpeg_params=ffmpeg_params,
        # # Set to 1 for faster, single-threaded processing; increase if needed
        # n_jobs=1,
        # Set a high bitrate or a quality factor (CRF) for better quality
        bitrate="5000k",  # 5 Mbps
    )
    print(f"\n--- SUCCESS! Video saved to {output_path} ---")



@click.command()
@click.argument("video_dir", type=click.Path(exists=True))
@click.option("--rows", type=int, default=4)
@click.option("--cols", type=int, default=5)
@click.option("--camera_name", type=str, default="third_person_camera_0")
@click.option("--video_name", type=str, default="main_annotated")
@click.option("--start_episode", type=int, default=0)
@click.option("--end_episode", type=int, default=20)
def merge_video_files(
    video_dir: str,
    rows: int,
    cols: int,
    camera_name: str,
    video_name: str,
    start_episode: int,
    end_episode: int,
):
    episode_dirs = sorted(glob.glob(os.path.join(video_dir, "episode_*")))
    output_path = os.path.join(video_dir, f"episodes_{start_episode}_{end_episode}.mp4")

    if len(episode_dirs) > 0:
        video_paths = []
        success_labels = []
        for episode_dir in episode_dirs[start_episode:end_episode]:
            video_paths.append(
                os.path.join(episode_dir, f"{camera_name}.zarr", f"{video_name}.mp4")
            )
            metadata = zarr.open(os.path.join(episode_dir, "metadata.zarr"))
            success_labels.append(metadata.attrs["is_successful"])
        print(success_labels)
    else:
        video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        video_paths = sorted(video_paths)[: rows * cols]
        success_labels = [True] * len(video_paths)
        success_labels[0] = False

    cropping_top_down_left_right = (0.1, 0.1, 0.1, 0.1)
    max_time = 0
    margin_size = 3
    speed = 2

    merge_videos(
        video_paths=video_paths,
        output_path=output_path,
        rows=rows,
        cols=cols,
        cropping_top_down_left_right=cropping_top_down_left_right,
        margin_size=margin_size,
        max_time=max_time,
        speed=speed,
        success_labels=success_labels,
    )


if __name__ == "__main__":
    merge_video_files()
