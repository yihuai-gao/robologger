"""Convert robologger data to training-ready zarr format.

Usage:
    python scripts/convert_robologger_to_zarr.py \
        --input_dir ~/robologger_data \
        --output_dir ~/converted_data \
        --dataset_name task_name (e.g., "pick_and_place_back")
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from robot_utils.data_utils import resize_with_cropping


def load_timestamps(zarr_path, key):
    """Load timestamps from a zarr file."""
    return np.array(zarr.open(zarr_path, 'r')[key])


def extract_video_frames(video_path):
    """Extract all frames from a video file as RGB arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.uint8)


def resample_to_common_timestamps(data, src_timestamps, target_timestamps):
    """Resample data to target timestamps using linear interpolation."""
    if data.ndim == 1:
        return make_interp_spline(src_timestamps, data, k=1)(target_timestamps)

    return np.stack([
        make_interp_spline(src_timestamps, data[:, i], k=1)(target_timestamps)
        for i in range(data.shape[1])
    ], axis=1)


def load_and_resample_zarr_data(zarr_group, pos_key, quat_key, ts_key, target_ts):
    """Load position and quaternion data from zarr and resample to target timestamps."""
    timestamps = np.array(zarr_group[ts_key])
    pos = resample_to_common_timestamps(np.array(zarr_group[pos_key]), timestamps, target_ts)
    quat = resample_to_common_timestamps(np.array(zarr_group[quat_key]), timestamps, target_ts)
    return np.concatenate([pos, quat], axis=1).astype(np.float64)


def load_and_resample_1d(zarr_group, key, ts_key, target_ts):
    """Load 1D data from zarr and resample to target timestamps."""
    timestamps = np.array(zarr_group[ts_key])
    data = resample_to_common_timestamps(np.array(zarr_group[key]), timestamps, target_ts)
    return (data[:, None] if data.ndim == 1 else data).astype(np.float64)


def save_zarr_array(zarr_group, key, data, chunks):
    """Save array to zarr group with specified chunks."""
    compressor = zarr.Blosc(cname='lz4', clevel=5) if data.dtype == np.uint8 else None
    zarr_group.create_dataset(key, data=data, chunks=chunks, compressor=compressor, overwrite=True)


def convert_episode(episode_dir, episode_group):
    """Convert a single robologger episode to the target zarr format."""
    episode_dir = Path(episode_dir)

    required_files = [
        episode_dir / "third_person_camera_0.zarr", # NOTE: need to adapt to your path
        episode_dir / "right_arm.zarr",
        episode_dir / "right_end_effector.zarr"
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(f"Missing required file: {required_file}")

    camera_timestamps = load_timestamps(
        episode_dir / "third_person_camera_0.zarr", "main_timestamps"
    )
    T = len(camera_timestamps)

    print(f"  Extracting video frames...")
    frames = extract_video_frames(episode_dir / "third_person_camera_0.zarr" / "main.mp4") # NOTE: need to adapt to your path

    if frames.shape[1:3] != (256, 256):
        frames = np.array([resize_with_cropping(frame, (256, 256)) for frame in frames], dtype=np.uint8)

    save_zarr_array(episode_group, "third_person_camera", frames, (1000, 256, 256, 3))

    print(f"  Processing arm data...")
    arm_zarr = zarr.open(episode_dir / "right_arm.zarr", 'r') # NOTE: need to adapt to your path
    robot0_tcp = load_and_resample_zarr_data(
        arm_zarr, 'state_pos_xyz', 'state_quat_wxyz', 'state_timestamps', camera_timestamps
    )
    save_zarr_array(episode_group, "robot0_tcp_xyz_wxyz", robot0_tcp, (1000, 7))

    print(f"  Processing gripper data...")
    gripper_zarr = zarr.open(episode_dir / "right_end_effector.zarr", 'r') # NOTE: need to adapt to your path
    robot0_gripper = load_and_resample_1d(
        gripper_zarr, 'state_joint_pos', 'state_timestamps', camera_timestamps
    )
    save_zarr_array(episode_group, "robot0_gripper_width", robot0_gripper, (1000, 1))

    print(f"  Creating action data...")
    action0_tcp = load_and_resample_zarr_data(
        arm_zarr, 'target_pos_xyz', 'target_quat_wxyz', 'target_timestamps', camera_timestamps # NOTE: need to adapt to your path
    )
    save_zarr_array(episode_group, "action0_tcp_xyz_wxyz", action0_tcp, (1000, 7))

    action0_gripper = load_and_resample_1d(
        gripper_zarr, 'target_joint_pos', 'target_timestamps', camera_timestamps
    )
    save_zarr_array(episode_group, "action0_gripper_width", action0_gripper, (1000, 1))

    print(f"  Converted episode with {T} timesteps")


def main():
    parser = argparse.ArgumentParser(description="Convert robologger data to zarr format")
    parser.add_argument("--input_dir", required=True, help="Input directory with robologger episodes")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted data")
    parser.add_argument("--dataset_name", required=True, help="Name of the output dataset")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_zarr_dir = Path(args.output_dir).expanduser() / args.dataset_name / "episode_data.zarr"
    output_zarr_dir.mkdir(parents=True, exist_ok=True)

    zarr_root = zarr.open_group(str(output_zarr_dir), mode='a')

    episodes = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    print(f"Found {len(episodes)} episodes to convert")
    print(f"Output: {output_zarr_dir}")

    for episode_dir in tqdm(episodes, desc="Converting episodes"):
        episode_num = int(episode_dir.name.split("_")[1])
        try:
            episode_group = zarr_root.create_group(f"episode_{episode_num}", overwrite=True)
            convert_episode(episode_dir, episode_group)
        except Exception as e:
            print(f"\nError converting {episode_dir.name}: {e}")

    print(f"\nConversion complete. Dataset saved to: {output_zarr_dir}")
    print(f"\nConfig settings:")
    print(f"  root_dir: {output_zarr_dir.parent.parent}")
    print(f"  name: {args.dataset_name}")


if __name__ == "__main__":
    main()