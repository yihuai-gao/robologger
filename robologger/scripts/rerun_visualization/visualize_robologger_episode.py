#!/usr/bin/env python3
"""Simple script to visualize a specific RoboLogger episode."""

import sys
import argparse
from pathlib import Path
from real_env.visualization.rerun_visualizer import UMIVisualizer


def visualize_episode(episode_path: str, max_frames: int = 500, video_rate: int = 3):
    """
    Visualize a RoboLogger episode.

    Args:
        episode_path: Path to episode (can be dataset root or specific episode)
        max_frames: Maximum frames to process
        video_rate: Video frame sampling rate (process every Nth frame)
    """
    episode_path = Path(episode_path)

    # Handle both formats:
    # 1. /path/to/robologger_data (dataset root)
    # 2. /path/to/robologger_data/episode_data.zarr/episode_0 (specific episode)

    if episode_path.name.startswith("episode_"):
        # User provided specific episode path
        # Go up to dataset root
        dataset_root = episode_path.parent.parent
        episode_num = int(episode_path.name.split("_")[1])
        print(f"Loading episode {episode_num} from {dataset_root}")
    else:
        # User provided dataset root
        dataset_root = episode_path
        episode_num = 0
        print(f"Loading episode {episode_num} from {dataset_root}")

    # Check if the dataset exists
    if not dataset_root.exists():
        print(f"Error: Dataset path does not exist: {dataset_root}")
        return 1

    # Check if episode exists
    zarr_path = dataset_root / "episode_data.zarr" / f"episode_{episode_num}"
    if not zarr_path.exists():
        print(f"Error: Episode {episode_num} does not exist at {zarr_path}")
        return 1

    print(f"\nInitializing visualizer...")
    print(f"  Dataset: {dataset_root}")
    print(f"  Episode: {episode_num}")
    print(f"  Max frames: {max_frames}")
    print(f"  Video rate: every {video_rate} frames")

    try:
        # Create visualizer - it will load episode_0 by default
        # TODO: Add support for selecting specific episodes
        viz = UMIVisualizer(
            dataset_root,
            max_frames=max_frames,
            video_rate=video_rate,
            dataset_type="robologger"
        )

        print("\n✓ Data loaded successfully!")
        print(f"\nDataset info:")
        print(f"  - Robot poses: {viz.data['ur5_poses'].shape[0]} timesteps")
        print(f"  - Duration: ~{viz.data['ur5_timestamps'][-1]:.2f} seconds")
        print(f"  - Video: {viz.data['total_frames']} frames @ {viz.data['video_fps']:.2f} FPS")

        if viz.data['ur5_targets'] is not None:
            print(f"  - Has commanded targets: Yes")
        else:
            print(f"  - Has commanded targets: No")

        print("\nStarting visualization...")
        print("The Rerun viewer will open in a new window.")
        print("Press Ctrl+C in this terminal to exit.\n")

        viz.visualize(app_id=f"RoboLogger Episode {episode_num}", spawn=True)

        return 0

    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize a RoboLogger episode using Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize episode_0 from dataset root
  python visualize_robologger_episode.py /home/jeff/real-env/robologger_data

  # Visualize specific episode
  python visualize_robologger_episode.py /home/jeff/real-env/robologger_data/episode_data.zarr/episode_0

  # Process all frames (slower but more detailed)
  python visualize_robologger_episode.py /home/jeff/real-env/robologger_data --max-frames 10000

  # Show every video frame (slower)
  python visualize_robologger_episode.py /home/jeff/real-env/robologger_data --video-rate 1
        """
    )

    parser.add_argument(
        "episode_path",
        type=str,
        help="Path to dataset root or specific episode directory"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to process for performance (default: 500)"
    )

    parser.add_argument(
        "--video-rate",
        type=int,
        default=3,
        help="Process every Nth video frame (default: 3)"
    )

    args = parser.parse_args()

    return visualize_episode(args.episode_path, args.max_frames, args.video_rate)


if __name__ == "__main__":
    sys.exit(main())