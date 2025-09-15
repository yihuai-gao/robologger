#!/usr/bin/env python3
"""
Usage:
    python aggregate_data.py --run_dir /path/to/run/directory
    python aggregate_data.py --run_dir /path/to/run/directory --output_name aggregated_data
    python aggregate_data.py --run_dir /path/to/run/directory --output_name aggregated_data --max_workers 12
"""

import os
import shutil
import argparse
import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import zarr

from robologger.utils.classes import CameraName
from robologger.utils.stdout_setup import setup_logging


class MetadataReader:
    """Utility class for reading zarr metadata."""
    
    @staticmethod
    def get_episode_success_status(metadata_zarr_path: str) -> Optional[str]:
        """
        Get episode success status from metadata.zarr.
        
        Returns:
            "successful" if is_successful is True
            "failed" if is_successful is False  
            "unlabeled" if is_successful attribute doesn't exist
            "corrupted" if metadata.zarr file is corrupted and unreadable
            None if metadata.zarr file doesn't exist
        """
        import os
        
        if not os.path.exists(metadata_zarr_path):
            return None
            
        try:
            group = zarr.open_group(metadata_zarr_path, mode='r')
            if 'is_successful' in group.attrs:
                return "successful" if group.attrs['is_successful'] else "failed"
            return "unlabeled"
        except Exception as e:
            logger.error(f"Failed to read metadata from {metadata_zarr_path}: {e}. Storing as corrupted.")
            return "corrupted_or_incomplete"
    
    @staticmethod
    def get_camera_configs(zarr_path: str) -> Dict[str, Dict]:
        """Get camera configurations from a video logger zarr file."""
        try:
            group = zarr.open_group(zarr_path, mode='r')
            return group.attrs.get('camera_configs', {})
        except Exception as e:
            logger.warning(f"Failed to read camera configs from {zarr_path}: {e}. Returning empty dict.")
            return {}


class CameraDetector:
    """Detects camera systems and extracts camera names."""
    
    def __init__(self):
        self.camera_patterns = [camera.value for camera in CameraName]
    
    def find_video_logger_zarrs(self, episode_dir: str) -> List[str]:
        """Find all video logger zarr directories in an episode."""
        zarr_dirs = []
        
        for item in os.listdir(episode_dir):
            item_path = os.path.join(episode_dir, item)
            if os.path.isdir(item_path) and item.endswith('.zarr'):
                if self._is_video_logger_zarr(item):
                    zarr_dirs.append(item_path)
        
        return sorted(zarr_dirs)
    
    def _is_video_logger_zarr(self, zarr_name: str) -> bool:
        """Check if zarr name matches video logger pattern."""
        name = zarr_name.replace('.zarr', '')
        
        for pattern in self.camera_patterns:
            if name.startswith(pattern):
                # e.g., right_wrist_camera_0, 1, 2, etc.
                suffix = name[len(pattern):]
                if suffix.isdigit():
                    return True
        
        return False
    
    def get_video_files_from_zarr(self, zarr_path: str) -> List[Tuple[str, str, str]]:
        """
        Get all video files from a zarr directory.
        
        Returns:
            List of (camera_system_name, camera_name, video_file_path) tuples
        """
        video_files = []
        camera_configs = MetadataReader.get_camera_configs(zarr_path)
        
        # extract camera system name from zarr path (e.g., "right_wrist_camera_0")
        zarr_name = os.path.basename(zarr_path).replace('.zarr', '')
        
        for camera_name in camera_configs.keys():
            video_file = os.path.join(zarr_path, f"{camera_name}.mp4")
            if os.path.exists(video_file):
                video_files.append((zarr_name, camera_name, video_file))
            else:
                logger.warning(f"Video file not found: {video_file}; Skipping.")
        
        return video_files


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process_episode(self, episode_dir: str, episode_name: str, 
                       success_status: str, output_dir: str) -> None:
        """Process a single episode."""
        pass


class VideoProcessor(BaseProcessor):
    """Processor for aggregating video data."""
    
    def __init__(self):
        self.camera_detector = CameraDetector()
    
    def process_episode(self, episode_dir: str, episode_name: str,
                       success_status: str, output_dir: str) -> None:
        """Process video files from an episode."""
        status_dir = os.path.join(output_dir, success_status)
        os.makedirs(status_dir, exist_ok=True)
        
        video_zarrs = self.camera_detector.find_video_logger_zarrs(episode_dir)
        
        for zarr_path in video_zarrs:
            video_files = self.camera_detector.get_video_files_from_zarr(zarr_path)
                
            for camera_system_name, camera_name, video_file_path in video_files:
                # {episode_name}_{camera_system_name}_{camera_name}.mp4
                output_filename = f"{episode_name}_{camera_system_name}_{camera_name}.mp4"
                output_path = os.path.join(status_dir, output_filename)
                
                try:
                    shutil.copy2(video_file_path, output_path)
                    logger.info(f"Copied {video_file_path} -> {output_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {video_file_path}: {e}")


class DataAggregator:
    """Main class for aggregating run data."""
    
    def __init__(self, run_dir: str, output_name: str = "videos"):
        self.run_dir = Path(run_dir)
        self.output_name = output_name
        self.metadata_reader = MetadataReader()
        
        self.processors = {
            'video': VideoProcessor()
        }
    
    def aggregate(self, max_workers: int = 4) -> None:
        """Aggregate data from all episodes in the run."""
        logger.info(f"Starting data aggregation for run: {self.run_dir} with {max_workers} workers")
        
        # find all episodes
        episodes = self._find_episodes()
        if not episodes:
            logger.warning("No episodes found in run directory")
            return
        
        logger.info(f"Found {len(episodes)} episodes to process")
        
        # create output directory
        output_dir = self.run_dir / self.output_name
        output_dir.mkdir(exist_ok=True)
        
        # process episodes in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for episode_dir in episodes:
                episode_name = episode_dir.name
                
                # determine success status
                success_status = self._get_episode_success_status(episode_dir)
                
                # submit processing task
                future = executor.submit(
                    self._process_episode, 
                    str(episode_dir), 
                    episode_name, 
                    success_status, 
                    str(output_dir)
                )
                futures[future] = episode_name
            
            # wait for completion and handle errors
            for future in as_completed(futures):
                episode_name = futures[future]
                try:
                    future.result()
                    logger.info(f"Successfully processed episode: {episode_name}")
                except Exception as e:
                    logger.error(f"Episode {episode_name} processing failed: {e}")
        
        logger.info(f"Data aggregation completed. Output: {output_dir}")
    
    def _find_episodes(self) -> List[Path]:
        """Find all episode directories in the run."""
        episodes = []
        
        for item in self.run_dir.iterdir():
            if item.is_dir() and item.name.startswith("episode_"):
                episodes.append(item)
        
        return sorted(episodes)
    
    def _get_episode_success_status(self, episode_dir: Path) -> str:
        """Determine episode success status from episode-level metadata."""
        episode_metadata = episode_dir / "metadata.zarr"
        episode_name = episode_dir.name
        
        status = self.metadata_reader.get_episode_success_status(str(episode_metadata))
        
        if status is None:
            logger.error(f"No metadata information exists for {episode_name}. Storing as corrupted/incomplete.")
            return "corrupted_or_incomplete"
        
        return status
    
    def _process_episode(self, episode_dir: str, episode_name: str,
                        success_status: str, output_dir: str) -> None:
        """Process a single episode with all available processors."""
        # setup logging for multiprocessing worker
        setup_logging()
        
        logger.info(f"Processing {episode_name} (status: {success_status})")
        
        for processor_name, processor in self.processors.items():
            try:
                processor.process_episode(episode_dir, episode_name, success_status, output_dir)
            except Exception as e:
                logger.error(f"Processor {processor_name} failed for {episode_name}: {e}. Skipping.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate robologger episode data")
    parser.add_argument("--run_dir", required=True, help="Path to run directory")
    parser.add_argument("--output_name", default="videos", help="Output folder name")
    parser.add_argument("--max_workers", type=int, default=min(4, mp.cpu_count()), help="Max parallel workers")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # validate run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Validation failed: {run_dir} does not exist")
        raise ValueError(f"Run directory does not exist: {run_dir}")
    
    if not run_dir.is_dir():
        logger.error(f"Validation failed: {run_dir} is not a directory")
        raise ValueError(f"Run path is not a directory: {run_dir}")
    
    aggregator = DataAggregator(str(run_dir), args.output_name)
    logger.info(f"Running with {args.max_workers} workers")
    aggregator.aggregate(max_workers=args.max_workers)


if __name__ == "__main__":
    main()