import os
import re
from typing import Dict, List, Optional, Union
from loguru import logger
from robotmq import RMQClient, deserialize, serialize
from robologger.utils.classes import Morphology, CameraName
from robologger.utils.stdout_setup import setup_logging
from atexit import register
import shutil
import zarr

class MainLogger:
    """
    Main logger coordinating multiple sub-loggers.

    Args:
        success_config: Controls how episode success is determined after stop_recording():
            - "none": Does not set is_successful (no prompt, no value assigned)
            - "input_true": Prompt user with [Y/n], defaults to successful
            - "input_false": Prompt user with [y/N], defaults to failed
            - "hardcode_true": Always mark episodes as successful (no prompt)
            - "hardcode_false": Always mark episodes as failed (no prompt)
    """
    def __init__(
        self,
        name: str,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        logger_endpoints: Dict[str, str], # {logger_name: logger_endpoint}
        morphology: Morphology,
        is_demonstration: bool = False,
        success_config: str = "none",
        # attr: dict,
    ):
        setup_logging()
        
        if root_dir[0] != "/":
            root_dir = f"{os.getcwd()}/{root_dir}"
        logger.info(f"Root directory: {root_dir}")
        self.root_dir = root_dir
        self.project_name = project_name
        self.task_name = task_name
        self.run_name = run_name
        self.run_dir: str = os.path.join(self.root_dir, self.project_name, self.task_name, self.run_name)
        
        if not os.path.exists(self.run_dir):
            logger.info(f"Creating run directory: {self.run_dir}")
            os.makedirs(self.run_dir)
        self.clients: Dict[str, RMQClient] = {}

        self.logger_endpoints: Dict[str, str] = logger_endpoints

        self.morphology = morphology
        self.is_demonstration = is_demonstration

        # validate video logger names
        self._validate_video_logger_names()

        for logger_name, logger_endpoint in logger_endpoints.items():
            self.clients[logger_name] = RMQClient(client_name=logger_name, server_endpoint=logger_endpoint)

        self.episode_idx: int = -1
        self.last_episode_idx: Optional[int] = None  # track last COMPLETED episode
        self.is_recording: bool = False

        # init success tracking
        success_config_options = ["none", "input_true", "input_false", "hardcode_true", "hardcode_false"]
        assert success_config.lower() in success_config_options, f"Invalid success_config: {success_config}. Must be one of: {success_config_options}"
        self.success_config = success_config.lower()

        register(self.on_exit)


    def _init_metadata(self, episode_dir: str):
        metadata_path = os.path.join(episode_dir, "metadata.zarr")
        self.zarr_group = zarr.open_group(metadata_path, mode="w")
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Initialized zarr group: {metadata_path}")

    def _store_metadata(self, is_successful: Optional[bool]):
        self.zarr_group.attrs["project_name"] = self.project_name
        self.zarr_group.attrs["task_name"] = self.task_name
        self.zarr_group.attrs["run_name"] = self.run_name
        self.zarr_group.attrs["morphology"] = self.morphology.value
        self.zarr_group.attrs["is_demonstration"] = self.is_demonstration
        self.zarr_group.attrs["is_successful"] = is_successful

    def on_exit(self):
        if self.is_recording:
            self.stop_recording()

    def validate_logger_endpoints(self):
        for logger_name, client in self.clients.items():
            topic_status = client.get_topic_status(topic="info", timeout_s=0.1)
            if topic_status <= 0:
                raise RuntimeError(f"Logger {logger_name} is not alive")
            raw_data, _ = client.peek_data(topic="info", n=1)
            data = deserialize(raw_data[0])
            if data["name"] != logger_name:
                raise RuntimeError(f"Requesting endpoint {self.logger_endpoints[logger_name]}, should be {logger_name}, but got {data['name']}")

    def start_recording(self, episode_idx: Optional[int] = None):
        if self.is_recording:
            logger.warning("Already recording, stopping current recording")
            self.stop_recording()

        self.validate_logger_endpoints()
        self.is_recording = True

        if episode_idx is not None:
            self.episode_idx = episode_idx
        else:
            self.episode_idx = self._get_next_episode_idx()
        assert self.episode_idx >= 0, "Episode index must be non-negative"
        episode_dir = os.path.join(self.run_dir, f"episode_{self.episode_idx:06d}")
        logger.info(f"Starting episode {self.episode_idx} in {episode_dir}")

        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
            logger.info(f"Created episode directory: {episode_dir}")
        else:
            logger.info(f"Episode directory already exists, deleting existing directory and creating a new one: {episode_dir}")
            shutil.rmtree(episode_dir)
            os.makedirs(episode_dir)

        # init metadata for this episode
        self._init_metadata(episode_dir)
        
        for logger_name, _ in self.logger_endpoints.items():
            self.clients[logger_name].put_data(topic="command", data=serialize({"type": "start", "episode_dir": episode_dir}))

        return self.episode_idx

    def get_alive_loggers(self) -> List[str]:
        alive_loggers: List[str] = []
        for logger_name, client in self.clients.items():
            topic_status = client.get_topic_status(topic="info", timeout_s=0.1)
            if topic_status >= 0:
                alive_loggers.append(logger_name)
            else:
                logger.warning(f"Logger {logger_name} is not alive")
        return alive_loggers

    def stop_recording(self) -> Optional[int]:
        if not self.is_recording:
            logger.warning("Not recording, ignoring stop command in main logger")
            return None
        self.is_recording = False
        alive_loggers = self.get_alive_loggers()
        for logger_name in alive_loggers:
            self.clients[logger_name].put_data(topic="command", data=serialize({"type": "stop"}))
        episode_dir = os.path.join(self.run_dir, f"episode_{self.episode_idx:06d}")
        logger.info(f"Episode {self.episode_idx} stopped recording for {len(alive_loggers)} loggers. Data has been saved to {episode_dir}")
        is_successful = self._get_is_successful()
        self._store_metadata(is_successful)
        self.last_episode_idx = self.episode_idx  # Track COMPLETED episode
        return self.episode_idx

    def _get_next_episode_idx(self) -> int:
        """Find the next available episode index."""
        if not os.path.exists(self.run_dir):
            return 0
        
        existing_episodes = []
        for item in os.listdir(self.run_dir):
            if item.startswith("episode_") and os.path.isdir(os.path.join(self.run_dir, item)):
                try:
                    episode_num = int(item.split("_")[1])
                    existing_episodes.append(episode_num)
                except (IndexError, ValueError):
                    logger.warning(f"Invalid episode directory name: {item}")
                    continue
        
        return max(existing_episodes, default=-1) + 1

    def _get_is_successful(self) -> Union[None, bool]:
        """Handle success setting based on configured mode."""
        if self.success_config.startswith("hardcode"):
            is_successful = self.success_config == "hardcode_true"
            logger.info(f"[MainLogger] Episode {self.episode_idx} marked as {'successful' if is_successful else 'failed'} (hardcoded).")
        elif self.success_config.startswith("input"):
            is_successful = self._prompt_for_success()
            logger.info(f"[MainLogger] Episode {self.episode_idx} marked as {'successful' if is_successful else 'failed'} (user input).")
        elif self.success_config == "none":
            # if success_config was None, don't set is_successful
            is_successful = None
            # logger.info("[MainLogger] Success mode is none, not setting is_successful. To set episode success manually, use set_successful(bool) method.")
            logger.info(f"[MainLogger] Episode {self.episode_idx} success unset.")
        else:
            raise ValueError(f"Invalid success_config: {self.success_config}")

        return is_successful

    def _prompt_for_success(self):
        """Prompt user for episode success status."""
        if self.success_config.endswith("true"):
            prompt = "Was this episode successful? (input might not show) [Y/n]: "
            default_response = "Y"
        else:
            prompt = "Was this episode successful? (input might not show) [y/N]: "
            default_response = "N"
        
        while True:
            try:
                user_input = input(prompt).strip()
                
                # handle empty input (use default)
                if not user_input:
                    user_input = default_response
                
                # process response
                if user_input.lower() in ['y', 'yes']:
                    logger.info("Episode marked as successful")
                    return True
                elif user_input.lower() in ['n', 'no']:
                    logger.info("Episode marked as failed")
                    return False
                else:
                    logger.info("Please enter 'y' for yes or 'n' for no.")
            except (EOFError, KeyboardInterrupt):
                # handle Ctrl+C or EOF gracefully
                logger.info("\nEpisode success status cancelled")
                return None

    def _is_video_logger(self, name: str) -> bool:
        """Check if logger name matches CameraName enum pattern (indicates video logger)."""
        pattern = re.compile(r'^(' + '|'.join(re.escape(cam.value) for cam in CameraName) + r')(\d+)$')
        return pattern.match(name) is not None

    def delete_last_episode(self) -> Optional[int]:
        """Delete the most recently completed episode.

        Returns:
            Episode index if deleted, None if no episode to delete or recording is active.
            Always check for None before using the returned value.
        """
        if self.is_recording:
            logger.warning("[Delete Last Episode] Cannot delete episode while recording is active")
            return None

        if self.last_episode_idx is None:
            if self._get_next_episode_idx() > 0:
                self.last_episode_idx = self._get_next_episode_idx() - 1
                logger.warning(f"[Delete Last Episode] Deleting the largest episode index {self.last_episode_idx}")
            else:
                logger.warning("[Delete Last Episode] There is no episode to delete, cannot delete")
                return None

        episode_dir = os.path.join(self.run_dir, f"episode_{self.last_episode_idx:06d}")

        if not os.path.exists(episode_dir):
            logger.warning(f"[Delete Last Episode] Episode {self.last_episode_idx} directory not found at {episode_dir}, cannot delete")
            self.last_episode_idx = None
            return None

        logger.info(f"[Delete Last Episode] Deleting episode {self.last_episode_idx}: {episode_dir}")

        # prompt for confirmation (default: yes)
        try:
            while True:
                response = input(f"Confirming deleting episode {self.last_episode_idx} at {episode_dir}? (input might not show) [Y/n]: ").strip().lower()
                if response == '' or response in ['y', 'yes']:
                    break
                elif response in ['n', 'no']:
                    logger.info(f"[Delete Last Episode] Deletion of episode {self.last_episode_idx} cancelled by user")
                    return None
                else:
                    print("Please enter 'y' for yes or 'n' for no (default is yes)")
        except (EOFError, KeyboardInterrupt):
            logger.info(f"[Delete Last Episode] Deletion of episode {self.last_episode_idx} cancelled by user")
            return None

        shutil.rmtree(episode_dir)

        deleted_idx = self.last_episode_idx
        self.last_episode_idx = None  # clear after deletion

        return deleted_idx

    def _validate_video_logger_names(self):
        """Validate all video logger names for pattern compliance and zero-indexed continuity."""
        video_loggers = {}  # {camera_base: [indices]}
        pattern = re.compile(r'^(' + '|'.join(re.escape(cam.value) for cam in CameraName) + r')(\d+)$')

        # group video loggers by camera base name
        for logger_name in self.logger_endpoints.keys():
            match = pattern.match(logger_name)
            if match:
                base_name, index = match.groups()
                if base_name not in video_loggers:
                    video_loggers[base_name] = []
                video_loggers[base_name].append(int(index))

        # validate zero-indexed continuity for each camera type
        for base_name, indices in video_loggers.items():
            indices.sort()
            expected = list(range(len(indices)))
            if indices != expected:
                logger_names = [f"{base_name}{i}" for i in indices]
                expected_names = [f"{base_name}{i}" for i in expected]
                raise ValueError(
                    f"Multiple VideoLoggers of type '{base_name}' must use zero-indexed continuous naming.\n"
                    f"Current loggers: {logger_names}\n"
                    f"Expected naming sequence: {expected_names}\n"
                    f"Please rename to follow continuous zero-indexing: {base_name}0, {base_name}1, {base_name}2, etc."
                )