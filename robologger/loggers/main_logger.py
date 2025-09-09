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
            - None (default): Does not set is_successful (no prompt, no value assigned)
            - "input_true": Prompt user with [Y/n], defaults to successful
            - "input_false": Prompt user with [y/N], defaults to failed
            - True (bool or str): Always mark episodes as successful (no prompt)
            - False (bool or str): Always mark episodes as failed (no prompt)
            Note: Users can always override manually using set_successful(bool)
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
        success_config: Union[str, bool, None] = None,
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
        self.is_recording: bool = False

        # init success tracking
        self._parse_success_config(success_config)

        self._init_metadata()

        register(self.on_exit)

    def _parse_success_config(self, config):
        """Parse success configuration from single parameter."""
        if config is None:
            # success_config not passed in, default to none
            self.success_mode = "none"
        elif config == "input_true":
            # input mode with True default [Y/n]
            self.success_mode = "input"
            self.default_success = True
        elif config == "input_false":
            # input mode with False default [y/N]
            self.success_mode = "input"
            self.default_success = False
        elif config is True or config.lower() == "true":
            # hardcode all as successful
            self.success_mode = "hardcode"
            self.hardcode_success = True
        elif config is False or config.lower() == "false":
            # hardcode all as failed
            self.success_mode = "hardcode" 
            self.hardcode_success = False
        else:
            raise ValueError(
                f"Invalid success_config: {config}. Must be one of: "
                "None (default to none), 'input_true', 'input_false', True, or False\n"
                f"Got: {config}\n"
                f"Read docstring for more information."
            )

    def _init_metadata(self):
        self.zarr_group = zarr.open_group(os.path.join(self.run_dir, "metadata.zarr"), mode="w")
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Initialized zarr group: {os.path.join(self.run_dir, 'metadata.zarr')}")

    def _store_metadata(self):
        self.zarr_group.attrs["project_name"] = self.project_name
        self.zarr_group.attrs["task_name"] = self.task_name
        self.zarr_group.attrs["run_name"] = self.run_name
        self.zarr_group.attrs["morphology"] = self.morphology
        self.zarr_group.attrs["is_demonstration"] = self.is_demonstration 
        if self.success_mode is not None:
            self.zarr_group.attrs["is_sucessful"] = self.is_sucessful

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

        self.is_recording = True
        self.validate_logger_endpoints()

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

        for logger_name, _ in self.logger_endpoints.items():
            self.clients[logger_name].put_data(topic="command", data=serialize({"type": "start", "episode_dir": episode_dir}))

    def get_alive_loggers(self) -> List[str]:
        alive_loggers: List[str] = []
        for logger_name, client in self.clients.items():
            topic_status = client.get_topic_status(topic="info", timeout_s=0.1)
            if topic_status >= 0:
                alive_loggers.append(logger_name)
            else:
                logger.warning(f"Logger {logger_name} is not alive")
        return alive_loggers

    def stop_recording(self):
        if not self.is_recording:
            raise RuntimeError("Not recording, but received stop command in main logger")
        self._handle_success_setting()
        self._store_metadata()
        self.is_recording = False
        alive_loggers = self.get_alive_loggers()
        for logger_name in alive_loggers:
            self.clients[logger_name].put_data(topic="command", data=serialize({"type": "stop"}))
            episode_dir = os.path.join(self.run_dir, f"episode_{self.episode_idx:06d}")
        logger.info(f"Stopped recording for {len(alive_loggers)} loggers. Data has been saved to {episode_dir}")

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

    def _handle_success_setting(self):
        """Handle success setting based on configured mode."""
        if self.success_mode == "hardcode":
            self.is_sucessful = self.hardcode_success
        elif self.success_mode == "input":
            self._prompt_for_success()
        elif self.success_mode == "none":
            # if success_config was None, don't set is_sucessful
            logger.info("[MainLogger] Success mode is none, not setting is_sucessful. To set episode success manually, use set_successful(bool) method.")
            return

    def _prompt_for_success(self):
        """Prompt user for episode success status."""
        if self.default_success:
            prompt = "Was this episode successful? [Y/n]: "
            default_response = "Y"
        else:
            prompt = "Was this episode successful? [y/N]: "
            default_response = "N"
        
        while True:
            try:
                user_input = input(prompt).strip()
                
                # handle empty input (use default)
                if not user_input:
                    user_input = default_response
                
                # process response
                if user_input.lower() in ['y', 'yes']:
                    self.is_sucessful = True
                    break
                elif user_input.lower() in ['n', 'no']:
                    self.is_sucessful = False
                    break
                else:
                    logger.warning("Please enter 'y' for yes or 'n' for no.")
            except (EOFError, KeyboardInterrupt):
                # handle Ctrl+C or EOF gracefully
                logger.warning(f"\nUsing default response: {self.default_success}")
                self.is_sucessful = self.default_success
                break

    def set_successful(self, is_successful: bool):
        """Manually set the success status for the current episode."""
        self.is_sucessful = is_successful
        logger.info(f"Episode {self.episode_idx} success status manually set to: {is_successful}")

    def _is_video_logger(self, name: str) -> bool:
        """Check if logger name matches CameraName enum pattern (indicates video logger)."""
        pattern = re.compile(r'^(' + '|'.join(re.escape(cam.value) for cam in CameraName) + r')(\d+)$')
        return pattern.match(name) is not None

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