import os
from typing import Any, Dict, List, Optional
from loguru import logger
from robotmq import RMQClient, deserialize, serialize
from robologger.loggers.base_logger import BaseLogger
from robologger.utils.classes import Morphology
from robologger.utils.stdout_setup import setup_logging
from atexit import register
import shutil

class MainLogger:
    """Main logger coordinating multiple sub-loggers."""
    def __init__(
        self,
        name: str,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        logger_endpoints: Dict[str, str], # {logger_name: logger_endpoint}
        morphology: Morphology,
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

        for logger_name, logger_endpoint in logger_endpoints.items():
            self.clients[logger_name] = RMQClient(client_name=logger_name, server_endpoint=logger_endpoint)

        self.episode_idx: int = -1
        self.is_recording: bool = False

        self._init_metadata()

        register(self.on_exit)

    def _init_metadata(self):
        self.zarr_group = zarr.open_group(os.path.join(self.run_dir, "metadata.zarr"), mode="w")
        assert self.zarr_group is not None, "Zarr group is not initialized"
        logger.info(f"Initialized zarr group: {os.path.join(self.run_dir, 'metadata.zarr')}")

    def _store_metadata(self):
        self.zarr_group.attrs["project_name"] = self.project_name
        self.zarr_group.attrs["task_name"] = self.task_name
        self.zarr_group.attrs["run_name"] = self.run_name
        self.zarr_group.attrs["morphology"] = self.morphology

        # TODO: pass in from after recording stop to update these two attributes
        self.zarr_group.attrs["is_demonstration"] = self.is_demonstration 
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

        for logger_name, logger_endpoint in self.logger_endpoints.items():
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