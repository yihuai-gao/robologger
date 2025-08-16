import os
from loguru import logger
from robologger.loggers.base_logger import BaseLogger

class MainLogger(BaseLogger):
    def __init__(
        self,
        name: str,
        root_dir: str,
        project_name: str,
        task_name: str,
        run_name: str,
        attr: dict,
    ):

        super().__init__(name, attr)
        self.root_dir = root_dir
        self.project_name = project_name
        self.task_name = task_name
        self.run_name = run_name
        self.run_dir: str = os.path.join(self.root_dir, self.project_name, self.task_name, self.run_name)
        
        if not os.path.exists(self.run_dir):
            logger.info(f"Creating run directory: {self.run_dir}")
            os.makedirs(self.run_dir)

        self.episode_idx: int = -1

    def _init_storage(self):
        ...

    def _close_storage(self):
        ...