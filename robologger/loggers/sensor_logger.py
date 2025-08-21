from robologger.loggers.base_logger import BaseLogger

class SensorLogger(BaseLogger):
    """Logger for sensor data."""
    
    def __init__(self, name: str, root_dir: str, project_name: str, task_name: str, run_name: str, attr: dict):
        """Initialize sensor logger."""
        super().__init__(name, root_dir, project_name, task_name, run_name, attr)

        # TODO: attr should include what kind of information is logged

    def _init_storage(self):
        """Initialize storage for sensor data."""
        ...

    def _close_storage(self):
        """Close sensor data storage."""
        ...
        
    def log(self, *, timestamp: float, data_dict: dict):
        """Log sensor data with timestamp."""
        super().log(timestamp=timestamp)
        # for key, value
        # TODO: