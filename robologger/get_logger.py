# Multiton pattern for loggers
from typing import Optional
from robologger.loggers.base_logger import LoggerType

def get_logger(
    name: str,
    logger_type: Optional[LoggerType],
    root_dir: Optional[str],
    project_name: Optional[str],
    task_name: Optional[str],
    run_name: Optional[str],
    attr: Optional[dict],
):
    pass